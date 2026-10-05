#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gh_push.py — 三级降级推送脚本 (作者: 晨星)

将当前目录推送到 GitHub 仓库 owner/name，三级降级保证在受限网络下仍可交付：

  L1  git push        : 标准 git 智能协议推送（gh 凭证助手）
  L2  Git Data API    : 经 gh api 创建 blobs/tree/commit/ref（绕过 git 协议代理 502）
  L3  Contents API    : 逐文件 PUT /contents 兜底（最终保底）

用法:
  python scripts/gh_push.py CJX0712/tensorforge --public --branch main
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", ".idea", ".mypy_cache"}
SKIP_EXTS = {".pyc"}
SKIP_FILES = set()  # benchmark*.json / failure_cases*.json 已在 .gitignore 排除，这里一并跳过
COMMIT_MSG = "TensorForge v0.1.0 — 世界顶级张量分解系统 (作者: 晨星)"


def run(cmd, input_text=None, env=None, check=False):
    """运行命令，返回 (rc, stdout, stderr)。"""
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        cmd,
        input=input_text,
        capture_output=True,
        text=True,
        env=e,
        shell=(os.name != "posix"),
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败 {cmd}: {p.stderr}")
    return p.returncode, p.stdout, p.stderr


def gh_api(method: str, path: str, payload: dict | None = None):
    """调用 gh api，返回解析后的 JSON。失败抛 RuntimeError。

    关键：gh api 仅在使用 --input - 时才从 stdin 读取 JSON 请求体，否则 body 为空。
    """
    cmd = ["gh", "api", "-X", method, path]
    inp = None
    if payload is not None:
        cmd += ["--input", "-"]
        inp = json.dumps(payload)
    rc, out, err = run(cmd, input_text=inp)
    if rc != 0:
        raise RuntimeError(f"gh api {method} {path} 失败: {err.strip()}")
    try:
        return json.loads(out) if out.strip() else {}
    except json.JSONDecodeError:
        return {"raw": out}


def list_files(root: str):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            if fn in SKIP_FILES:
                continue
            if any(fn.endswith(e) for e in SKIP_EXTS):
                continue
            if fn.startswith("benchmark") and fn.endswith(".json"):
                continue
            if fn.startswith("failure_cases") and fn.endswith(".json"):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            out.append((rel, full))
    out.sort()
    return out


# ── L1: git push ────────────────────────────────────────────────────────────
def push_git(owner: str, name: str, branch: str) -> bool:
    print("[L1] 尝试标准 git push ...")
    repo_url = f"https://github.com/{owner}/{name}.git"
    run(["git", "init", "-q"])
    run(["git", "config", "user.email", "tensorforge@morningstar.local"])
    run(["git", "config", "user.name", "晨星"])
    run(["gh", "auth", "setup-git"])  # 配置 gh 凭证助手
    run(["git", "add", "-A"])
    # 已提交过则跳过提交
    rc, _, _ = run(["git", "diff", "--cached", "--quiet"])
    if rc != 0:
        run(["git", "commit", "-q", "-m", COMMIT_MSG])
    run(["git", "branch", "-M", branch])
    rc, _, err = run(
        ["git", "remote", "add", "origin", repo_url],
        env={"GIT_TERMINAL_PROMPT": "0"},
    )
    if rc != 0:
        run(["git", "remote", "set-url", "origin", repo_url])
    rc, _, err = run(
        ["git", "push", "-u", "origin", branch],
        env={"GIT_TERMINAL_PROMPT": "0"},
    )
    if rc == 0:
        print(f"[L1] git push 成功 -> {repo_url}")
        return True
    print(f"[L1] git push 失败（可能代理拦截），降级到 L2。stderr 摘要: {err.strip()[:200]}")
    return False


# ── L2: Git Data API ─────────────────────────────────────────────────────────
def push_gitdata(owner: str, name: str, branch: str, files) -> bool:
    print(f"[L2] 尝试 GitHub Git Data API 推送（{len(files)} 个文件）...")
    repo = f"{owner}/{name}"
    # 获取 base commit（空仓库时无）
    base_sha = None
    rc, out, _ = run(["gh", "api", f"repos/{repo}/git/ref/heads/{branch}"])
    if rc == 0:
        try:
            base_sha = json.loads(out)["object"]["sha"]
        except Exception:
            base_sha = None

    # 创建 blobs
    tree_entries = []
    for rel, full in files:
        with open(full, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        blob = gh_api("POST", f"repos/{repo}/git/blobs", {"content": b64, "encoding": "base64"})
        tree_entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    # 创建 tree
    tree_payload = {"tree": tree_entries}
    if base_sha:
        tree_payload["base_tree"] = base_sha
    tree = gh_api("POST", f"repos/{repo}/git/trees", tree_payload)

    # 创建 commit
    commit_payload = {"message": COMMIT_MSG, "tree": tree["sha"]}
    if base_sha:
        commit_payload["parents"] = [base_sha]
    commit = gh_api("POST", f"repos/{repo}/git/commits", commit_payload)
    commit_sha = commit["sha"]

    # 更新/创建 ref
    if base_sha:
        gh_api("PATCH", f"repos/{repo}/git/refs/heads/{branch}", {"sha": commit_sha})
    else:
        gh_api("POST", f"repos/{repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": commit_sha})
    print(f"[L2] Git Data API 推送成功 -> https://github.com/{repo}")
    return True


# ── L3: Contents API 逐文件兜底 ─────────────────────────────────────────────
def push_contents(owner: str, name: str, branch: str, files) -> bool:
    print(f"[L3] 尝试 Contents API 逐文件兜底（{len(files)} 个文件）...")
    repo = f"{owner}/{name}"
    for rel, full in files:
        with open(full, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        # 查询是否已存在（拿 sha 用于更新）
        sha = None
        rc, out, _ = run(["gh", "api", f"repos/{repo}/contents/{rel}?ref={branch}"])
        if rc == 0:
            try:
                sha = json.loads(out).get("sha")
            except Exception:
                sha = None
        payload = {"message": COMMIT_MSG, "content": b64, "branch": branch}
        if sha:
            payload["sha"] = sha
        gh_api("PUT", f"repos/{repo}/contents/{rel}", payload)
    print(f"[L3] Contents API 兜底完成 -> https://github.com/{repo}")
    return True


def ensure_repo(owner: str, name: str, public: bool):
    repo = f"{owner}/{name}"
    rc, _, _ = run(["gh", "repo", "view", repo])
    if rc == 0:
        print(f"[repo] 已存在: {repo}")
        return
    print(f"[repo] 创建仓库: {repo} (public={public})")
    vis = "--public" if public else "--private"
    # --add-readme 生成初始提交，使仓库非空，便于 L2 Git Data API 一次性提交
    rc, _, err = run(
        ["gh", "repo", "create", repo, vis, "--add-readme", "--description", COMMIT_MSG, "--confirm"]
    )
    if rc != 0:
        # 某些 gh 版本要求 owner/repo 拆开
        rc2, _, err2 = run(
            ["gh", "repo", "create", name, vis, "--add-readme", "--description", COMMIT_MSG, "--confirm"]
        )
        if rc2 != 0:
            raise RuntimeError(f"仓库创建失败: {err.strip()} / {err2.strip()}")


def main():
    ap = argparse.ArgumentParser(description="gh_push 三级降级推送 (作者: 晨星)")
    ap.add_argument("repo", help="owner/name，例如 CJX0712/tensorforge")
    ap.add_argument("--public", action="store_true", help="公开仓库（默认私有）")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    if "/" in args.repo:
        owner, name = args.repo.split("/", 1)
    else:
        owner, name = "CJX0712", args.repo

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[error] 目录不存在: {root}")
        return 2

    # gh 登录校验
    rc, _, err = run(["gh", "auth", "status"])
    if rc != 0:
        print(f"[error] gh 未登录: {err.strip()}")
        return 3

    ensure_repo(owner, name, args.public)
    files = list_files(root)
    print(f"[files] 待推送 {len(files)} 个文件")

    # L1 → L2 → L3 逐级降级
    if push_git(owner, name, args.branch):
        return 0
    try:
        if push_gitdata(owner, name, args.branch, files):
            return 0
    except Exception as e:
        print(f"[L2] 异常: {e}")
    try:
        if push_contents(owner, name, args.branch, files):
            return 0
    except Exception as e:
        print(f"[L3] 异常: {e}")
    print("[fail] 三级推送均失败，请检查网络/权限。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
