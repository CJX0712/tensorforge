.PHONY: install lint format test demo ci clean

install:
	pip install -r requirements.txt
	pip install ruff pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

test:
	python -m pytest -q -W ignore::UserWarning

demo:
	python examples/run_demo.py

ci: lint test demo

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	find . -name '*.pyc' -delete ; \
	rm -rf .pytest_cache .ruff_cache benchmark.json failure_cases.json
