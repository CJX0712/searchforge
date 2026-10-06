# SearchForge · 作者 晨星 (CJX0712)
.PHONY: install lint format test demo bench tune clean

install:
	python -m pip install -r requirements.lock.txt
	python -m pip install -e .

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest -q -W ignore::UserWarning --cov=searchforge --cov-report=term

demo:
	python examples/run_demo.py

bench:
	python -m searchforge.cli bench --seeds 42,43,44 --out benchmark.json

tune:
	python -m searchforge.cli tune --trials 16

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ .coverage
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
