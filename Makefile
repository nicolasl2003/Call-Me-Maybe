 # Makefile - Call-Me-Maybe

install:
	uv sync

run:
	uv run python -m src

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf data/output

lint:
	-uv run flake8 src
	-uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-uv run flake8 src
	-uv run mypy src --strict

.PHONY: install run clean lint lint-strict