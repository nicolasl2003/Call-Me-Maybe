 # Makefile - Call-Me-Maybe

install:
	uv sync

run:
	uv run python -m src

clean:
	rm -rf .venv

lint:
	-uv run flake8 src
	-uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint_strict:
	-uv run flake8 src
	-uv run mypy src --strict

.PHONY: install run clean lint lint_strict