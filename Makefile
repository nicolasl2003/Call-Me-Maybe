# Makefile - Call-Me-Maybe

install:
	uv sync

run:
	uv run python -m src

clean:
	rm -rf .venv

lint:
	-uv run flake8 src
	-uv run mypy src

lint_strict:
	-uv run flake8 src
	-uv run mypy --strict src

.PHONY: install run clean lint lint_strict