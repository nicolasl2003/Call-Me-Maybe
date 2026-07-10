# Makefile - Call-Me-Maybe

install:
	uv sync

run:
	uv run python -m src

clean:
	rm -rf .venv



