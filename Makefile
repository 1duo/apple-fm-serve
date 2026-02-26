.PHONY: sync fmt lint typecheck test check run

sync:
	uv sync --extra dev --extra apple

fmt:
	uv run --extra dev ruff format src tests
	uv run --extra dev ruff check --fix src tests

lint:
	uv run --extra dev ruff check src tests

typecheck:
	uv run --extra dev mypy src tests

test:
	uv run --extra dev pytest

check: lint typecheck test

run:
	./serve
