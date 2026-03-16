# Default recipe
default: help

# Show available commands
help:
    @just --list

# Install all dependencies
install:
    uv sync --frozen --dev

# Run linter and formatter checks
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# Auto-fix linting and formatting issues
lint-fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# Run tests
test:
    uv run pytest

# Run tests with coverage report
test-cov:
    uv run pytest --cov=src --cov-report=term-missing

# Run all checks before committing (lint + tests)
check-commit: lint test

# Interactive conventional commit
commit:
    uv run cz commit
