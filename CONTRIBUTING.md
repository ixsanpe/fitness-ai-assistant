# Contributing

Thank you for your interest in contributing to fitness-ai-assistant!

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [just](https://just.systems/man/en/)
2. Make sure you have Python 3.11+ installed
3. Fork the repository
4. Clone your fork: `git clone https://github.com/YOUR-USERNAME/fitness-ai-assistant.git`
5. Install dependencies:

```bash
just install
```

6. Install pre-commit hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

## Development Workflow

1. Create a new branch from `main`:

```bash
git checkout -b feat/your-feature
```

2. Make your changes

3. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```bash
just commit
```

Or write the message manually following the format: `type(scope): description`
Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

4. Ensure linting and tests pass:

```bash
just check-commit
```

5. Submit a pull request to `main`

## Code Style

- We use `ruff` for linting and formatting
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public classes and functions

## Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality
3. Ensure CI passes
4. Maintainers will review your code
5. Address review feedback

## Available Commands

| Command | Description |
|---|---|
| `just install` | Install all dependencies |
| `just lint` | Check linting and formatting |
| `just lint-fix` | Auto-fix linting and formatting |
| `just test` | Run tests |
| `just test-cov` | Run tests with coverage |
| `just check-commit` | Run all checks (lint + tests) |
| `just commit` | Interactive conventional commit |
