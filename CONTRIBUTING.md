# Contributing to Chimera

Thank you for your interest in contributing! This document covers how to set up
your environment, run tests, submit pull requests, and follow our code style.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Adding a New Feature](#adding-a-new-feature)
7. [Reporting Bugs](#reporting-bugs)
8. [Pull Request Guidelines](#pull-request-guidelines)
9. [Security Disclosures](#security-disclosures)

---

## Getting Started

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/<your-username>/chimera
cd chimera

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify everything works
pytest
```

---

## Project Structure

```
chimera/
├── chimera/          Source package — one module per responsibility
├── tests/            Unit and integration tests
├── examples/         Runnable usage examples
├── docs/             Markdown documentation
├── benchmarks/       Performance benchmarks
├── paper/            Academic paper PDF
└── .github/          CI / release workflows
```

---

## Development Workflow

```bash
# Create a feature branch
git checkout -b feat/my-feature

# Make changes, then lint and format
black chimera tests examples
ruff check chimera tests examples
mypy chimera

# Run the full test suite
pytest --cov=chimera

# Commit (uses Conventional Commits style)
git commit -m "feat(transform): add glottal tension layer"

# Push and open a Pull Request
git push origin feat/my-feature
```

---

## Code Style

- **Formatter:** [Black](https://github.com/psf/black) — `line-length = 100`. Run `black .` before committing.
- **Linter:** [Ruff](https://docs.astral.sh/ruff/) — run `ruff check .`.
- **Type annotations:** All public functions must have full PEP 484 annotations.
  Run `mypy chimera` to verify.
- **Docstrings:** Google-style docstrings on all public classes and functions.
- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).

---

## Testing

Tests live in `tests/` and use [pytest](https://pytest.org).

```bash
pytest                        # full suite
pytest -m "not slow"          # skip slow tests
pytest --cov=chimera          # with coverage report
pytest tests/test_keygen.py   # single file
```

Guidelines:
- Every new public function must have at least one unit test.
- Every bug fix must have a regression test.
- Tests must not write to disk outside `tmp_path` (pytest fixture).
- Mark tests that take > 5 s with `@pytest.mark.slow`.
- Mark tests requiring `sounddevice` with `@pytest.mark.realtime`.

---

## Adding a New Feature

1. **Open an issue first** — discuss the design before writing code.
2. Add the implementation to the appropriate module (or create a new one).
3. Export new public names from `chimera/__init__.py` and `__all__`.
4. Write tests achieving > 80 % branch coverage for the new code.
5. Add an example in `examples/` if the feature has a typical use case.
6. Update `docs/api_reference.md` and `CHANGELOG.md` (under `[Unreleased]`).

---

## Reporting Bugs

Please open a [GitHub issue](https://github.com/Ohswedd/chimera/issues) with:

- Python version and OS.
- Chimera version (`python -c "import chimera; print(chimera.__version__)"`).
- Minimal reproducible example.
- Full traceback.

---

## Pull Request Guidelines

- PRs must target the `main` branch.
- All CI checks must pass before merge.
- At least one approval from a maintainer is required.
- Squash-merge is preferred for small features; merge-commit for large ones.

---

## Security Disclosures

Please **do not** open a public issue for security vulnerabilities.
Instead, use [GitHub's private vulnerability reporting](https://github.com/Ohswedd/chimera/security/advisories/new).
