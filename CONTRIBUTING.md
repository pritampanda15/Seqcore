# Contributing to Seqcore

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/pritampanda15/Seqcore.git
cd Seqcore
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=seqcore --cov-report=html
```

## Code Style

We use:
- **Black** for formatting
- **isort** for import ordering
- **Ruff** for linting
- **MyPy** for type checking

Run all checks (CI enforces the first three):
```bash
black --check seqcore/
isort --check-only seqcore/
ruff check .
mypy seqcore/ --ignore-missing-imports
```

To apply the formatting fixes rather than just checking:
```bash
black seqcore/
isort seqcore/
ruff check . --fix
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and add tests
4. Run tests and linting
5. Commit with clear message
6. Push and open a pull request

## Code Guidelines

- Follow PEP 8 style
- Add type hints to all functions
- Write docstrings (Google style)
- Add tests for new functionality
- Keep functions focused and small

## Adding New Features

1. Create module in appropriate subpackage
2. Add exports to `__init__.py`
3. Write tests
4. Update README if needed
5. Add an entry under `## [Unreleased]` in `CHANGELOG.md`

## Documentation

The Sphinx site lives in `docs/`. Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html -W --keep-going docs docs/_build/html
```

CI builds the docs with `-W`, so any Sphinx warning fails the build. If you add
a new module, add a corresponding page under `docs/api/`.

## Performance Claims

Any performance number quoted in the README or docs must be reproducible with
the scripts in `benchmarks/`. Report measurements as observed, including cases
where Seqcore is slower than the comparison library.
