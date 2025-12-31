# Contributing to Seqcore

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/seqcore/seqcore.git
cd seqcore
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
- **Ruff** for linting
- **MyPy** for type checking

Run all checks:
```bash
black seqcore tests
ruff check seqcore tests
mypy seqcore
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
