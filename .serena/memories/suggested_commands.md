# FLEXT-CORE Essential Commands

## Quality Gates (MANDATORY)
```bash
make validate              # Complete validation pipeline (lint + type + security + test)
make check                 # Quick validation (lint + type-check only)
```

## Individual Quality Checks
```bash
make lint                  # Ruff linting with basic rules
make type-check            # MyPy strict mode checking (zero tolerance in src/)
make test                  # Full test suite (75% coverage minimum required)
make security              # Bandit + pip-audit security scanning
make format                # Auto-format code (79 char line limit)
make fix                   # Auto-fix linting issues
```

## Testing Commands
```bash
make test-unit             # Unit tests only (fast feedback)
make test-integration      # Integration tests only  
make test-fast             # Tests without coverage (quick iteration)
make coverage-html         # Generate HTML coverage report

# Specific test execution
PYTHONPATH=src poetry run pytest tests/unit/test_result.py -v
PYTHONPATH=src poetry run pytest tests/unit/test_container.py::TestFlextContainer::test_basic_registration -v
```

## Development Utilities
```bash
make setup                 # Complete dev environment setup
make shell                 # Python REPL with project loaded
make deps-show             # Show dependency tree
make deps-update           # Update all dependencies
make clean                 # Clean build artifacts
make reset                 # Full reset (clean + setup)
```

## Build and Deploy
```bash
make build                 # Build the package
make docs                  # Build documentation
make docs-serve            # Serve documentation locally
```

## Single Letter Aliases (Speed)
```bash
make t                     # test
make l                     # lint  
make f                     # format
make tc                    # type-check
make v                     # validate
```