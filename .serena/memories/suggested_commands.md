# FLEXT-DB-ORACLE Development Commands

## Essential Commands

### Setup & Installation
```bash
make setup                    # Complete project setup (install deps + pre-commit)
make install                  # Install dependencies only
make install-dev              # Install dev dependencies
```

### Quality Gates (MANDATORY)
```bash
make validate                 # Run all quality gates (lint + type + security + test)
make check                    # Quick health check (lint + type)
make lint                     # Ruff linting
make type-check              # MyPy strict type checking  
make security                # Bandit + pip-audit security scanning
make fix                     # Auto-fix issues
```

### Testing
```bash
make test                     # Run tests with 90% coverage requirement
make test-unit               # Unit tests only (fast)
make test-integration        # Integration tests with Oracle container
make test-e2e               # End-to-end tests
make test-fast              # Tests without coverage
make coverage-html          # Generate HTML coverage report
```

### Oracle Operations
```bash
make oracle-test            # Test Oracle connection
make oracle-connect         # Test Oracle connectivity
make oracle-schema          # Validate Oracle schema access
make oracle-validate        # Validate Oracle configuration
make oracle-operations      # Run all Oracle validations
```

### Development
```bash
make format                 # Format code (ruff format)
make build                  # Build package
make clean                  # Clean build artifacts
make shell                  # Open Python shell
make pre-commit            # Run pre-commit hooks
make diagnose              # Project diagnostics
make doctor                # Health check
```

### Short Aliases
```bash
make t                     # test
make l                     # lint  
make f                     # format
make tc                    # type-check
make c                     # clean
make i                     # install
make v                     # validate
```

## Oracle Container Commands
```bash
# Start Oracle XE 21c container
docker-compose -f docker-compose.oracle.yml up -d

# Test Oracle connectivity  
make oracle-connect

# Stop Oracle container
docker-compose -f docker-compose.oracle.yml down
```

## Key Quality Requirements
- **Zero tolerance**: All quality gates must pass
- **Coverage**: 90% minimum test coverage
- **Type Safety**: MyPy strict mode with zero errors
- **Security**: Bandit vulnerability scanning
- **FLEXT Compliance**: Must follow FLEXT architectural patterns