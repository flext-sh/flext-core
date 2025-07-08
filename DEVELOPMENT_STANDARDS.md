# 🚀 FLEXT Core - Development Standards

## 100% PEP Strict Compliance Achieved

This project maintains **100% strict compliance** with all Python Enhancement Proposals (PEPs) and uses Poetry for complete dependency and tool management.

## 📋 Standards Overview

### ✅ PEP Compliance

- **PEP 8**: Style Guide for Python Code - Enforced via Black + Ruff
- **PEP 257**: Docstring Conventions - Google style enforced
- **PEP 484**: Type Hints - 100% type coverage with MyPy strict
- **PEP 517**: Build system - Poetry-based build backend
- **PEP 518**: pyproject.toml specification - Fully compliant
- **PEP 561**: Distributing and Packaging Type Information
- **PEP 621**: Project metadata in pyproject.toml

### 🛠️ Tool Stack (All Managed by Poetry)

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Poetry** | Dependency & environment management | `pyproject.toml` |
| **Black** | Code formatting (PEP 8) | Line length: 100 |
| **Ruff** | Linting (17 rule categories) | Strict mode |
| **isort** | Import sorting (PEP 8) | Black profile |
| **MyPy** | Type checking (PEP 484) | --strict mode |
| **Bandit** | Security scanning | Medium+ severity |
| **Safety** | Dependency vulnerabilities | All known CVEs |
| **Vulture** | Dead code detection | 80% confidence |
| **Radon** | Complexity analysis | CC/MI metrics |
| **pytest** | Testing framework | 90%+ coverage |
| **pre-commit** | Git hooks | All tools integrated |
| **commitizen** | Commit standards | Conventional commits |
| **MkDocs** | Documentation | Material theme |

## 🎯 Development Workflow

### 1. Environment Setup

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Complete setup (installs deps + pre-commit)
make setup
```

### 2. Daily Development

```bash
# Check current compliance status
make status

# Run all quality checks
make check

# Auto-fix all possible issues
make fix

# Run tests in watch mode
make test-watch

# Validate strict compliance
make validate
```

### 3. Pre-commit Integration

All quality checks run automatically on commit:

```bash
# Manual pre-commit run
make pre-commit-run

# Commit with conventional commits
git commit -m "feat: add new feature"
```

## 📊 Quality Gates

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make check` | Run ALL quality checks |
| `make lint` | Ruff linting (17 categories) |
| `make format-check` | Black + Ruff formatting |
| `make type-check` | MyPy strict mode |
| `make security` | Bandit + Safety + detect-secrets |
| `make complexity` | Radon CC/MI + Vulture |
| `make test` | pytest with 90%+ coverage |
| `make fix` | Auto-fix all issues |
| `make validate` | Validate 100% compliance |

### CI/CD Pipeline

GitHub Actions workflows enforce:

1. **Quality Job**: All linting, formatting, type checking
2. **Security Job**: Bandit, Safety, secret detection
3. **Test Job**: Full test suite with coverage
4. **Build Job**: Distribution building and validation
5. **Docs Job**: Documentation building

## 🔧 IDE Integration

### VSCode

Complete `.vscode/` configuration includes:

- `settings.json`: Python interpreter, formatters, linters
- `extensions.json`: Recommended extensions
- `launch.json`: Debug configurations
- `tasks.json`: Build and test tasks

### Cursor AI

`.cursor/` configuration for AI-assisted development:

- `settings.json`: AI context and rules
- `.cursorrules`: Project-specific AI guidelines

## 📁 Project Structure

```
flext-core/
├── .github/              # CI/CD workflows
│   ├── workflows/        # GitHub Actions
│   │   ├── ci.yml       # Main CI pipeline
│   │   └── release.yml  # Release automation
│   └── dependabot.yml   # Dependency updates
├── .vscode/             # VSCode configuration
├── .cursor/             # Cursor AI configuration
├── src/                 # Source code
│   └── flext_core/      # Main package
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation
├── pyproject.toml      # Poetry + tool configs
├── poetry.lock         # Locked dependencies
├── Makefile           # Development tasks
├── .pre-commit-config.yaml  # Git hooks
├── .cursorrules       # AI coding rules
└── README.md          # Project readme
```

## 🏆 Current Status

```
╔═══════════════════════════════════════════════════════════╗
║      ✅ 100% STRICT COMPLIANCE ACHIEVED!                 ║
╚═══════════════════════════════════════════════════════════╝

  Lint violations:  0
  Type errors:      0
  Security issues:  0
  
  All PEP standards: COMPLIANT
  All tools: POETRY-MANAGED
  All checks: AUTOMATED
```

## 🚨 Zero Tolerance Policy

This project maintains **ZERO TOLERANCE** for:

- ❌ Lint violations
- ❌ Type errors
- ❌ Security issues
- ❌ Formatting inconsistencies
- ❌ Import sorting issues
- ❌ Complexity violations
- ❌ Test coverage < 90%

Every commit must pass ALL checks with 0 violations.

## 📚 Documentation

- **API Docs**: Auto-generated from docstrings
- **Architecture**: Clean Architecture + DDD
- **Standards**: This document + pyproject.toml
- **AI Rules**: .cursorrules for consistency

## 🔄 Continuous Improvement

- **Weekly**: Dependency updates via Dependabot
- **Monthly**: Security audit of all dependencies
- **Quarterly**: Performance and complexity review

---

**Last Updated**: 2025-07-08
**Maintained By**: FLEXT Team
**Standards Version**: 1.0.0