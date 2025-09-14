# Installation Guide

Professional installation guide for FLEXT-Core foundation library.

## System Requirements

### Minimum Requirements

- **Python 3.13+**: Required for modern type annotations
- **pip 23.0+** or **Poetry 1.7+**: Package management
- **Git 2.0+**: For development installation

### Verify Prerequisites

```bash
# Check Python version
python --version  # Must show 3.13.0 or higher

# Check pip version
pip --version

# Check Poetry (if using)
poetry --version
```

## Installation Methods

### Development Installation (Source)

```bash
# Clone the repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Install with Poetry (recommended)
poetry install --with dev,test
poetry shell

# Or install with pip
pip install -e ".[dev,test]"

# Run setup
make setup  # Installs pre-commit hooks and tools
```

### Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install FLEXT-Core
pip install -e .

# Verify installation
python -c "from flext_core import FlextResult; print('FLEXT-Core ready')"
```

## Verification

### Quick Verification

```python
# test_installation.py
from flext_core import FlextResult, FlextContainer

def test_installation():
    """Test core functionality."""
    # Test FlextResult
    result = FlextResult[str].ok("Installation successful!")
    assert result.success
    print(f"✅ FlextResult: {result.unwrap()}")

    # Test FlextContainer
    container = FlextContainer.get_global()
    container.register("test_service", "test_value")

    service_result = container.get("test_service")
    assert service_result.success
    assert service_result.unwrap() == "test_value"
    print("✅ FlextContainer: Working")

    print("🎉 FLEXT-Core installed successfully!")

if __name__ == "__main__":
    test_installation()
```

### Import Verification

```python
# verify_imports.py
"""Verify all FLEXT-Core imports are working."""

def verify_core_imports():
    """Test core pattern imports."""
    from flext_core import (
        FlextResult,
        FlextContainer,
        FlextConfig,
        FlextModels,
        FlextLogger,
    )

    # Test FlextResult
    result = FlextResult[str].ok("Success")
    assert result.success
    assert result.unwrap() == "Success"
    print("✅ FlextResult working")

    # Test FlextContainer
    container = FlextContainer.get_global()
    container.register("test", "value")
    test_result = container.get("test")
    assert test_result.success
    print("✅ FlextContainer working")

    print("✅ All core imports verified")

if __name__ == "__main__":
    print("Verifying FLEXT-Core installation...")
    verify_core_imports()
    print("🎉 All imports verified successfully!")
```

## Development Environment Setup

### Essential Development Commands

```bash
# Quality gates (run before committing)
make validate     # All checks (lint + type + test)
make check        # Quick check (lint + type only)
make test         # Run test suite with coverage

# Code quality
make lint         # Ruff linting
make type-check   # MyPy type checking
make format       # Auto-format code

# Development utilities
make clean        # Clean build artifacts
make docs         # Build documentation
make deps-show    # Show dependency tree
```

## Project Structure for FLEXT Projects

### Recommended Structure

```
my_flext_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── domain/              # Domain layer
│       │   ├── entities.py      # FlextModels.Entity
│       │   └── value_objects.py # FlextModels.Value
│       ├── application/         # Application layer
│       │   ├── commands.py      # Business operations
│       │   └── handlers.py      # Command handlers
│       ├── infrastructure/      # Infrastructure layer
│       │   ├── repositories.py  # Data persistence
│       │   └── database.py      # Database configuration
│       └── config.py            # FlextConfig settings
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── conftest.py             # Pytest fixtures
├── pyproject.toml               # Poetry configuration
├── Makefile                     # Development commands
└── README.md
```

### Configuration Template

```python
# src/my_project/config.py
"""Application configuration using FLEXT-Core."""

from flext_core import FlextConfig
from pydantic import Field

class AppConfig(FlextConfig):
    """Main application configuration."""
    app_name: str = Field("My FLEXT App", description="Application name")
    version: str = Field("0.9.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")

    # Database settings
    database_url: str = Field("sqlite:///app.db")

    # API settings
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)

    class Config:
        env_prefix = "APP_"
        env_file = ".env"

# Global configuration instance
app_config = AppConfig()
```

## Usage Patterns

### Railway Pattern

```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult.fail("Empty data")
    return FlextResult.ok(data.upper())

# Usage
result = process_data("hello")
if result.success:
    print(f"Processed: {result.unwrap()}")
else:
    print(f"Error: {result.error}")
```

### Domain Modeling

```python
from flext_core import FlextModels, FlextResult
from datetime import datetime

class User(FlextModels.Entity):
    """User entity with business logic."""
    username: str
    email: str
    created_at: datetime
    is_active: bool = True

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult.fail("Already active")
        self.is_active = True
        return FlextResult.ok(None)

# Usage
user = User(
    id="user_123",
    username="johndoe",
    email="john@example.com",
    created_at=datetime.now()
)

activation = user.activate()
if activation.success:
    print("User activated")
```

## Troubleshooting

### Python Version Error

**Problem:**
```
ERROR: Python 3.13+ required
```

**Solution:**
```bash
# Install Python 3.13 using pyenv
pyenv install 3.13.0
pyenv local 3.13.0

# Or use system package manager
sudo apt install python3.13  # Ubuntu/Debian
brew install python@3.13     # macOS
```

### Import Errors

**Problem:**
```python
ImportError: cannot import name 'FlextResult' from 'flext_core'
```

**Solutions:**
```bash
# Verify installation
pip show flext-core

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall
pip uninstall flext-core -y
pip install -e .

# If using Poetry
poetry cache clear pypi --all
poetry install
```

### Type Checking Issues

**Problem:**
```
error: Module "flext_core" has no attribute "FlextResult"  [attr-defined]
```

**Solution:**
```bash
# Install type stubs
pip install types-pydantic

# Configure mypy (create mypy.ini)
[mypy]
plugins = pydantic.mypy
ignore_missing_imports = True
```

## IDE Setup

### VS Code Configuration

```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "ruff",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [79]
}
```

### PyCharm Setup

1. Set Python interpreter to 3.13+
2. Enable type checking: Settings → Editor → Inspections → Python → Type checker
3. Configure line length: Settings → Editor → Code Style → Python → Hard wrap at 79
4. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Learn basic usage patterns
2. **[API Reference](../api/core.md)** - Explore the complete API
3. **[Examples](../examples/overview.md)** - See practical examples
4. **[Development Guide](../development/best-practices.md)** - Development guidelines

---

**Installation completed!** FLEXT-Core is ready for building robust, type-safe applications with railway-oriented programming and domain-driven design patterns.