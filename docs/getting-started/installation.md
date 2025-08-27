# Installation Guide

Step-by-step guide to installing and setting up FLEXT Core.

## System Requirements

### Minimum Requirements

- **Python 3.13+**: Required for modern type hints and features
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

### Option 1: Production Installation (PyPI)

```bash
# Using pip
pip install flext-core

# Using Poetry (recommended for projects)
poetry add flext-core

# With optional dependencies
pip install "flext-core[dev,test]"
```

### Option 2: Development Installation (Source)

```bash
# Clone the repository
git clone https://github.com/flext-sh/flext-core.git
cd flext-core

# Install with Poetry (recommended)
poetry install --with dev,test,docs
poetry shell

# Or install with pip
pip install -e ".[dev,test,docs]"

# Run setup
make setup  # Installs pre-commit hooks and tools
```

### Option 3: Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install FLEXT Core
pip install flext-core

# Verify installation
python -c "import flext_core; print(flext_core.__version__)"
```

## Verification

### Quick Verification

```python
# test_installation.py
from flext_core import FlextResult, FlextContainer

def test_installation():
    """Test core functionality."""
    # Test FlextResult
    result = FlextResult[None].ok("Installation successful!")
    assert result.success
    print(f"✅ FlextResult: {result.data}")

    # Test FlextContainer
    container = FlextContainer()
    reg_result = container.register("test_service", "test_value")
    assert reg_result.success

    get_result = container.get("test_service")
    assert get_result.success
    assert get_result.data == "test_value"
    print("✅ FlextContainer: Working")

    print("🎉 FLEXT Core installed successfully!")

if __name__ == "__main__":
    test_installation()
```

### Complete Import Verification

```python
# verify_imports.py
"""Verify all FLEXT Core imports are working."""

def verify_core_imports():
    """Test core pattern imports."""
from flext_core import (
        FlextResult,
        FlextContainer,
        get_flext_container,
        FlextConfig,
    )

    # Test FlextResult
    result = FlextResult[None].ok("Success")
    assert result.success
    assert result.unwrap() == "Success"
    print("✅ FlextResult working")

    # Test FlextContainer
    container = FlextContainer.get_global()
    container.register("test", "value")
    assert container.has("test")
    print("✅ FlextContainer working")

    # Test FlextConfig
    class Settings(FlextConfig):
        app_name: str = "test"

    settings = Settings()
    assert settings.app_name == "test"
    print("✅ FlextConfig working")

def verify_domain_imports():
    """Test domain pattern imports."""
from flext_core import (
        FlextEntity,
        FlextValue,
        FlextAggregateRoot,
    )

    # Test domain patterns are importable
    print("✅ Domain patterns available")

def verify_utility_imports():
    """Test utility imports."""
from flext_core.utilities import generate_id, generate_uuid
from flext_core import get_logger

    # Test utilities
    entity_id = generate_id("test")
    assert entity_id.startswith("test_")

    uuid = generate_uuid()
    assert len(uuid) == 36

    logger = get_logger(__name__)
    assert logger is not None

    print("✅ Utilities working")

if __name__ == "__main__":
    print("Verifying FLEXT Core installation...")
    verify_core_imports()
    verify_domain_imports()
    verify_utility_imports()
    print("🎉 All imports verified successfully!")
```

## Development Environment Setup

### Essential Development Commands

```bash
# Quality gates (run before committing)
make validate     # Run ALL checks (lint + type + security + test)
make check        # Quick check (lint + type only)
make test         # Run test suite with coverage

# Code quality
make lint         # Ruff linting with all rules
make type-check   # MyPy strict mode checking
make format       # Auto-format code
make security     # Security vulnerability scan

# Development utilities
make shell        # Open Python REPL with project loaded
make clean        # Clean build artifacts
make build        # Build distribution packages
make docs         # Build documentation

# Dependency management
make deps-show    # Show dependency tree
make deps-audit   # Audit dependencies for vulnerabilities
make deps-update  # Update all dependencies
```

### Recommended Project Structure

```
my_flext_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── domain/              # Domain layer (entities, value objects)
│       │   ├── __init__.py
│       │   ├── entities.py
│       │   ├── value_objects.py
│       │   └── aggregates.py
│       ├── application/         # Application layer (use cases)
│       │   ├── __init__.py
│       │   ├── commands.py
│       │   ├── queries.py
│       │   └── handlers.py
│       ├── infrastructure/      # Infrastructure layer (persistence, external)
│       │   ├── __init__.py
│       │   ├── repositories.py
│       │   ├── database.py
│       │   └── external_apis.py
│       └── config.py            # Configuration using FlextConfig
├── tests/
│   ├── unit/                    # Unit tests (isolated)
│   ├── integration/             # Integration tests
│   ├── conftest.py             # Pytest fixtures
│   └── test_e2e.py             # End-to-end tests
├── pyproject.toml               # Poetry configuration
├── .env.example                 # Example environment variables
├── Makefile                     # Development commands
└── README.md                    # Project documentation
```

### Project Configuration Template

```python
# src/my_project/config.py
"""Application configuration using FLEXT Core."""

from flext_core import FlextConfig
from pydantic import Field
from typing import Optional

class DatabaseConfig(FlextConfig):
    """Database configuration."""
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, ge=1, le=65535)
    name: str = Field("myapp", description="Database name")
    user: str = Field("postgres")
    password: str = Field("", description="Database password")

    @property
    def url(self) -> str:
        """Build database URL."""
        auth = f"{self.user}:{self.password}@" if self.password else f"{self.user}@"
        return f"postgresql://{auth}{self.host}:{self.port}/{self.name}"

    class Config:
        env_prefix = "DB_"

class AppConfig(FlextConfig):
    """Main application configuration."""
    app_name: str = Field("My FLEXT App", description="Application name")
    version: str = Field("0.9.0", description="Application version")
    debug: bool = Field(False, description="Debug mode")
    environment: str = Field("development", pattern="^(development|staging|production)$")

    # API settings
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000, ge=1, le=65535)
    api_workers: int = Field(1, ge=1, le=16)

    # Security
    secret_key: str = Field(..., min_length=32, description="Secret key for JWT")
    cors_origins: list[str] = Field(default_factory=list)

    # Optional features
    enable_metrics: bool = Field(False)
    enable_tracing: bool = Field(False)
    redis_url: Optional[str] = Field(None)

    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        case_sensitive = False

# Global configuration instances
database_config = DatabaseConfig()
app_config = AppConfig()
```

## Common Usage Patterns

### 1. Railway Pattern

```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult[None].fail("Empty data")
    return FlextResult[None].ok(data.upper())

# Usage
result = process_data("hello")
if result.success:
    print(f"Processed: {result.data}")
else:
    print(f"Error: {result.error}")
```

### 2. Dependency Injection

```python
from flext_core import FlextContainer

# Setup
container = FlextContainer()
container.register("config", {"db_url": "sqlite:///app.db"})

# Usage
config_result = container.get("config")
if config_result.success:
    config = config_result.data
    print(f"Database: {config['db_url']}")
```

### 3. Domain Modeling

```python
from flext_core import FlextEntity, FlextValue, FlextResult
from datetime import datetime

class Email(FlextValue):
    """Email value object."""
    address: str

    def __init__(self, **data):
        address = data.get('address', '')
        if '@' not in address:
            raise ValueError(f"Invalid email: {address}")
        data['address'] = address.lower()
        super().__init__(**data)

class User(FlextEntity):
    """User entity."""
    username: str
    email: Email
    created_at: datetime
    is_active: bool = True

    def activate(self) -> FlextResult[None]:
        """Activate user account."""
        if self.is_active:
            return FlextResult[None].fail("Already active")
        self.is_active = True
        return FlextResult[None].ok(None)

# Usage
email = Email(address="john@example.com")
user = User(
    id="user_123",
    username="johndoe",
    email=email,
    created_at=datetime.now()
)

activation = user.activate()
if activation.success:
    print("User activated")
```

## Troubleshooting

### Common Issues and Solutions

#### Python Version Error

**Problem:**

```
ERROR: Python 3.13+ required
flext-core requires Python >=3.13
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

#### Import Errors

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
pip install --no-cache-dir flext-core

# If using Poetry
poetry cache clear pypi --all
poetry install
```

#### Type Checking Issues

**Problem:**

```
error: Module "flext_core" has no attribute "FlextResult"  [attr-defined]
```

**Solution:**

```bash
# Install type stubs
pip install types-pydantic

# Configure mypy.ini
cat > mypy.ini << EOF
[mypy]
plugins = pydantic.mypy
ignore_missing_imports = True
EOF
```

#### Virtual Environment Issues

**Problem:**

```
Command 'poetry' not found or 'make' not working
```

**Solution:**

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install make (if missing)
sudo apt install build-essential  # Ubuntu/Debian
brew install make                  # macOS
```

### Comprehensive Health Check

```python
#!/usr/bin/env python3
"""FLEXT Core installation health check."""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version requirement."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 13):
        print("❌ Python 3.13+ required")
        return False

    print("✅ Python version OK")
    return True

def check_flext_core():
    """Check FLEXT Core installation."""
    try:
        import flext_core
        print(f"✅ FLEXT Core {flext_core.__version__} installed")

        # Test core functionality
    from flext_core import FlextResult
        result = FlextResult[None].ok("test")
        assert result.success
        print("✅ Core functionality working")

        return True
    except ImportError as e:
        print(f"❌ FLEXT Core not installed: {e}")
        return False
    except Exception as e:
        print(f"❌ FLEXT Core error: {e}")
        return False

def check_dependencies():
    """Check required dependencies."""
    required = ["pydantic", "pydantic_settings", "structlog"]
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} missing")

    return len(missing) == 0

def check_development_tools():
    """Check development tools (optional)."""
    tools = {
        "poetry": "poetry --version",
        "make": "make --version",
        "git": "git --version",
        "ruff": "ruff --version",
        "mypy": "mypy --version"
    }

    print("\nDevelopment tools:")
    for tool, command in tools.items():
        try:
            import shutil
            tool_bin = command.split()[0]
            if shutil.which(tool_bin):
                print(f"✅ {tool} available")
            else:
                print(f"⚠️  {tool} not available (optional)")
        except Exception:
            print(f"⚠️  {tool} not available (optional)")

def main():
    """Run complete health check."""
    print("FLEXT Core Health Check")
    print("=" * 40)

    checks = [
        ("Python version", check_python_version),
        ("FLEXT Core", check_flext_core),
        ("Dependencies", check_dependencies)
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        if not check_func():
            all_passed = False

    # Optional development tools
    check_development_tools()

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! FLEXT Core is ready to use.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)**: Learn basic usage patterns
2. **[Examples Guide](../examples/overview.md)**: See practical examples
3. **[API Reference](../api/core.md)**: Explore the complete API
4. **[Configuration Guide](../configuration/overview.md)**: Set up configuration
5. **[Best Practices](../development/best-practices.md)**: Development guidelines

## IDE Setup

### VS Code

```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "editor.formatOnSave": true,
    "editor.rulers": [79, 120]
}
```

### PyCharm

1. Set Python interpreter to 3.13+
2. Enable type checking: Settings → Editor → Inspections → Python → Type checker
3. Configure line length: Settings → Editor → Code Style → Python → Hard wrap at 79
4. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest

## Additional Resources

- **GitHub Repository**: Source code and issues
- **Documentation**: Full documentation at `/docs`
- **Examples**: Working examples in `/examples`
- **Tests**: Test examples in `/tests`

---

For support, please check the GitHub issues or documentation.
