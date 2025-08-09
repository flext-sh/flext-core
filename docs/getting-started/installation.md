# FLEXT Core Installation Guide

**Installation and setup guide based on actual implementation**

## 🎯 System Requirements

### Required

- **Python 3.13+** (library requires modern Python features)
- **pip** or **Poetry** for dependency management
- **Git** for version control (development)

### Python Version Check

```bash
python --version  # Must be 3.13+
pip --version     # Any recent version
```

## 📦 Installation Methods

### 1. Poetry Installation (Recommended)

```bash
# Install FLEXT Core
poetry add flext-core

# For development
poetry install --with dev,test
```

### 2. pip Installation

```bash
# Basic installation
pip install flext-core

# In virtual environment (recommended)
python -m venv flext-env
source flext-env/bin/activate  # Linux/Mac
pip install flext-core
```

### 3. Development Installation

```bash
# Clone and setup
git clone <repository-url>
cd flext-core

# Install with Poetry
poetry install --with dev,test
poetry shell

# Setup development environment
make setup
```

## 🔧 Verification

### 1. Basic Functionality Test

```python
# test_installation.py
from flext_core import FlextResult, FlextContainer

def test_installation():
    """Test core functionality."""
    # Test FlextResult
    result = FlextResult.ok("Installation successful!")
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

### 2. Available Imports Test

```python
# verify_imports.py - Test REAL available imports
from flext_core import (
    # Core patterns (WORKING)
    FlextResult,
    FlextContainer,
    FlextSettings,

    # Domain patterns (AVAILABLE)
    FlextValueObject,
    FlextAggregateRoot,

    # Commands (AVAILABLE)
    FlextCommands,

    # Configuration (WORKING)
    FlextConfig,
)
from flext_core.models import FlextEntity
from flext_core.validation import FlextValidators
from flext_core.utilities import FlextUtilities

def test_imports():
    """Test that all core imports work."""
    # FlextResult
    result = FlextResult.ok("test")
    assert result.success
    print("✅ FlextResult import working")

    # FlextContainer
    container = FlextContainer()
    print("✅ FlextContainer import working")

    # Configuration
    class TestSettings(FlextSettings):
        test_field: str = "default"

    settings = TestSettings()
    assert settings.test_field == "default"
    print("✅ FlextSettings import working")

    print("🎉 All core imports verified")

if __name__ == "__main__":
    test_imports()
```

## 🏗️ Development Setup

### 1. Development Commands

```bash
# Validation (MANDATORY before commit)
make validate     # Complete validation
make check        # Quick lint + type check
make test         # Run tests
make format       # Format code

# Individual commands
make lint         # Ruff linting
make type-check   # MyPy type checking
make security     # Security checks
```

### 2. Project Structure

```
my_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── domain/           # Domain models
│       ├── services/         # Application services
│       └── infrastructure/   # Technical implementations
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── pyproject.toml           # Poetry config
└── README.md
```

### 3. Configuration Example

```python
# config.py
from flext_core import FlextSettings

class AppSettings(FlextSettings):
    app_name: str = "My App"
    debug: bool = False
    database_url: str = "sqlite:///app.db"

    class Config:
        env_prefix = "APP_"

# Usage
settings = AppSettings()
print(f"App: {settings.app_name}")
```

## 🧪 Common Patterns

### 1. Railway Pattern

```python
from flext_core import FlextResult

def process_data(data: str) -> FlextResult[str]:
    if not data:
        return FlextResult.fail("Empty data")
    return FlextResult.ok(data.upper())

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

### 3. Domain Entity (models API)

```python
from flext_core.models import FlextEntity

class User(FlextEntity):
    def __init__(self, user_id: str, name: str, email: str):
        super().__init__(user_id)
        self.name = name
        self.email = email

# Usage
user = User("123", "John Doe", "john@example.com")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Python Version Error

```bash
ERROR: Python 3.13+ required
```

**Solution:** Install Python 3.13+

#### 2. Import Error

```python
ImportError: No module named 'flext_core'
```

**Solutions:**

```bash
# Check installation
pip list | grep flext-core

# Reinstall
pip uninstall flext-core
pip install flext-core
```

#### 3. MyPy Errors

MyPy may report errors during development. This is expected as the library is in active development.

### Health Check

```python
# health_check.py
import sys

def check_system():
    """Basic system health check."""
    print("🔍 FLEXT Core Health Check")

    # Python version
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 13):
        print("❌ Python 3.13+ required")
        return False

    # Test import
    try:
        import flext_core
        print(f"✅ FLEXT Core imported successfully")
        print(f"Version: {flext_core.__version__}")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    print("✅ System healthy")
    return True

if __name__ == "__main__":
    check_system()
```

## 📚 Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Basic usage patterns
2. **[Core API Reference](../api/core.md)** - Main API documentation
3. **[Architecture Overview](../architecture/overview.md)** - Design principles
4. **[Examples](../examples/overview.md)** - Practical examples

## ⚠️ Important Notes

- This guide reflects the **ACTUAL** implementation in src/flext_core/
- All import examples are **TESTED** against current code
- Some features mentioned in other docs may be in development
- For advanced features, check the source code directly

---

**FLEXT Core** - Foundation library for clean architecture patterns
