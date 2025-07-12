# Installation

## Requirements

- Python 3.13 or higher
- pip or Poetry for package management

## Install from PyPI

### Using pip

```bash
pip install flext-core
```

### Using Poetry

```bash
poetry add flext-core
```

## Install from Source

### Clone the repository

```bash
git clone https://github.com/flext-sh/flext-core.git
cd flext-core
```

### Install with pip

```bash
pip install -e .
```

### Install with Poetry

```bash
poetry install
```

## Development Installation

For development, install with all optional dependencies:

### Using pip

```bash
pip install -e ".[dev,test,docs]"
```

### Using Poetry

```bash
poetry install --with dev,test,docs
```

## Verify Installation

```python
import flext_core
print(flext_core.__version__)
# Output: 0.6.0
```

## Next Steps

- Check out the [Quick Start](quickstart.md) guide
- Learn about [Core Concepts](concepts.md)
- Explore the [API Reference](../api/domain.md)
