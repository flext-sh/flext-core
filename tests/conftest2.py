"""Pytest configuration for flext-core tests.

This file configures pytest to properly find and import flext_core modules.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Configure pytest markers
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
]
