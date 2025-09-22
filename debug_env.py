#!/usr/bin/env python3
"""Debug environment variables."""

import os
import sys
from typing import cast

sys.path.insert(0, 'src')

from flext_core import FlextLogger, FlextTypes

# Check current environment variables
print("=== Environment Variables ===")
print(f"FLEXT_LOG_LEVEL: {os.getenv('FLEXT_LOG_LEVEL')}")
print(f"PYTEST_CURRENT_TEST: {os.getenv('PYTEST_CURRENT_TEST')}")

# Check for any project-specific vars
for key in os.environ:
    if 'LOG_LEVEL' in key:
        print(f"{key}: {os.environ[key]}")

# Set pytest environment
os.environ['PYTEST_CURRENT_TEST'] = 'debug_test'

# Clear any conflicting env vars
if 'FLEXT_LOG_LEVEL' in os.environ:
    del os.environ['FLEXT_LOG_LEVEL']

# Test again
print("\n=== After Cleanup ===")
logger = FlextLogger(
    "test_clean",
    _level=cast("FlextTypes.Config.LogLevel", ""),
)
print(f"Empty string result after cleanup: {logger._level}")