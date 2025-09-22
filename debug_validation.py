#!/usr/bin/env python3
"""Debug Pydantic validation behavior."""

import sys
sys.path.insert(0, 'src')

from flext_core import FlextModels

print("=== Testing Pydantic Validation ===")

# Test 1: Invalid level
try:
    model1 = FlextModels.LoggerInitializationModel(
        name="test",
        log_level="INVALID_LEVEL"
    )
    print(f"INVALID_LEVEL validation: SUCCESS, log_level={model1.log_level}")
except Exception as e:
    print(f"INVALID_LEVEL validation: FAILED - {e}")

# Test 2: Empty string
try:
    model2 = FlextModels.LoggerInitializationModel(
        name="test",
        log_level=""
    )
    print(f"Empty string validation: SUCCESS, log_level={model2.log_level}")
except Exception as e:
    print(f"Empty string validation: FAILED - {e}")

# Test 3: Empty string with or logic
try:
    level = "" or "INFO"
    print(f"Empty string or INFO: {level}")
    model3 = FlextModels.LoggerInitializationModel(
        name="test",
        log_level=level
    )
    print(f"Empty string or INFO validation: SUCCESS, log_level={model3.log_level}")
except Exception as e:
    print(f"Empty string or INFO validation: FAILED - {e}")