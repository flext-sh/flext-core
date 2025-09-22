#!/usr/bin/env python3
"""Debug FlextConfig behavior."""

import os
import sys
from typing import cast

sys.path.insert(0, 'src')

from flext_core import FlextConfig, FlextConstants

# Set pytest environment
os.environ['PYTEST_CURRENT_TEST'] = 'debug_test'

print("=== FlextConfig Debug ===")
config = FlextConfig.get_global_instance()
print(f"config.log_level: {config.log_level}")
print(f"config.environment: {config.environment}")

valid_levels = FlextConstants.Logging.VALID_LEVELS
print(f"valid_levels: {valid_levels}")

cfg_level = str(config.log_level).upper()
print(f"cfg_level: {cfg_level}")
print(f"cfg_level in valid_levels: {cfg_level in valid_levels}")

default_fallback = (
    cfg_level
    if cfg_level in valid_levels
    else FlextConstants.Logging.DEFAULT_LEVEL
)
print(f"default_fallback: {default_fallback}")
print(f"FlextConstants.Logging.DEFAULT_LEVEL: {FlextConstants.Logging.DEFAULT_LEVEL}")

# Check what happens when resolved_level is None
print(f"\nStep 4 logic when resolved_level is None:")
print(f"Would set resolved_level to: {default_fallback}")