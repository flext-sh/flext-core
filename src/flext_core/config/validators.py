"""Configuration validators for FLEXT projects.

This module provides validation functions for configuration values.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError


def validate_url(url: str) -> bool:
    """Validate if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_port(port: int) -> bool:
    """Validate if a number is a valid port."""
    return 1 <= port <= 65535


def validate_hostname(hostname: str) -> bool:
    """Validate if a string is a valid hostname."""
    if not hostname or len(hostname) > 253:
        return False

    # Check for valid characters
    pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$"
    return bool(re.match(pattern, hostname))


def validate_database_name(db_name: str) -> bool:
    """Validate database name."""
    if not db_name or len(db_name) > 63:
        return False

    # PostgreSQL naming rules
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    return bool(re.match(pattern, db_name))


def validate_project_name(name: str) -> bool:
    """Validate project name."""
    if not name or len(name) > 50:
        return False

    # Allow letters, numbers, hyphens, underscores
    pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
    return bool(re.match(pattern, name))


def validate_environment(env: str) -> bool:
    """Validate environment name."""
    valid_environments = {"development", "staging", "production", "test"}
    return env.lower() in valid_environments


def validate_log_level(level: str) -> bool:
    """Validate log level."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    return level.upper() in valid_levels


def validate_config_dict(config: dict[str, Any]) -> bool:
    """Validate configuration dictionary structure."""
    required_keys = {"project_name", "environment"}
    return all(key in config for key in required_keys)


def validate_pydantic_model(model: Any) -> bool:
    """Validate if an object is a valid Pydantic model."""
    try:
        # Try to validate the model
        if hasattr(model, "model_validate"):
            model.model_validate(model.model_dump())
        return True
    except ValidationError:
        return False
