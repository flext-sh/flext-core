"""Domain utilities for FLEXT components.

This module consolidates utility functions to eliminate duplication
across different projects.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pydantic import EmailStr


def normalize_email(email: EmailStr) -> EmailStr:
    """Normalize email address to lowercase and strip whitespace.

    Args:
        email: The email address to normalize.

    Returns:
        The normalized email address.

    """
    return email.lower().strip()


def is_expired(expires_at: Any) -> bool:
    """Check if token is expired.

    Args:
        expires_at: The expiration timestamp.

    Returns:
        True if expired, False otherwise.

    """
    return bool(datetime.now(UTC) > expires_at)


def validate_token_format(token: str) -> bool:
    """Validate basic token format.

    Args:
        token: The token string to validate.

    Returns:
        True if format is valid, False otherwise.

    """
    if not token or not isinstance(token, str):
        return False

    # Basic JWT format validation (3 parts separated by dots)
    parts = token.split(".")
    return len(parts) == 3 and all(part for part in parts)
