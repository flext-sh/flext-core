"""Formatting Constants and Helper Functions.

Provides enterprise-grade constants and utility functions for data formatting,
validation, and type-safe operations following SOLID principles.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib

from flext_core import FlextUtilities, TEntityId

# =============================================================================
# FORMATTING CONSTANTS - Data conversion and size formatting
# =============================================================================

# Time formatting constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Grade threshold constants
GRADE_A_THRESHOLD = 90
GRADE_B_THRESHOLD = 80

# Byte conversion constants
BYTES_PER_KB = 1024  # Standard bytes per kilobyte for binary calculations

# Age categorization constants
YOUNG_ADULT_AGE_THRESHOLD = 25
ADULT_AGE_THRESHOLD = 40
MIDDLE_AGED_THRESHOLD = 60

# Discount validation constants
MAX_DISCOUNT_PERCENTAGE = 100

# =============================================================================
# UTILITY HELPER FUNCTIONS - To bridge missing methods
# =============================================================================


def generate_prefixed_id(prefix: str, length: int) -> TEntityId:
    """Generate prefixed ID with specified length using TEntityId."""
    base_id = FlextUtilities.generate_id()
    return f"{prefix}-{base_id[:length]}"


def generate_hash_id(data: str) -> TEntityId:
    """Generate hash-based ID from data using TEntityId."""
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def generate_short_id(length: int = 8) -> TEntityId:
    """Generate short ID with specified length - improved default."""
    base_id = FlextUtilities.generate_id()
    return base_id[:length]


def get_age_category(age_value: int) -> str:
    """Categorize age into groups with enterprise constants."""
    if age_value < YOUNG_ADULT_AGE_THRESHOLD:
        return "young_adult"
    elif age_value < ADULT_AGE_THRESHOLD:
        return "adult"
    elif age_value < MIDDLE_AGED_THRESHOLD:
        return "middle_aged"
    else:
        return "senior"
