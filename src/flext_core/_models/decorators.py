"""Decorator configuration models for FLEXT decorators.

This module contains configuration models for decorators that require
structured validation and serialization. Simple decorators (inject, log_operation,
railway, combined) do not need models and use built-in types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextModelsDecorators:
    """Decorator configuration model container class.

    This class acts as a namespace container for decorator configuration models.
    All nested classes are accessed via FlextModels.Decorator.* in the main models.py.
    """
