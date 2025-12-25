"""Factory decorator discovery utilities.

This module provides factory discovery functionality that can be used by
container and decorators without creating circular dependencies.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._decorators.discovery import FactoryDecoratorsDiscovery

__all__ = ["FactoryDecoratorsDiscovery"]
