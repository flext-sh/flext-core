"""Exceptions facade for flext-infra.

Extends flext-core exception hierarchy with infra-specific namespaces.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextExceptions


class FlextInfraExceptions(FlextExceptions):
    class Infra:
        pass


e = FlextInfraExceptions

__all__ = ["FlextInfraExceptions", "e"]
