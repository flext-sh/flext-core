"""Shared service foundation for flext-cli components.

Reads settings through the service runtime so every consumer uses
``settings`` as the single settings access point.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_cli import m, p, t
from flext_core import s


class FlextCliServiceBase[TDomainResult: p.Base = m.Cli.RuntimeStatus](
    s[TDomainResult]
):
    """Base class for flext-cli services with typed configuration access.

    Note: This is an abstract base class. Subclasses must implement the
    `execute` method from s.
    """


s = FlextCliServiceBase

__all__: t.MutableSequenceOf[str] = ["FlextCliServiceBase", "s"]
