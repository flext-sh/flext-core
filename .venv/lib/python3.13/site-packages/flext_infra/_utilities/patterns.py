"""Deprecated module — patterns moved to ``c.Infra.*_RE`` constants.

This module previously hosted ``FlextInfraUtilitiesPatterns`` as a wrapper
over the canonical regex patterns. Per the no-wrappers/no-aliases policy
the wrapper has been removed; consumers must reference patterns directly
through ``c.Infra.<NAME>_RE`` (defined in
``flext_infra._constants.source_code``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

__all__: list[str] = []
