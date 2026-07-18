"""Tests for runtime enforcement — query ``u.check(target)`` reports.

Every legacy per-rule helper (``u.check_no_any``, ``u.is_exempt``, etc.)
is gone. Tests assert over ``m.Report.violations`` filtered
by ``layer`` / ``severity`` / message fragment.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


def messages(report: p.Report, *, fragment: str) -> list[str]:
    return [v.message for v in report.violations if fragment in v.message]


def make_class(name: str, body: dict[str, object]) -> type:
    cls = type(name, (), body)
    cls.__qualname__ = name  # strip test-method qualname prefix
    cls.__module__ = "flext_core.synthetic"
    return cls
