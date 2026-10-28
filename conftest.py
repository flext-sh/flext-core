# AUTO-GENERATED FILE — Regenerate with: make gen
"""Pytest bootstrap for flext_core local package resolution.

Rendered from flext-infra ``templates/project/base/conftest.py.j2``: change the
template or the shared ``install_local_packages`` owner, then run ``make gen``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_tests import install_local_packages


class FlextPytestBootstrap:
    """Bind this checkout's local tiers through the shared bootstrap owner."""

    @staticmethod
    def main() -> None:
        """Install this checkout's local tier packages before collection."""
        install_local_packages(Path(__file__).resolve().parent)


FlextPytestBootstrap.main()

__all__: list[str] = ["FlextPytestBootstrap"]
