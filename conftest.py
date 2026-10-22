"""Pytest bootstrap for flext-core local package resolution.

Copyright (c) 2026 FLEXT Team. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

from flext_tests import install_local_packages


class FlextCorePytestBootstrap:
    """Bind local test namespaces through the shared bootstrap owner."""

    @staticmethod
    def main() -> None:
        """Install this checkout's example and test packages before collection."""
        install_local_packages(Path(__file__).resolve().parent, ("examples", "tests"))


FlextCorePytestBootstrap.main()

__all__ = ["FlextCorePytestBootstrap"]
