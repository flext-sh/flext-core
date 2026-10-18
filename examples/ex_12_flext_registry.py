"""Golden-file example for the registry DSL public APIs.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from examples.ex_12_registry_plugins import ExamplesFlextRegistryPlugins


class ExamplesFlextRegistryDsl(ExamplesFlextRegistryPlugins):
    """Exercise the canonical registry DSL public API."""

    @override
    def exercise(self) -> None:
        """Run all registry DSL example sections."""
        registry = self._exercise_discovery()
        self._exercise_atomic_bindings()
        self._exercise_plugins(registry)


if __name__ == "__main__":
    ExamplesFlextRegistryDsl(caller_file=Path(__file__)).run()
