"""Process-local registry for runtime violation reports.

Drained by the flext-infra correction layer. Beartype hooks and minimal-AST
detectors append ``m.ViolationReport`` entries here; ``flext_infra`` reads
them to plan rope/ast-grep corrections.

Per AGENTS.md §2.3 (single-namespaced-class) the registry is a class with
classmethods only — no instance state. Per §3.2 no ``Any``; entries are
typed as ``m.ViolationReport``. Per §3.5 (Maximize reuse) callers go through
the ``u.runtime_violation_registry()`` accessor on the ``u`` facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from typing import ClassVar

from flext_core._models.enforcement import FlextModelsEnforcement as _me


class FlextUtilitiesRuntimeViolationRegistry:
    """Thread-safe process-local registry of runtime violation reports.

    Append-only from detectors; drain returns the buffered tuple and clears
    in one atomic step. ``clear()`` resets state without returning the
    contents (used by tests).
    """

    _items: ClassVar[list[_me.ViolationReport]] = []
    _lock: ClassVar[Lock] = Lock()

    @classmethod
    def runtime_violation_registry(
        cls,
    ) -> type[FlextUtilitiesRuntimeViolationRegistry]:
        """Return the registry class for chained ``append`` / ``drain``.

        Exposing the class (rather than an instance) keeps the registry
        process-local and stateless at the consumer surface.
        """
        return cls

    @classmethod
    def append(cls, report: _me.ViolationReport) -> None:
        """Append a violation report under the registry lock."""
        with cls._lock:
            cls._items.append(report)

    @classmethod
    def drain(cls) -> Sequence[_me.ViolationReport]:
        """Return current buffer and clear it atomically."""
        with cls._lock:
            out = tuple(cls._items)
            cls._items.clear()
            return out

    @classmethod
    def clear(cls) -> None:
        """Reset registry state without returning the contents."""
        with cls._lock:
            cls._items.clear()


__all__: list[str] = ["FlextUtilitiesRuntimeViolationRegistry"]
