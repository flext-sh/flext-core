"""Shared deterministic harness base for flext-core examples."""

from __future__ import annotations

import hashlib
import string
from collections.abc import MutableSequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from examples import m
from flext_core import r, u


class ExamplesFlextSharedBase(m.BaseModel):
    """Base state and deterministic helpers for golden-file examples."""

    SEED: int = 42
    _BOOL_THRESHOLD: ClassVar[float] = 0.5

    caller_file: Path
    _results: MutableSequence[str] = u.PrivateAttr(default_factory=list)
    _counter: int = u.PrivateAttr(default_factory=lambda: 0)

    def __init__(self, caller_file: str | Path) -> None:
        """Initialise with the caller's ``__file__`` for golden-file resolution."""
        super().__init__(caller_file=Path(caller_file))

    def _next_unit_float(self) -> float:
        payload = f"{self.SEED}:{self._counter}".encode()
        digest = hashlib.sha256(payload).digest()
        self._counter += 1
        raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
        max_u64 = (1 << 64) - 1
        return raw / max_u64

    def exercise(self) -> None:
        """Override in subclasses to exercise the target class."""
        msg = m.Examples.ErrorMessages.EXERCISE_NOT_IMPLEMENTED
        raise NotImplementedError(msg)

    def rand_bool(self) -> bool:
        """Return a deterministic pseudo-random boolean."""
        return self._next_unit_float() >= self._BOOL_THRESHOLD

    def rand_dict(self, n: int = 3) -> m.ConfigMap:
        """Return a ConfigMap with ``n`` random string keys to int values."""
        return m.ConfigMap(
            root={self.rand_str(4): self.rand_int(0, 100) for _ in range(n)}
        )

    def rand_float(self, lo: float = -1000.0, hi: float = 1000.0) -> float:
        """Return a deterministic pseudo-random float rounded to 4 decimals."""
        span = hi - lo
        return round(lo + (self._next_unit_float() * span), 4)

    def rand_int(self, lo: int = -1000, hi: int = 1000) -> int:
        """Return a deterministic pseudo-random integer in ``[lo, hi]``."""
        span = (hi - lo) + 1
        return lo + int(self._next_unit_float() * span)

    def rand_str(self, length: int = 8) -> str:
        """Return a deterministic pseudo-random lowercase ASCII string."""
        alphabet = string.ascii_lowercase
        return "".join(
            alphabet[int(self._next_unit_float() * len(alphabet)) % len(alphabet)]
            for _ in range(length)
        )

    def section(self, name: str) -> None:
        """Start a new named section."""
        if self._results:
            self._results.append("")
        self._results.append(f"[{name}]")

    def ser(self, v: object | None) -> str:
        """Deterministic, human-readable serialisation for golden-file output."""
        if v is None:
            return "None"
        if isinstance(v, r):
            return "Result"
        result: str
        if isinstance(v, bool | int | float):
            result = str(v)
        elif isinstance(v, str):
            result = repr(v)
        elif isinstance(v, datetime):
            result = v.isoformat()
        elif isinstance(v, Path):
            result = str(v)
        elif isinstance(v, (list, dict)):
            result = type(v).__name__.lower()
        else:
            result = type(v).__name__
        return result


__all__: list[str] = ["ExamplesFlextSharedBase"]
