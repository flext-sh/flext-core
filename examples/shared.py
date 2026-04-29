"""Shared golden-file test harness for flext-core examples.

Provides ``Examples`` — a single MRO base class that every ``ex_*.py``
script subclasses.  Through MRO each subclass inherits:

- ``check`` / ``section`` / ``ser`` / ``verify`` — golden-file recording
- ``rand_int`` / ``rand_float`` / ``rand_str`` / ``rand_bool`` — deterministic
  random value generators (fixed seed for reproducible golden-file output)
- ``rand_person`` / ``rand_dict`` — composite random generators
- ``Person`` / ``Handle`` — shared Pydantic models
- ``bind_probe`` / ``bind_status`` — r probe helpers

Usage (inside an ``ex_*.py`` file)::

    from examples import Examples


    class Ex01r(Examples):
        def exercise(self) -> None:
            val = self.rand_int()
            result = r[int].ok(val)
            self.check("ok.value_matches", result.value == val)


    if __name__ == "__main__":
        Ex01r(__file__).run()
"""

from __future__ import annotations

import hashlib
import string
import sys
from collections.abc import (
    MutableSequence,
)
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from flext_core import p, r, t, u

from .models import m


class ExamplesFlextShared(m.BaseModel):
    """Base class for golden-file example scripts.

    Subclass once per ``ex_*.py`` module.  Implement ``exercise()`` to
    record checks via randomised inputs, then call ``run()`` to verify.

    All random generators use a fixed seed (``SEED = 42``) so the output
    is fully deterministic and golden-file comparison works reliably.
    """

    SEED: int = 42

    caller_file: Path
    _results: MutableSequence[str] = u.PrivateAttr(default_factory=list)
    _counter: int = u.PrivateAttr(default_factory=lambda: 0)

    def __init__(self, caller_file: str) -> None:
        """Initialise with the caller's ``__file__`` for golden-file resolution."""
        super().__init__(caller_file=Path(caller_file))

    def _next_unit_float(self) -> float:
        payload = f"{self.SEED}:{self._counter}".encode()
        digest = hashlib.sha256(payload).digest()
        self._counter += 1
        raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
        max_u64 = (1 << 64) - 1
        return raw / max_u64

    def audit_check(
        self,
        label: str,
        value: object | None,
    ) -> None:
        """Append ``label: <serialised value>`` to the results buffer."""
        separator = m.Examples.LABEL_VALUE_SEPARATOR
        self._results.append(f"{label}{separator}{self.ser(value)}")

    def exercise(self) -> None:
        """Override in subclasses to exercise the target class."""
        msg = m.Examples.ErrorMessages.EXERCISE_NOT_IMPLEMENTED
        raise NotImplementedError(msg)

    def rand_bool(self) -> bool:
        """Return a deterministic pseudo-random boolean."""
        return self._next_unit_float() >= 0.5

    def rand_dict(self, n: int = 3) -> m.ConfigMap:
        """Return a ConfigMap with ``n`` random string keys → int values."""
        return m.ConfigMap(
            root={self.rand_str(4): self.rand_int(0, 100) for _ in range(n)},
        )

    def rand_float(self, lo: float = -1000.0, hi: float = 1000.0) -> float:
        """Return a deterministic pseudo-random float rounded to 4 decimals."""
        span = hi - lo
        return round(lo + (self._next_unit_float() * span), 4)

    def rand_int(self, lo: int = -1000, hi: int = 1000) -> int:
        """Return a deterministic pseudo-random integer in ``[lo, hi]``."""
        span = (hi - lo) + 1
        return lo + int(self._next_unit_float() * span)

    def rand_person(self) -> ExamplesFlextShared.Person:
        """Return a ``Person`` with random name (6 chars) and age (1–99)."""
        return self.Person(name=self.rand_str(6), age=self.rand_int(1, 99))

    def rand_str(self, length: int = 8) -> str:
        """Return a deterministic pseudo-random lowercase ASCII string."""
        alphabet = string.ascii_lowercase
        return "".join(
            alphabet[int(self._next_unit_float() * len(alphabet)) % len(alphabet)]
            for _ in range(length)
        )

    def run(self) -> None:
        """Execute exercise → verify lifecycle."""
        self.exercise()
        self.verify()

    def section(self, name: str) -> None:
        """Start a new named section (blank-line separated in the output)."""
        if self._results:
            self._results.append("")
        self._results.append(f"[{name}]")

    def ser(
        self,
        v: object | None,
    ) -> str:
        """Deterministic, human-readable serialisation for golden-file output.

        Handles ``None``, bools, numbers, strings, lists, dicts, types,
        ``datetime``, ``Path``, and falls back to ``type(v).__name__``.
        """
        if v is None:
            return "None"
        if isinstance(v, r):
            return "Result"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int | float):
            return str(v)
        if isinstance(v, str):
            return repr(v)
        if isinstance(v, list):
            return "list"
        if isinstance(v, dict):
            return "dict"
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        return type(v).__name__

    def verify(self) -> None:
        """Compare accumulated results against the ``.expected`` golden file.

        * If the golden file exists and matches → prints ``PASS``.
        * If it exists but mismatches → writes ``.actual`` and exits 1.
        * If it does not exist → generates it and prints ``GENERATED``.
        """
        actual = "\n".join(self._results).strip() + "\n"
        expected_path = self.caller_file.with_suffix(".expected")
        checks = sum(
            1
            for line in self._results
            if m.Examples.RESULT_LINE_PATTERN.match(line) is not None
        )
        if expected_path.exists():
            expected = expected_path.read_text(encoding="utf-8")
            if actual == expected:
                pass_template = m.Examples.TEMPLATE_BY_KIND[m.Examples.OutputKind.PASS]
                _ = sys.stdout.write(
                    pass_template.format(
                        kind=m.Examples.OutputKind.PASS,
                        stem=self.caller_file.stem,
                        checks=checks,
                    ),
                )
                return
            actual_path = self.caller_file.with_suffix(".actual")
            _ = actual_path.write_text(actual, encoding="utf-8")
            fail_template = m.Examples.TEMPLATE_BY_KIND[m.Examples.OutputKind.FAIL]
            _ = sys.stdout.write(
                fail_template.format(
                    kind=m.Examples.OutputKind.FAIL,
                    stem=self.caller_file.stem,
                    expected_name=expected_path.name,
                    actual_name=actual_path.name,
                ),
            )
            sys.exit(1)
        _ = expected_path.write_text(actual, encoding="utf-8")
        generated_template = m.Examples.TEMPLATE_BY_KIND[
            m.Examples.OutputKind.GENERATED
        ]
        _ = sys.stdout.write(
            generated_template.format(
                kind=m.Examples.OutputKind.GENERATED,
                expected_name=expected_path.name,
                checks=checks,
            ),
        )

    class Person(m.Examples.Person):
        """Tiny Pydantic model used across several examples."""

    class Handle(m.Examples.Handle):
        """Tiny model used to exercise ``with_resource``."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    @staticmethod
    def bind_probe(result_obj: p.Result[int], delta: int) -> int | str:
        """Safely attempt ``result_obj.bind(lambda n: r[int].ok(n + delta))``."""
        try:
            return result_obj.flat_map(lambda n: r[int].ok(n + delta)).unwrap_or(-1)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"{type(exc).__name__}:{exc}"

    @staticmethod
    def bind_status(
        value: t.JsonValue,
    ) -> t.JsonValue:
        """Return a summary ConfigMap when *value* is a ``r``."""
        return value
