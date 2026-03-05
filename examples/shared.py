"""Shared golden-file test harness for flext-core examples.

Provides ``Examples`` — a single MRO base class that every ``ex_*.py``
script subclasses.  Through MRO each subclass inherits:

- ``check`` / ``section`` / ``ser`` / ``verify`` — golden-file recording
- ``rand_int`` / ``rand_float`` / ``rand_str`` / ``rand_bool`` — deterministic
  random value generators (fixed seed for reproducible golden-file output)
- ``rand_person`` / ``rand_dict`` — composite random generators
- ``Person`` / ``Handle`` — shared Pydantic/dataclass models
- ``bind_probe`` / ``bind_status`` — FlextResult probe helpers

Usage (inside an ``ex_*.py`` file)::

    from .shared import Examples


    class Ex01FlextResult(Examples):
        def exercise(self) -> None:
            val = self.rand_int()
            result = r[int].ok(val)
            self.check("ok.value_matches", result.value == val)


    if __name__ == "__main__":
        Ex01FlextResult(__file__).run()
"""

from __future__ import annotations

import random
import string
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from flext_core import FlextResult, r, t, u


class Examples:
    """Base class for golden-file example scripts.

    Subclass once per ``ex_*.py`` module.  Implement ``exercise()`` to
    record checks via randomised inputs, then call ``run()`` to verify.

    All random generators use a fixed seed (``SEED = 42``) so the output
    is fully deterministic and golden-file comparison works reliably.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    SEED: int = 42

    def __init__(self, caller_file: str) -> None:
        """Initialise with the caller's ``__file__`` for golden-file resolution."""
        self._results: list[str] = []
        self._caller = Path(caller_file)
        self._rng = random.Random(self.SEED)  # noqa: S311

    def exercise(self) -> None:
        """Override in subclasses to exercise the target class."""
        msg = "Subclasses must implement exercise()"
        raise NotImplementedError(msg)

    def run(self) -> None:
        """Execute exercise → verify lifecycle."""
        self.exercise()
        self.verify()

    # ------------------------------------------------------------------
    # Random value generators (deterministic via fixed seed)
    # ------------------------------------------------------------------

    def rand_int(self, lo: int = -1000, hi: int = 1000) -> int:
        """Return a deterministic pseudo-random integer in ``[lo, hi]``."""
        return self._rng.randint(lo, hi)

    def rand_float(self, lo: float = -1000.0, hi: float = 1000.0) -> float:
        """Return a deterministic pseudo-random float rounded to 4 decimals."""
        return round(self._rng.uniform(lo, hi), 4)

    def rand_str(self, length: int = 8) -> str:
        """Return a deterministic pseudo-random lowercase ASCII string."""
        return "".join(self._rng.choices(string.ascii_lowercase, k=length))

    def rand_bool(self) -> bool:
        """Return a deterministic pseudo-random boolean."""
        return bool(self._rng.randint(0, 1))

    def rand_person(self) -> "Examples.Person":
        """Return a ``Person`` with random name (6 chars) and age (1–99)."""
        return self.Person(name=self.rand_str(6), age=self.rand_int(1, 99))

    def rand_dict(self, n: int = 3) -> dict[str, int]:
        """Return a dict with ``n`` random string keys → int values."""
        return {self.rand_str(4): self.rand_int(0, 100) for _ in range(n)}

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def check(self, label: str, value: t.ContainerValue | None) -> None:
        """Append ``label: <serialised value>`` to the results buffer."""
        self._results.append(f"{label}: {self.ser(value)}")

    def section(self, name: str) -> None:
        """Start a new named section (blank-line separated in the output)."""
        if self._results:
            self._results.append("")
        self._results.append(f"[{name}]")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def ser(self, v: t.ContainerValue | None) -> str:
        """Deterministic, human-readable serialisation for golden-file output.

        Handles ``None``, bools, numbers, strings, lists, dicts, types,
        ``datetime``, ``Path``, and falls back to ``type(v).__name__``.
        """
        if v is None:
            return "None"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return repr(v)
        if u.is_list(v):
            return "[" + ", ".join(self.ser(item) for item in v) + "]"
        if u.is_dict_like(v):
            pairs = ", ".join(
                f"{self.ser(k)}: {self.ser(val)}"
                for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
            )
            return "{" + pairs + "}"
        if isinstance(v, type):
            return v.__name__
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        return type(v).__name__

    # ------------------------------------------------------------------
    # Golden-file verification
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Compare accumulated results against the ``.expected`` golden file.

        * If the golden file exists and matches → prints ``PASS``.
        * If it exists but mismatches → writes ``.actual`` and exits 1.
        * If it does not exist → generates it and prints ``GENERATED``.
        """
        actual = "\n".join(self._results).strip() + "\n"
        expected_path = self._caller.with_suffix(".expected")
        checks = sum(
            1 for line in self._results if ": " in line and not line.startswith("[")
        )

        if expected_path.exists():
            expected = expected_path.read_text(encoding="utf-8")
            if actual == expected:
                sys.stdout.write(f"PASS: {self._caller.stem} ({checks} checks)\n")
                return

            actual_path = self._caller.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {self._caller.stem}"
                f" — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)

        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({checks} checks)\n")

    # ------------------------------------------------------------------
    # Shared example models
    # ------------------------------------------------------------------

    class Person(BaseModel):
        """Tiny Pydantic model used across several examples."""

        name: str
        age: int

    @dataclass
    class Handle:
        """Tiny dataclass used to exercise ``with_resource``."""

        value: int
        cleaned: bool = False

    # ------------------------------------------------------------------
    # Probe helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bind_probe(result_obj: FlextResult[int], delta: int) -> t.ContainerValue:
        """Safely attempt ``result_obj.bind(lambda n: r[int].ok(n + delta))``."""
        try:
            method = getattr(result_obj, "bind")
        except AttributeError as exc:
            return f"AttributeError:{exc}"
        if not callable(method):
            return "bind-not-callable"
        try:
            return method(lambda n: r[int].ok(n + delta))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"{type(exc).__name__}:{exc}"

    @staticmethod
    def bind_status(value: t.ContainerValue) -> t.ContainerValue:
        """Return a summary dict when *value* is a ``FlextResult``."""
        if isinstance(value, FlextResult):
            return {
                "is_success": value.is_success,
                "error": value.error,
                "unwrap_or": value.unwrap_or(-1),
            }
        return value
