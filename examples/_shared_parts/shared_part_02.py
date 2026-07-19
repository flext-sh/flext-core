"""Golden-file verification and shared models for flext-core examples."""

from __future__ import annotations

import sys
from typing import cast

from examples._shared_parts.shared_part_01 import ExamplesFlextSharedBase
from examples import m
from flext_core import p, r, t


class ExamplesFlextShared(ExamplesFlextSharedBase):
    """Base class for golden-file example scripts."""

    def audit_check(self, label: str, value: object | None) -> None:
        """Append ``label: <serialised value>`` to the results buffer."""
        separator = m.Examples.LABEL_VALUE_SEPARATOR
        self._results.append(f"{label}{separator}{self.ser(value)}")

    def rand_person(self) -> ExamplesFlextShared.Person:
        """Return a ``Person`` with random name and age."""
        return self.Person(name=self.rand_str(6), age=self.rand_int(1, 99))

    def run(self) -> None:
        """Execute exercise and verify lifecycle."""
        self.exercise()
        self.verify()

    def verify(self) -> None:
        """Compare accumulated results against the ``.expected`` golden file."""
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
                pass_template = m.Examples.TEMPLATE_BY_KIND[
                    m.Examples.OutputKind.SUCCESS
                ]
                _ = sys.stdout.write(
                    pass_template.format(
                        kind=m.Examples.OutputKind.SUCCESS,
                        stem=self.caller_file.stem,
                        checks=checks,
                    )
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
                )
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
            )
        )

    class Person(m.Examples.Person):
        """Tiny Pydantic model used across several examples."""

    class Handle(m.Examples.Handle):
        """Tiny model used to exercise ``with_resource``."""

    @staticmethod
    def bind_probe(result_obj: p.Result[int], delta: int) -> int | str:
        """Safely attempt adding ``delta`` to a successful result."""
        try:
            return cast(
                "int | str",
                result_obj.flat_map(lambda n: r[int].ok(n + delta)).unwrap_or(-1),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            return f"{type(exc).__name__}:{exc}"

    @staticmethod
    def bind_status(value: t.JsonValue) -> t.JsonValue:
        """Return a summary ConfigMap when *value* is a ``r``."""
        return value


__all__: list[str] = ["ExamplesFlextShared"]
