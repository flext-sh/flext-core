"""Behavioral contract: FLEXT enforcement warnings stay visible to callers.

Enforcement violations reach callers as the public ``FlextMroViolation`` /
``FlextSmellViolation`` warning categories. A caller depends on three things:
the categories behave as ``UserWarning`` subclasses (visible by default and
catchable), the smell category inherits MRO-violation semantics so filtering
the parent also captures it, and the shipped ``filterwarnings`` configuration
never silences them again. These tests assert only that observable behaviour.
"""

from __future__ import annotations

import sys
import textwrap
import warnings
from pathlib import Path
from typing import cast

import pytest

from flext_core._constants.enforcement import FlextMroViolation, FlextSmellViolation
from tests import u

from tests import t


class TestsFlextCoreEnforcementWarningVisibility:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

    @pytest.mark.parametrize(
        "category",
        [FlextMroViolation, FlextSmellViolation],
    )
    def test_enforcement_categories_are_userwarnings(
        self,
        category: type[FlextMroViolation],
    ) -> None:
        # Arrange / Act / Assert: default warning filters surface UserWarning,
        # so a caller sees enforcement violations without extra configuration.
        assert issubclass(category, UserWarning)

    def test_smell_violation_is_a_mro_violation(self) -> None:
        # A caller filtering or catching FlextMroViolation must also capture
        # the more specific smell category.
        assert issubclass(FlextSmellViolation, FlextMroViolation)

    @pytest.mark.parametrize(
        "category",
        [FlextMroViolation, FlextSmellViolation],
    )
    def test_emitted_violation_is_observable_with_message(
        self,
        category: type[FlextMroViolation],
    ) -> None:
        # Act: emit the violation the way the enforcement engine does.
        # Assert: pytest.warns observes the exact category and its message.
        with pytest.warns(category, match="ENFORCE-probe") as record:
            warnings.warn(
                "ENFORCE-probe: runtime violation visibility",
                category,
                stacklevel=2,
            )

        assert len(record) == 1
        assert record[0].category is category

    def test_parent_category_captures_smell_violation(self) -> None:
        # Catching the parent category must capture a smell violation too,
        # since callers filter on FlextMroViolation broadly.
        with pytest.warns(FlextMroViolation) as record:
            warnings.warn(
                "ENFORCE-probe: smell via parent",
                FlextSmellViolation,
                stacklevel=2,
            )

        assert record[0].category is FlextSmellViolation

    def test_real_filterwarnings_keep_mro_violations_visible(
        self,
        tmp_path: Path,
    ) -> None:
        # Arrange: reconstruct the shipped pytest filterwarnings config so the
        # test fails if a future edit silences enforcement warnings again.
        payload = u.config_load(self._PROJECT_ROOT / "pyproject.toml").unwrap()
        tool = cast("t.JsonMapping", payload.get("tool", {}))
        pytest_tool = cast("t.JsonMapping", tool.get("pytest", {}))
        ini_options = cast("t.JsonMapping", pytest_tool.get("ini_options", {}))
        filters = cast("list[str]", ini_options.get("filterwarnings", []))

        filter_lines = "\n".join(f"    {item}" for item in filters)
        (tmp_path / "pytest.ini").write_text(
            f"[pytest]\nfilterwarnings =\n{filter_lines}\n",
            encoding="utf-8",
        )
        (tmp_path / "test_probe.py").write_text(
            textwrap.dedent(
                """
                import warnings

                from flext_core._constants.enforcement import FlextMroViolation


                def test_probe() -> None:
                    warnings.warn(
                        "ENFORCE-probe: runtime violation visibility",
                        FlextMroViolation,
                        stacklevel=2,
                    )
                """,
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        # Act: run a real, isolated pytest session under that config.
        result = u.Cli.run_raw(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_probe.py",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=tmp_path,
        )

        # Assert: the warning survives to the caller's output.
        assert result.success, result.error
        output = result.value.stdout + result.value.stderr
        assert result.value.exit_code == 0, output
        assert "FlextMroViolation" in output, (
            "runtime enforcement warning suppressed by shipped filterwarnings:\n"
            + output
        )
