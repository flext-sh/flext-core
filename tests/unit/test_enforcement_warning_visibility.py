"""Regression: FlextMroViolation warnings must be visible in pytest output.

Runs a sandboxed pytest session (safe-validation rule: zero workspace
mutation) configured with the REAL flext-core ``filterwarnings`` list read
from this project's pyproject.toml, so the test fails whenever the shipped
configuration silences runtime enforcement warnings again.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests.utilities import u

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextEnforcementWarningVisibility:
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]

    def test_real_filterwarnings_keep_mro_violations_visible(
        self,
        tmp_path: Path,
    ) -> None:
        payload = u.load_pyproject_toml(self._PROJECT_ROOT)
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

                from flext_core import FlextMroViolation


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
        assert result.success, result.error
        output = result.value.stdout + result.value.stderr
        assert result.value.exit_code == 0, output
        assert "FlextMroViolation" in output, (
            "runtime enforcement warning suppressed by shipped filterwarnings:\n"
            + output
        )
