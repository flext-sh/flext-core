"""Public enforcement distinguishes dependencies from compatibility exports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from flext_tests import tm

from flext_core import u


class TestsEnforcementImportProvenance:
    """Exercise real imported consumer modules through the public query."""

    @pytest.mark.parametrize(
        ("binding", "rule", "rejected"),
        [
            ("from flext_core import m\nError = m.ValidationError", "ENFORCE-070", False),
            ("from pydantic_core import ValidationError", "ENFORCE-070", True),
            ("import pydantic_core as dependency", "ENFORCE-070", True),
            (
                "from .base import FlextProbeBase as Dependency\n"
                "class FlextProbeDerived(Dependency):\n    pass",
                "ENFORCE-066",
                False,
            ),
            (
                "from .base import FlextProbeBase\nLegacyName = FlextProbeBase",
                "ENFORCE-066",
                True,
            ),
            (
                "from .base import FlextProbeBase as LegacyName\n"
                "__all__ = ['LegacyName', 'FlextProbeConsumer']",
                "ENFORCE-066",
                True,
            ),
        ],
    )
    def test_real_consumer_import_contract(
        self, tmp_path: Path, binding: str, rule: str, *, rejected: bool
    ) -> None:
        """Facade dependencies are legal; direct imports and rename exports fail."""
        package = tmp_path / "flext_probe"
        package.mkdir()
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "flext-probe"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "base.py").write_text(
            "class FlextProbeBase:\n    pass\n", encoding="utf-8"
        )
        (package / "consumer.py").write_text(
            f"{binding}\n\nclass FlextProbeConsumer:\n    pass\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module("flext_probe.consumer")
            report = u.check(module.FlextProbeConsumer)
            tm.that(any(item.rule_id == rule for item in report.violations), eq=rejected)
        finally:
            sys.path.remove(str(tmp_path))
            for name in ("flext_probe.consumer", "flext_probe.base", "flext_probe"):
                sys.modules.pop(name, None)
