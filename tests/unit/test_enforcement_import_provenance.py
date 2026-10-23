"""Public enforcement distinguishes dependencies from compatibility exports.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path

import pytest

from flext_core import FlextStrictYamlConfigSource
from tests import c, tm, u


class TestsFlextEnforcementImportProvenance:
    """Exercise real imported consumer modules through the public query."""

    @staticmethod
    @pytest.mark.parametrize(
        ("binding", "rule", "rejected"),
        [
            (
                "from flext_core import m\nError = m.ValidationError",
                "ENFORCE-070",
                False,
            ),
            (
                "import flext_core\nError = flext_core.m.ValidationError",
                "ENFORCE-070",
                False,
            ),
            (
                (
                    "from flext_core import m\n"
                    "Intermediate = m.ValidationError\nError = Intermediate"
                ),
                "ENFORCE-070",
                False,
            ),
            ("from pydantic_core import ValidationError", "ENFORCE-070", True),
            ("import pydantic_core as dependency", "ENFORCE-070", True),
            ("dependency = __import__('pydantic_core')", "ENFORCE-070", True),
            ("from pydantic import TypeAdapter", "ENFORCE-045", True),
            ("from pydantic import TypeAdapter as Adapter", "ENFORCE-045", True),
            ("import pydantic as dependency", "ENFORCE-045", True),
            ("from pydantic import JsonValue", "ENFORCE-045", True),
            ("dependency = __import__('pydantic')", "ENFORCE-045", True),
            (
                "from flext_core import m\nAdapter = m.TypeAdapter",
                "ENFORCE-045",
                False,
            ),
            (
                (
                    "from .base import FlextProbeBase as Dependency\n"
                    "class FlextProbeDerived(Dependency):\n    pass"
                ),
                "ENFORCE-066",
                False,
            ),
            (
                (
                    "from .base import FlextProbeBase as Dependency\n"
                    "class FlextProbeDerived:\n    child: Dependency"
                ),
                "ENFORCE-066",
                False,
            ),
            (
                "from .base import FlextProbeBase\nLegacyName = FlextProbeBase",
                "ENFORCE-066",
                True,
            ),
            (
                (
                    "from .base import FlextProbeBase as LegacyName\n"
                    "__all__ = ['LegacyName', 'FlextProbeConsumer']"
                ),
                "ENFORCE-066",
                True,
            ),
            (
                (
                    "from .base import FlextProbeBase as LegacyName\n"
                    "__all__ = ['FlextProbeConsumer']\n__all__ += ['LegacyName']"
                ),
                "ENFORCE-066",
                True,
            ),
        ],
    )
    @pytest.mark.parametrize("module_name", ["consumer", "_consumer"])
    def test_real_consumer_import_contract(
        tmp_path: Path,
        binding: str,
        rule: str,
        module_name: str,
        *,
        rejected: bool,
    ) -> None:
        """Facade dependencies are legal; direct imports and rename exports fail."""
        package = tmp_path / "flext_probe"
        package.mkdir()
        (tmp_path / c.PYPROJECT_FILENAME).write_text(
            '[project]\nname = "flext-probe"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "base.py").write_text(
            "class FlextProbeBase:\n    pass\n",
            encoding="utf-8",
        )
        (package / f"{module_name}.py").write_text(
            f"{binding}\n\nclass FlextProbeConsumer:\n    pass\n",
            encoding="utf-8",
        )
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module(f"flext_probe.{module_name}")
            report = u.check(module.FlextProbeConsumer)
            tm.that(
                any(item.rule_id == rule for item in report.violations),
                eq=rejected,
            )
        finally:
            sys.path.remove(str(tmp_path))
            for name in (
                f"flext_probe.{module_name}",
                "flext_probe.base",
                "flext_probe",
            ):
                sys.modules.pop(name, None)

    @staticmethod
    def test_native_config_source_obeys_owner_and_layer_zero_boundary() -> None:
        """Core owns its native adapter; config ingress never imports its own tiers."""
        report = u.check(FlextStrictYamlConfigSource)
        tm.that(
            any(item.rule_id == "ENFORCE-045" for item in report.violations),
            truth=False,
        )
        tm.that(
            any(item.rule_id == "ENFORCE-070" for item in report.violations),
            truth=False,
        )
        module = importlib.import_module(FlextStrictYamlConfigSource.__module__)
        namespace = FlextStrictYamlConfigSource.__module__.split(".")[0]
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                tm.that(node.level, eq=0)
                tm.that((node.module or "").split(".")[0], ne=namespace)
            elif isinstance(node, ast.Import):
                tm.that(
                    all(alias.name.split(".")[0] != namespace for alias in node.names),
                    truth=True,
                )
