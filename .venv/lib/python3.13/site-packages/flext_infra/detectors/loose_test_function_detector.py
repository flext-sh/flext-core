"""Detect module-level ``test_*`` functions outside a ``Tests*`` class via rope.

Config-driven engine: the business rule (test glob, function prefix, required
class prefix) is sourced from ``rules/test-tree-rules.yml`` (ADR-005 SSOT) and
validated into ``m.Infra.TestTreeRulesConfig``. This module holds no rule
literal — it is a pure rope engine over those parameters.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, m, u
from flext_infra._utilities.rope_analysis import FlextInfraUtilitiesRopeAnalysis
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraLooseTestFunctionDetector:
    """Flag module-level ``test_*`` functions that live outside a ``Tests*`` class."""

    _RULES_FILE: ClassVar[Path] = (
        Path(__file__).parent.parent / "rules" / "test-tree-rules.yml"
    )
    _rules_cache: ClassVar[m.Infra.TestTreeRulesConfig | None] = None

    @classmethod
    def _rules(cls) -> m.Infra.TestTreeRulesConfig:
        """Return the validated, cached test-tree rule parameters."""
        cached = cls._rules_cache
        if cached is None:
            raw = u.Cli.yaml_load_mapping(cls._RULES_FILE)
            cached = m.Infra.TestTreeRulesConfig.model_validate(raw)
            cls._rules_cache = cached
        return cached

    @classmethod
    def _is_test_file(
        cls, ctx: m.Infra.DetectorContext, rules: m.Infra.TestTreeRulesConfig
    ) -> bool:
        """Return True when ``ctx.file_path`` sits under a configured test glob."""
        root = ctx.project_root or ctx.file_path.parent
        try:
            relative = ctx.file_path.resolve().relative_to(root.resolve())
        except ValueError:
            relative = ctx.file_path
        posix = relative.as_posix()
        return any(fnmatch(posix, glob) for glob in rules.test_dir_globs)

    @classmethod
    def detect_file(
        cls, ctx: m.Infra.DetectorContext
    ) -> t.SequenceOf[m.Infra.LooseTestFunctionViolation]:
        """Return one violation per loose ``test_*`` function in a test module."""
        rules = cls._rules()
        if not cls._is_test_file(ctx, rules):
            return []
        res = FlextInfraUtilitiesRopeCore.get_resource_from_path(
            ctx.rope_project, ctx.file_path
        )
        if res is None:
            return []
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(ctx.rope_project, res)
        except u.Infra.rope_runtime_errors():
            return []
        violations: list[m.Infra.LooseTestFunctionViolation] = []
        for definition in FlextInfraUtilitiesRopeAnalysis.scope_definitions(pymodule):
            if definition.kind != c.Infra.RopeScopeKind.FUNCTION:
                continue
            if not definition.is_module_level:
                continue
            if not definition.name.startswith(rules.test_fn_prefix):
                continue
            violations.append(
                m.Infra.LooseTestFunctionViolation(
                    file=str(ctx.file_path),
                    line=definition.line,
                    name=definition.name,
                    suggestion=(
                        f"Nest {definition.name} inside a single "
                        f"{rules.required_class_prefix}<Module> class "
                        "(one nested class per test module)."
                    ),
                )
            )
        return violations


__all__: list[str] = ["FlextInfraLooseTestFunctionDetector"]
