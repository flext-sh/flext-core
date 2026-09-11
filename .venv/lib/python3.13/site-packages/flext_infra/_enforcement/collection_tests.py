"""flext-tests backed enforcement collectors."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m
from flext_infra._enforcement.collection_base import FlextInfraEnforcementCollectionBase

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p


class FlextInfraEnforcementTestsCollector(FlextInfraEnforcementCollectionBase):
    """Collect violations from flext-tests validator methods."""

    def collect_tests_validator(
        self, project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Run the flext-tests validator method declared by ``rule``."""
        source = rule.source
        if source.kind != "flext_tests_validator":
            return self._empty_failure(
                project_dir, rule, f"invalid validator source kind {source.kind!r}"
            )
        try:
            validator_cls = getattr(
                importlib.import_module("flext_tests.validator"),
                "FlextTestsValidator",
                None,
            )
        except ImportError as exc:
            return self._empty_failure(
                project_dir, rule, f"unable to import flext_tests.validator: {exc}"
            )
        if validator_cls is None:
            return self._empty_failure(
                project_dir, rule, "flext_tests.validator missing FlextTestsValidator"
            )
        method = getattr(validator_cls, source.method, None)
        if method is None or not callable(method):
            return self._empty_failure(
                project_dir,
                rule,
                f"flext_tests validator method {source.method!r} missing",
            )
        try:
            result = method(project_dir)
        except c.EXC_BROAD_RUNTIME as exc:
            return self._empty_failure(
                project_dir,
                rule,
                f"flext_tests validator method {source.method!r} failed: {exc}",
            )
        if getattr(result, "failure", False):
            error = getattr(result, "error", "") or "validator returned failure"
            return self._empty_failure(project_dir, rule, str(error))
        scan = getattr(result, "value", None)
        if scan is None:
            return self._empty_failure(
                project_dir, rule, "validator returned empty scan payload"
            )
        wanted_ids = frozenset(source.rule_ids)
        out: list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]] = []
        for violation in getattr(scan, "violations", ()):
            if wanted_ids and getattr(violation, "rule_id", "") not in wanted_ids:
                continue
            out.append((rule, violation))
        return out, []


__all__: list[str] = ["FlextInfraEnforcementTestsCollector"]
