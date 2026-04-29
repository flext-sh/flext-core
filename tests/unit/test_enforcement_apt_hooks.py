"""Behavior contract for ENFORCE-039..044 + ENFORCE-054/055.

Asserts the runtime dispatch chain after Phase 1 of the beartype-driven
engine refactor:

* Each catalog rule still resolves to a stable ``ENFORCE-NNN`` id.
* Each ENFORCE-039/041/043/044/054/055 rule's source is ``EnforcementBeartypeSource``
  with a typed ``predicate_kind`` (no more ``hook`` strings on the model).
* ENFORCE-040 stays a ``EnforcementRuffSource`` (PGH003).
* ENFORCE-042 (Settings Law) routes through the LOOSE_SYMBOL predicate.
* ``FlextUtilitiesBeartypeEngine.apply(kind, params, target)`` is the single
  runtime dispatch path; resilient to source-unavailable targets.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._utilities.beartype_engine import (
    FlextUtilitiesBeartypeEngine as ube,
)
from flext_core._utilities.enforcement import PREDICATE_BINDINGS
from tests import c, m, u


class TestsFlextEnforcementAptHooks:
    """Catalog entries + beartype dispatch table for A-PT rules."""

    A_PT_RULE_IDS: ClassVar[tuple[str, ...]] = (
        "ENFORCE-039",
        "ENFORCE-040",
        "ENFORCE-041",
        "ENFORCE-042",
        "ENFORCE-043",
        "ENFORCE-044",
        "ENFORCE-054",
        "ENFORCE-055",
    )
    A_PT_BEARTYPE_TAGS: ClassVar[tuple[str, ...]] = (
        "cast_outside_core",
        "model_rebuild_call",
        "pass_through_wrapper",
        "private_attr_probe",
        "no_core_tests_namespace",
        "no_wrapper_root_alias_import",
    )

    # --- Catalog membership invariants ---

    def test_all_six_a_pt_rule_ids_are_present_in_catalog(self) -> None:
        ids = {rule.id for rule in u.build_canonical_catalog().rules}
        for rule_id in self.A_PT_RULE_IDS:
            assert rule_id in ids, f"missing {rule_id}"

    def test_a_pt_rules_carry_agents_md_anchors(self) -> None:
        for rule_id in self.A_PT_RULE_IDS:
            rule = u.build_canonical_catalog().by_id(rule_id)
            assert rule is not None
            assert rule.agents_md_anchor != "", f"{rule_id} missing anchor"

    def test_catalog_invariants_unchanged(self) -> None:
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        for rid in ids:
            assert rid.startswith("ENFORCE-")
            assert len(rid) == len("ENFORCE-NNN")
        assert len(ids) == len(set(ids))

    # --- Per-rule source contracts ---

    def test_enforce_039_uses_beartype_source_with_deprecated_syntax_kind(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-039")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert (
            rule.source.predicate_kind == c.EnforcementPredicateKind.DEPRECATED_SYNTAX
        )

    def test_enforce_040_uses_ruff_source_pgh003(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-040")
        assert rule is not None
        assert rule.source.kind == "ruff"
        assert rule.source.rule_code == "PGH003"

    def test_enforce_041_uses_beartype_source_with_deprecated_syntax_kind(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-041")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert (
            rule.source.predicate_kind == c.EnforcementPredicateKind.DEPRECATED_SYNTAX
        )

    def test_enforce_042_routes_through_loose_symbol_predicate(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-042")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert rule.source.predicate_kind == c.EnforcementPredicateKind.LOOSE_SYMBOL
        # SSOT: settings_inheritance is bound in the dispatch table.
        kind, params = PREDICATE_BINDINGS["settings_inheritance"]
        assert kind == c.EnforcementPredicateKind.LOOSE_SYMBOL
        assert isinstance(params, m.LooseSymbolParams)
        assert params.require_settings_base is True

    def test_enforce_043_uses_wrapper_predicate(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-043")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert rule.source.predicate_kind == c.EnforcementPredicateKind.WRAPPER

    def test_enforce_044_uses_deprecated_syntax_predicate(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-044")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert (
            rule.source.predicate_kind == c.EnforcementPredicateKind.DEPRECATED_SYNTAX
        )

    def test_enforce_054_uses_deprecated_syntax_predicate(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-054")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert (
            rule.source.predicate_kind == c.EnforcementPredicateKind.DEPRECATED_SYNTAX
        )

    def test_enforce_055_uses_deprecated_syntax_predicate(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-055")
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert (
            rule.source.predicate_kind == c.EnforcementPredicateKind.DEPRECATED_SYNTAX
        )

    # --- Runtime dispatcher wiring contracts ---

    def test_each_a_pt_tag_has_predicate_binding(self) -> None:
        for tag in self.A_PT_BEARTYPE_TAGS:
            assert tag in PREDICATE_BINDINGS, (
                f"tag '{tag}' missing from PREDICATE_BINDINGS"
            )
            kind, params = PREDICATE_BINDINGS[tag]
            assert isinstance(kind, c.EnforcementPredicateKind)
            assert params is not None

    def test_each_a_pt_tag_is_registered_in_enforcement_rules(self) -> None:
        for tag in self.A_PT_BEARTYPE_TAGS:
            assert tag in c.ENFORCEMENT_RULES_TEXT, (
                f"tag '{tag}' missing from c.ENFORCEMENT_RULES_TEXT"
            )
            problem, fix = c.ENFORCEMENT_RULES_TEXT[tag]
            assert isinstance(problem, str) and problem
            assert isinstance(fix, str) and fix

    def test_centralized_detection_constants_are_in_flext_constants(self) -> None:
        # Per the user directive: detection sentinels live on
        # ``FlextConstantsEnforcement`` (centralized SSOT) — never as loose
        # module-level constants on ``beartype_engine.py``.
        assert hasattr(c, "EnforceAstHookSymbol")
        assert c.EnforceAstHookSymbol.CAST_CALL.value == "cast"
        assert c.EnforceAstHookSymbol.MODEL_REBUILD_ATTR.value == "model_rebuild"
        assert hasattr(c, "ENFORCE_FLEXT_CORE_PATH_MARKERS")
        assert hasattr(c, "ENFORCE_NON_WORKSPACE_PATH_MARKERS")
        assert hasattr(c, "ENFORCE_PRIVATE_PROBE_BUILTINS")

    # --- apply() resilience contract (source-skip on dynamic class) ---

    def test_apply_returns_none_for_source_unavailable_target(self) -> None:
        # Dynamically constructed classes have no source file. Every
        # predicate dispatched via apply() MUST silently skip per the
        # runtime-safety contract — never raise.
        dyn_class = type("FlextAptDynamicProbeFixture", (), {})
        for tag in self.A_PT_BEARTYPE_TAGS:
            kind, params = PREDICATE_BINDINGS[tag]
            result = ube.apply(kind, params, dyn_class)
            assert result is None, (
                f"apply({kind}) must skip on source-unavailable target; got {result!r}"
            )
