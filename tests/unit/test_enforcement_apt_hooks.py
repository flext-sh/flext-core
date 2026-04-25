"""Lane A-PT — beartype hooks + ENFORCEMENT_RULES + ENFORCEMENT_CATALOG entries.

Per AGENT_COORDINATION.md §4.1 / §2.1 / §4.8, A-PT owns ENFORCE-039..044.
SSOT/DRY/YAGNI architecture (per user directive 2026-04-25):

* ENFORCE-040 delegates to ``ruff PGH003`` via ``EnforcementRuffSource``
  — no new code in flext-core.
* ENFORCE-039 / ENFORCE-041 / ENFORCE-042 / ENFORCE-043 / ENFORCE-044
  catalog rows use ``EnforcementSkillPointerSource`` (the only currently
  registered source kind that names a skill anchor without requiring
  per-rule discriminated-union extension). Runtime detection is dispatched
  via the existing ``c.ENFORCEMENT_RULES`` string-tag system: each tag is
  paired by naming convention with a ``check_<tag>`` static method on
  ``FlextUtilitiesBeartypeEngine`` and an item-iterator arm in
  ``FlextUtilitiesEnforcementCollect._namespace_items``.
* ENFORCE-042 reuses the existing ``check_settings_inheritance`` hook —
  zero new detection code for that rule.
* ENFORCE-039 / ENFORCE-041 / ENFORCE-043 / ENFORCE-044 add four new
  ``check_<tag>`` static methods that share the centralized regex/path/
  builtin sentinels declared on ``FlextConstantsEnforcement``.

No loose module-level constants — fixture data is held as ``ClassVar`` on
the single test class per AGENTS.md §3.6 (one ``Tests<...>`` class per
module).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core._utilities.beartype_engine import (
    FlextUtilitiesBeartypeEngine as ube,
)
from tests import c


class TestsFlextCoreEnforcementAptHooks:
    """Catalog entries + beartype hooks + dispatcher wiring for A-PT rules."""

    A_PT_RULE_IDS: ClassVar[tuple[str, ...]] = (
        "ENFORCE-039",
        "ENFORCE-040",
        "ENFORCE-041",
        "ENFORCE-042",
        "ENFORCE-043",
        "ENFORCE-044",
    )
    A_PT_BEARTYPE_TAGS: ClassVar[tuple[str, ...]] = (
        "cast_outside_core",
        "model_rebuild_call",
        "pass_through_wrapper",
        "private_attr_probe",
    )

    # --- Catalog membership invariants ---

    def test_all_six_a_pt_rule_ids_are_present_in_catalog(self) -> None:
        ids = {rule.id for rule in c.ENFORCEMENT_CATALOG.rules}
        for rule_id in self.A_PT_RULE_IDS:
            assert rule_id in ids, f"missing {rule_id}"

    def test_a_pt_rules_carry_agents_md_anchors(self) -> None:
        for rule_id in self.A_PT_RULE_IDS:
            rule = c.ENFORCEMENT_CATALOG.by_id(rule_id)
            assert rule is not None
            assert rule.agents_md_anchor != "", f"{rule_id} missing anchor"

    def test_catalog_invariants_unchanged(self) -> None:
        ids = [rule.id for rule in c.ENFORCEMENT_CATALOG.rules]
        for rid in ids:
            assert rid.startswith("ENFORCE-")
            assert len(rid) == len("ENFORCE-NNN")
        assert len(ids) == len(set(ids))

    # --- Per-rule source contracts ---

    def test_enforce_039_uses_skill_pointer_for_strict_typing(self) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-039")
        assert rule is not None
        assert rule.source.kind == "skill_pointer"
        assert rule.source.skill == "flext-strict-typing"

    def test_enforce_040_uses_ruff_source_pgh003(self) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-040")
        assert rule is not None
        assert rule.source.kind == "ruff"
        assert rule.source.rule_code == "PGH003"

    def test_enforce_041_uses_skill_pointer_for_patterns(self) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-041")
        assert rule is not None
        assert rule.source.kind == "skill_pointer"
        assert rule.source.skill == "flext-patterns"

    def test_enforce_042_uses_skill_pointer_and_existing_hook_is_present(
        self,
    ) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-042")
        assert rule is not None
        assert rule.source.kind == "skill_pointer"
        assert rule.source.skill == "flext-patterns"
        # SSOT/DRY: the runtime detection is the existing hook on the
        # canonical predicate surface — not duplicated by A-PT.
        assert hasattr(ube, "check_settings_inheritance")
        assert "settings_inheritance" in c.ENFORCEMENT_RULES

    def test_enforce_043_uses_skill_pointer_for_refactoring_workflow(self) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-043")
        assert rule is not None
        assert rule.source.kind == "skill_pointer"
        assert rule.source.skill == "flext-refactoring-workflow"

    def test_enforce_044_uses_skill_pointer_for_strict_typing(self) -> None:
        rule = c.ENFORCEMENT_CATALOG.by_id("ENFORCE-044")
        assert rule is not None
        assert rule.source.kind == "skill_pointer"
        assert rule.source.skill == "flext-strict-typing"

    # --- Runtime dispatcher wiring contracts ---

    def test_each_new_hook_is_a_callable_static_method_on_engine(self) -> None:
        for tag in self.A_PT_BEARTYPE_TAGS:
            method = getattr(ube, f"check_{tag}", None)
            assert callable(method), (
                f"missing check_{tag} on FlextUtilitiesBeartypeEngine"
            )

    def test_each_new_tag_is_registered_in_enforcement_rules(self) -> None:
        for tag in self.A_PT_BEARTYPE_TAGS:
            assert tag in c.ENFORCEMENT_RULES, (
                f"tag '{tag}' missing from c.ENFORCEMENT_RULES"
            )
            row = c.ENFORCEMENT_RULES[tag]
            assert len(row) == 5
            *_, problem, fix = row
            assert isinstance(problem, str) and problem
            assert isinstance(fix, str) and fix

    def test_centralized_detection_constants_are_in_flext_constants(self) -> None:
        # Per the user directive: detection sentinels live on
        # ``FlextConstantsEnforcement`` (centralized SSOT) — never as loose
        # module-level constants on ``beartype_engine.py``.
        assert hasattr(c, "ENFORCE_CAST_TYPING_IMPORT_RE")
        assert hasattr(c, "ENFORCE_CAST_CALL_RE")
        assert hasattr(c, "ENFORCE_MODEL_REBUILD_CALL_RE")
        assert hasattr(c, "ENFORCE_FLEXT_CORE_PATH_MARKERS")
        assert hasattr(c, "ENFORCE_PRIVATE_PROBE_BUILTINS")

    # --- Hook resilience contract (source-skip on dynamic class) ---

    def test_each_new_hook_returns_none_for_source_unavailable_target(self) -> None:
        # Dynamically constructed classes have no source file. Every hook
        # MUST silently skip per the runtime-safety contract — never raise.
        dyn_class = type("FlextAptDynamicProbeFixture", (), {})
        for tag in self.A_PT_BEARTYPE_TAGS:
            method = getattr(ube, f"check_{tag}")
            result = method(dyn_class)
            assert result is None, (
                f"check_{tag} must skip on source-unavailable target; "
                f"got {result!r}"
            )
