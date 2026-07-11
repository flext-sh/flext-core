"""Behavioral contract for the A-PT enforcement rules (ENFORCE-039..044/054/055).

Everything here is asserted through the PUBLIC surface only:

* ``u.build_canonical_catalog()`` — the published rule catalog (an
  ``EnforcementRuleSpec`` per rule, with a discriminated ``source`` model).
* ``u.check(target)`` — the runtime enforcement entrypoint that returns a
  public ``Report`` of violations.

No private ``flext_core._utilities`` internals, no dispatch-table poking, and
no assertions on how predicates are wired — only the observable contract a
caller depends on.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest

from tests.constants import c
from tests.utilities import u

if TYPE_CHECKING:
    from tests.typings import t


# Note: the top-level class name is intentionally ``TestsFlextEnforcementAptHooks``
# (not ``TestsFlextCore...``). Three export modules import this exact name
# (tests/_exports_lazy_part_03.py, tests/_exports_typing_unit.py,
# tests/unit/_exports_lazy_part_02.py) and are out of scope for this change, so
# renaming would break collection. The name follows the repo's ``TestsFlext<Module>``
# convention, which coexists with ``TestsFlextCore<Module>`` across tests/unit/.
class TestsFlextEnforcementAptHooks:
    """Public catalog + runtime contract for the A-PT enforcement rules."""

    A_PT_RULE_IDS: ClassVar[t.StrSequence] = (
        "ENFORCE-039",
        "ENFORCE-040",
        "ENFORCE-041",
        "ENFORCE-042",
        "ENFORCE-043",
        "ENFORCE-044",
        "ENFORCE-054",
        "ENFORCE-055",
    )

    # --- Catalog membership & invariants (public spec models) ---

    @pytest.mark.parametrize("rule_id", A_PT_RULE_IDS)
    def test_a_pt_rule_is_published_in_canonical_catalog(self, rule_id: str) -> None:
        catalog_ids = {rule.id for rule in u.build_canonical_catalog().rules}
        assert rule_id in catalog_ids

    @pytest.mark.parametrize("rule_id", A_PT_RULE_IDS)
    def test_a_pt_rule_carries_agents_md_anchor(self, rule_id: str) -> None:
        rule = u.build_canonical_catalog().by_id(rule_id)
        assert rule is not None
        assert rule.agents_md_anchor != ""

    @pytest.mark.parametrize("rule_id", A_PT_RULE_IDS)
    def test_a_pt_rule_documents_problem_and_fix(self, rule_id: str) -> None:
        # The published spec is what a caller reads to understand/repair a
        # violation — both narrative fields must be populated.
        rule = u.build_canonical_catalog().by_id(rule_id)
        assert rule is not None
        assert rule.description != ""
        assert rule.fix_action is not None
        assert rule.fix_action.target != ""

    @pytest.mark.parametrize("rule_id", A_PT_RULE_IDS)
    def test_a_pt_rule_severity_is_a_named_level(self, rule_id: str) -> None:
        rule = u.build_canonical_catalog().by_id(rule_id)
        assert rule is not None
        assert rule.severity.value != ""

    def test_every_catalog_rule_id_is_well_formed_and_unique(self) -> None:
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        for rid in ids:
            assert rid.startswith("ENFORCE-")
            assert len(rid) == len("ENFORCE-NNN")
        assert len(ids) == len(set(ids))

    # --- Per-rule source contract (public discriminated ``source`` model) ---

    @pytest.mark.parametrize(
        ("rule_id", "predicate_kind"),
        [
            ("ENFORCE-039", c.EnforcementPredicateKind.DEPRECATED_SYNTAX),
            ("ENFORCE-041", c.EnforcementPredicateKind.DEPRECATED_SYNTAX),
            ("ENFORCE-042", c.EnforcementPredicateKind.LOOSE_SYMBOL),
            ("ENFORCE-043", c.EnforcementPredicateKind.WRAPPER),
            ("ENFORCE-044", c.EnforcementPredicateKind.DEPRECATED_SYNTAX),
            ("ENFORCE-054", c.EnforcementPredicateKind.DEPRECATED_SYNTAX),
            ("ENFORCE-055", c.EnforcementPredicateKind.DEPRECATED_SYNTAX),
        ],
    )
    def test_beartype_rule_binds_expected_predicate_kind(
        self,
        rule_id: str,
        predicate_kind: c.EnforcementPredicateKind,
    ) -> None:
        rule = u.build_canonical_catalog().by_id(rule_id)
        assert rule is not None
        assert rule.source.kind == "beartype"
        assert rule.source.predicate_kind == predicate_kind

    def test_enforce_040_is_delegated_to_ruff_pgh003(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-040")
        assert rule is not None
        assert rule.source.kind == "ruff"
        assert rule.source.rule_code == "PGH003"

    # --- Runtime behavior via the public ``u.check`` entrypoint ---

    def test_check_skips_source_unavailable_target_without_raising(self) -> None:
        # Dynamically constructed classes have no source file. The runtime
        # contract is: never raise, and emit no source-based A-PT violation.
        dyn_class = type("FlextAptDynamicProbeFixture", (), {})
        report = u.check(dyn_class)
        emitted_a_pt = {
            v.rule_id for v in report.violations if v.rule_id in self.A_PT_RULE_IDS
        }
        assert emitted_a_pt == set()

    def test_check_emits_only_catalog_consistent_violations(self) -> None:
        # Full runtime pipeline (dispatch -> visitors -> emit -> Report). Every
        # violation the engine produces must be well-formed and, when it names a
        # rule, reference a rule that actually exists in the public catalog.
        catalog_ids = {rule.id for rule in u.build_canonical_catalog().rules}
        report = u.check(type(self))
        for violation in report.violations:
            assert violation.message != ""
            assert (not violation.rule_id) or violation.rule_id in catalog_ids
