"""Behavior contract for the typed enforcement source variants.

Covers ``EnforcementSourceKind`` membership and the ``EnforcementBeartypeSource``
discriminator schema. Catalog-completeness invariant requires every declared
source kind to appear on at least one ``EnforcementRuleSpec``.
"""

from __future__ import annotations

import pytest

from tests.constants import c
from tests.models import m


class TestsFlextModelsEnforcementSources:
    """Behavior contract for surviving EnforcementSource variants."""

    # --- EnforcementSourceKind ---

    def test_enforcement_source_kind_includes_beartype(self) -> None:
        assert m.EnforcementSourceKind.BEARTYPE.value == "beartype"

    def test_enforcement_source_kind_has_no_minimal_ast(self) -> None:
        assert "minimal_ast" not in {kind.value for kind in m.EnforcementSourceKind}

    # --- EnforcementBeartypeSource ---

    def test_beartype_source_kind_discriminator(self) -> None:
        src = m.EnforcementBeartypeSource(
            predicate_kind=c.EnforcementPredicateKind.MODULE_ALIAS,
        )
        assert src.kind == "beartype"
        assert src.predicate_kind is c.EnforcementPredicateKind.MODULE_ALIAS

    def test_beartype_source_rejects_unknown_predicate_kind(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementBeartypeSource.model_validate(
                {"predicate_kind": "not_a_kind"},
            )
