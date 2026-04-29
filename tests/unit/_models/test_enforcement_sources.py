"""Behavior contract for the typed enforcement source variants.

Covers ``EnforcementSourceKind`` membership and the
``EnforcementMinimalAstSource`` discriminator schema. Catalog-completeness
invariant requires every declared source kind to appear on at least one
``EnforcementRuleSpec``.
"""

from __future__ import annotations

import pytest

from tests import c, m


class TestsFlextModelsEnforcementSources:
    """Behavior contract for surviving EnforcementSource variants."""

    # --- EnforcementSourceKind ---

    def test_enforcement_source_kind_includes_beartype(self) -> None:
        assert m.EnforcementSourceKind.BEARTYPE.value == "beartype"

    def test_enforcement_source_kind_includes_minimal_ast(self) -> None:
        assert m.EnforcementSourceKind.MINIMAL_AST.value == "minimal_ast"

    # --- EnforcementMinimalAstSource ---

    def test_minimal_ast_source_default_require_source_true(self) -> None:
        src = m.EnforcementMinimalAstSource(pattern="$X = $Y")
        assert src.kind == "minimal_ast"
        assert src.pattern == "$X = $Y"
        assert src.require_source is True

    def test_minimal_ast_source_explicit_require_source_false(self) -> None:
        src = m.EnforcementMinimalAstSource(pattern="$X = $Y", require_source=False)
        assert src.require_source is False

    def test_minimal_ast_source_rejects_empty_pattern(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementMinimalAstSource(pattern="")
