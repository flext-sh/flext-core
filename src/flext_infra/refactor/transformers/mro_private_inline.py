"""CST transformer for private-symbol inlining in MRO migration."""

from __future__ import annotations

from typing import override

import libcst as cst

from flext_infra import c, m, p, t, u


class FlextInfraRefactorMROPrivateInlineTransformer(cst.CSTTransformer):
    """Inline configured private-name values after migration."""

    def __init__(self, *, replacement_values: dict[str, cst.BaseExpression]) -> None:
        """Initialize with symbol-to-value mapping for private constant inlining."""
        self.replacement_values = replacement_values

    @override
    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if original_node.value in self.replacement_values:
            return self.replacement_values[original_node.value]
        return updated_node


__all__ = ["FlextInfraRefactorMROPrivateInlineTransformer"]
