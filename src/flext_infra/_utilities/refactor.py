"""Refactor/CST helper utilities for infrastructure code analysis.

Centralizes CST (Concrete Syntax Tree) helpers previously defined as
module-level functions in ``flext_infra.refactor.analysis``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import libcst as cst


class FlextInfraUtilitiesRefactor:
    """CST/refactor helpers for code analysis.

    Usage via namespace::

        from flext_infra import u

        name = u.Infra.Refactor.dotted_name(cst_expr)
    """

    @staticmethod
    def dotted_name(expr: cst.BaseExpression) -> str:
        """Extract dotted name string from a CST expression.

        Recursively traverses ``Name`` and ``Attribute`` nodes to build
        a fully qualified dotted name (e.g., ``"foo.bar.baz"``).

        Args:
            expr: CST expression node.

        Returns:
            Dotted name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            root = FlextInfraUtilitiesRefactor.dotted_name(expr.value)
            if not root:
                return ""
            return f"{root}.{expr.attr.value}"
        return ""

    @staticmethod
    def root_name(expr: cst.BaseExpression) -> str:
        """Extract the root (leftmost) name from a CST expression.

        Handles ``Name``, ``Attribute``, and ``Call`` nodes to find the
        base identifier.

        Args:
            expr: CST expression node.

        Returns:
            Root name string, or empty string for unsupported nodes.

        """
        if isinstance(expr, cst.Name):
            return expr.value
        if isinstance(expr, cst.Attribute):
            return FlextInfraUtilitiesRefactor.root_name(expr.value)
        if isinstance(expr, cst.Call):
            return FlextInfraUtilitiesRefactor.root_name(expr.func)
        return ""

    @staticmethod
    def asname_to_local(asname: cst.AsName | None) -> str | None:
        """Extract the local alias name from a CST AsName node.

        Args:
            asname: CST AsName node, or None.

        Returns:
            Local alias string, or None if no alias or unsupported node.

        """
        if asname is None:
            return None
        if isinstance(asname.name, cst.Name):
            return asname.name.value
        return None


__all__ = ["FlextInfraUtilitiesRefactor"]
