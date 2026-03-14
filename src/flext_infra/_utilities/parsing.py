"""Parsing utilities for infrastructure code analysis.

Provides safe AST and CST parsing as static methods.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

import libcst as cst

from flext_infra.constants import FlextInfraConstants as c


class FlextInfraUtilitiesParsing:
    """Static parsing utilities for Python source analysis.

    All methods are ``@staticmethod`` — no instantiation required.
    Exposed via ``u.Infra.parse_module_ast()`` through MRO.
    """

    @staticmethod
    def parse_module_ast(file_path: Path) -> ast.Module | None:
        """Parse a Python file into an AST module, returning None on error.

        Args:
            file_path: Path to the Python file.

        Returns:
            Parsed AST module, or None on read/parse error.

        """
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            return ast.parse(source)
        except (OSError, UnicodeDecodeError, SyntaxError):
            return None

    @staticmethod
    def parse_ast_from_source(source: str) -> ast.Module | None:
        """Parse a source string into an AST module, returning None on error.

        Use when the caller already has source text (e.g. for rewriting,
        ``ast.get_source_segment``, or in-memory transformations).

        Args:
            source: Python source code string.

        Returns:
            Parsed AST module, or None on syntax error.

        """
        try:
            return ast.parse(source)
        except SyntaxError:
            return None

    @staticmethod
    def parse_cst_from_source(source: str) -> cst.Module | None:
        """Parse a source string into a CST module, returning None on error.

        Use when the caller already has source text (e.g. for CST
        transformations or in-memory rewrites).

        Args:
            source: Python source code string.

        Returns:
            Parsed CST module, or None on parse error.

        """
        try:
            return cst.parse_module(source)
        except cst.ParserSyntaxError:
            return None

    @staticmethod
    def parse_module_cst(file_path: Path) -> cst.Module | None:
        """Parse a Python file into a CST module, returning None on error.

        Args:
            file_path: Path to the Python file.

        Returns:
            Parsed CST module, or None on read/parse error.

        """
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            return cst.parse_module(source)
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
            return None


__all__ = ["FlextInfraUtilitiesParsing"]
