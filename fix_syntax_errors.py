#!/usr/bin/env python3
"""FLEXT Syntax Error Fixer.

Copyright (c) 2025 Flext. All rights reserved.
SPDX-License-Identifier: MIT

Script for fixing critical syntax errors in FLEXT projects.
"""

import re
from pathlib import Path


def fix_file(file_path: Path) -> bool:
    """Fix syntax errors in a Python file.

    Returns:
        True if file was modified, False otherwise.

    """
    try:
        content = file_path.read_text(encoding="utf-8")

        original_content = content

        # Fix broken class definitions
        content = re.sub(
            r'class (\w+):\s+([^"]*?)\s*"""([^"]*?)"""',
            r'class \1:\n    """\2\3"""',
            content,
            flags=re.MULTILINE | re.DOTALL,
        )

        # Fix broken docstrings: """text"""" -> """text"""
        content = re.sub(r'"""([^"]*?)"""[^"]*?"""', r'"""\1"""', content)

        # Fix broken docstrings: text.""" -> """text."""
        content = re.sub(r'([A-Za-z][^"]*?)\."""', r'"""\1."""', content)

        # Fix missing class inheritance
        content = re.sub(
            r'class (DomainError|ValidationError|RepositoryError|NotFoundError):\s*"""',
            r'class \1(Exception):\n    """',
            content,
        )

        # Fix method definitions without proper body
        content = re.sub(
            r"(async def \w+\([^)]*\)[^:]*:)\s*\.\.\.",
            r"\1\n        ...\n",
            content,
        )
        content = re.sub(
            r"(def \w+\([^)]*\)[^:]*:)\s*\.\.\.",
            r"\1\n        ...\n",
            content,
        )

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True
    except Exception:
        return False
    else:
        return False


def main() -> None:
    """Fix all Python files in the module."""
    src_dir = Path("src")
    if not src_dir.exists():
        return

    fixed_count = 0
    for py_file in src_dir.rglob("*.py"):
        if fix_file(py_file):
            fixed_count += 1


if __name__ == "__main__":
    main()
