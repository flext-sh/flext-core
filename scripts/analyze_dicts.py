#!/usr/bin/env python3
"""Analyze dict[str, ...] usage in flext-core."""

import ast
import sys
from pathlib import Path


def find_dict_annotations(file_path: Path) -> list[tuple[int, str]]:
    """Find all dict[str, ...] type annotations."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, str(file_path))
    except SyntaxError:
        return []

    results = []
    lines = content.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "dict":
                if hasattr(node, "lineno"):
                    line_content = (
                        lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    )
                    results.append((node.lineno, line_content.strip()))

    return results


def analyze_dicts(src_dir: Path) -> dict:
    """Analyze all dict usages."""
    results = {}
    py_files = list(src_dir.rglob("*.py"))

    for file_path in py_files:
        dicts = find_dict_annotations(file_path)
        if dicts:
            rel_path = str(file_path.relative_to(src_dir.parent.parent))
            results[rel_path] = dicts

    return results


if __name__ == "__main__":
    src_path = Path(__file__).parent.parent / "src"
    results = analyze_dicts(src_path)

    total_dicts = sum(len(v) for v in results.values())

    print(
        f"Found {total_dicts} dict[str, ...] annotations in {len(results)} files\n"
    )

    for file_path, dicts in sorted(results.items()):
        print(f"\n{file_path} ({len(dicts)} occurrences):")
        for lineno, line in dicts:
            print(f"  Line {lineno}: {line[:80]}...")

    if total_dicts > 0:
        print("\n⚠️  Review these dicts and classify as BUSINESS or DYNAMIC")
        sys.exit(1)
    else:
        print("\n✅ No dict annotations found")
        sys.exit(0)
