#!/usr/bin/env python3
"""Detect circular import dependencies in flext-core."""

import ast
import sys
from collections import defaultdict
from pathlib import Path


def find_imports(file_path: Path) -> set[str]:
    """Extract all imports from a Python file."""
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            tree = ast.parse(f.read(), str(file_path))
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    return imports


def detect_cycles(src_dir: Path) -> list[tuple[str, list[str]]]:
    """Detect circular dependencies."""
    # Build dependency graph
    deps = defaultdict(set)
    py_files = list(src_dir.rglob("*.py"))

    for file_path in py_files:
        if file_path.stem == "__init__":
            continue
        module_name = (
            str(file_path.relative_to(src_dir)).replace("/", ".").replace(".py", "")
        )
        imports = find_imports(file_path)
        for imp in imports:
            if imp.startswith("flext_core"):
                deps[module_name].add(imp)

    # Find cycles using DFS
    cycles = []
    visited = set()
    rec_stack = []

    def dfs(node: str) -> bool:
        if node in rec_stack:
            cycle_start = rec_stack.index(node)
            cycles.append((node, rec_stack[cycle_start:] + [node]))
            return True

        if node in visited:
            return False

        visited.add(node)
        rec_stack.append(node)

        for neighbor in deps.get(node, []):
            if dfs(neighbor):
                return True

        rec_stack.pop()
        return False

    for module in list(deps.keys()):
        if module not in visited:
            dfs(module)

    return cycles


if __name__ == "__main__":
    src_path = Path(__file__).parent.parent / "src" / "flext_core"
    cycles = detect_cycles(src_path)

    if cycles:
        print(f"❌ Found {len(cycles)} circular dependencies:")
        for i, (start, path) in enumerate(cycles, 1):
            print(f"\n{i}. Cycle starting at {start}:")
            print(" → ".join(path))
        sys.exit(1)
    else:
        print("✅ No circular dependencies detected")
        sys.exit(0)
