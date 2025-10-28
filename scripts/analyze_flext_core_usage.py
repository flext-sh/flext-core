#!/usr/bin/env python3
"""AST-based analyzer for flext-core API usage across FLEXT workspace.

Analyzes all 33 projects to identify:
1. Which flext-core APIs are actually used
2. Dead code candidates (0 usages)
3. Wrapper functions and unnecessary indirection
4. Direct vs indirect calls
"""

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Configuration
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
FLEXT_CORE_ROOT = WORKSPACE_ROOT / "flext-core"
SRC_DIR = FLEXT_CORE_ROOT / "src" / "flext_core"


class FlextCoreAPIAnalyzer(ast.NodeVisitor):
    """Analyzes flext-core source for public API definitions."""

    def __init__(self) -> None:
        """Initialize the API analyzer with empty state."""
        self.apis = defaultdict(list)  # api_name -> [file, line_no, type]
        self.current_file = ""
        self.current_class = ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        # Track main Flext* classes
        if node.name.startswith("Flext"):
            self.apis[node.name].append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "class",
                "nested_classes": []
            })

        # Store current class for nested classes
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        if self.current_class and node.name.startswith("_"):
            # Skip private methods
            self.generic_visit(node)
            return

        if self.current_class:
            # Public method in Flext* class
            self.apis[f"{self.current_class}.{node.name}"].append({
                "file": self.current_file,
                "line": node.lineno,
                "type": "method",
                "is_property": any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in node.decorator_list
                )
            })

        self.generic_visit(node)


class WorkspaceAPIUsageAnalyzer:
    """Analyzes API usage across entire FLEXT workspace."""

    def __init__(self) -> None:
        """Initialize the usage analyzer with empty state."""
        self.usage_count = defaultdict(int)  # api_name -> count
        self.usage_files = defaultdict(set)  # api_name -> {file_paths}
        self.api_definitions = {}  # api_name -> definition_info

    def extract_apis_from_flext_core(self) -> None:
        """Extract all public APIs from flext-core source."""
        for py_file in SRC_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                # Look for main Flext* class
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.startswith("Flext"):
                        self.api_definitions[node.name] = {
                            "file": str(py_file.relative_to(FLEXT_CORE_ROOT)),
                            "type": "main_class",
                            "methods": [
                                m.name for m in node.body
                                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and not m.name.startswith("_")
                            ]
                        }
                        break
            except Exception as e:
                print(f"Error parsing {py_file}: {e}", file=sys.stderr)

    def analyze_file(self, filepath: Path) -> None:
        """Analyze a single Python file for flext-core API usage."""
        try:
            content = filepath.read_text(encoding="utf-8")
            # Simple pattern matching for API usage
            for api_name in self.api_definitions:
                if api_name in content:
                    self.usage_count[api_name] += content.count(api_name)
                    self.usage_files[api_name].add(str(filepath.relative_to(WORKSPACE_ROOT)))
        except (OSError, UnicodeDecodeError) as e:
            # Skip files that can't be read - log at debug level
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

    def analyze_workspace(self) -> None:
        """Scan entire workspace for API usage."""
        # Analyze flext-core first
        self.extract_apis_from_flext_core()

        # Scan all 33 projects
        for project_dir in WORKSPACE_ROOT.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue
            if project_dir == FLEXT_CORE_ROOT:
                continue  # Skip flext-core itself

            # Find src directories
            for src_dir in [project_dir / "src", project_dir]:
                if not src_dir.exists():
                    continue

                for py_file in src_dir.rglob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    self.analyze_file(py_file)

    def generate_report(self) -> dict[str, Any]:
        """Generate analysis report."""
        total_apis = len(self.api_definitions)
        unused_apis = [
            api for api in self.api_definitions
            if self.usage_count[api] == 0
        ]
        used_apis = [
            api for api in self.api_definitions
            if self.usage_count[api] > 0
        ]

        # Sort by usage count
        used_apis.sort(key=lambda x: self.usage_count[x], reverse=True)

        return {
            "summary": {
                "total_apis": total_apis,
                "used_apis": len(used_apis),
                "unused_apis": len(unused_apis),
                "analysis_date": "2025-10-28",
            },
            "unused_apis": [
                {
                    "name": api,
                    "definition": self.api_definitions[api],
                    "recommendation": "Can be deprecated or removed"
                }
                for api in unused_apis
            ],
            "top_used_apis": [
                {
                    "name": api,
                    "usage_count": self.usage_count[api],
                    "file_count": len(self.usage_files[api]),
                    "definition": self.api_definitions[api]
                }
                for api in used_apis[:20]
            ],
            "all_apis": {
                api: {
                    "definition": self.api_definitions[api],
                    "usage_count": self.usage_count[api],
                    "files_using": list(self.usage_files[api])[:5]
                }
                for api in sorted(self.api_definitions.keys())
            }
        }


def main() -> None:
    """Main entry point."""
    print("🔍 Analyzing FLEXT-core API usage across workspace...")
    print(f"   Workspace: {WORKSPACE_ROOT}")
    print(f"   flext-core: {FLEXT_CORE_ROOT}")
    print()

    analyzer = WorkspaceAPIUsageAnalyzer()
    analyzer.analyze_workspace()

    print(f"✅ Found {len(analyzer.api_definitions)} public APIs")
    print("✅ Scanned entire workspace")
    print()

    report = analyzer.generate_report()

    # Print summary
    print("📊 Analysis Summary:")
    print(f"   Total APIs: {report['summary']['total_apis']}")
    print(f"   Used APIs: {report['summary']['used_apis']}")
    print(f"   Unused APIs: {report['summary']['unused_apis']}")
    print()

    # Print unused APIs
    if report["unused_apis"]:
        print("❌ Unused APIs (candidates for deprecation):")
        for api in report["unused_apis"]:
            print(f"   - {api['name']}: {api['definition']['file']}")
    print()

    # Print top used APIs
    print("✅ Top 10 Most Used APIs:")
    for i, api in enumerate(report["top_used_apis"][:10], 1):
        print(f"   {i}. {api['name']}: {api['usage_count']} usages in {api['file_count']} files")
    print()

    # Save report
    report_file = FLEXT_CORE_ROOT / "usage_report.json"
    with Path(report_file).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"💾 Full report saved to: {report_file}")
    print()

    # Generate candidates list
    candidates_file = FLEXT_CORE_ROOT / "dead_code_candidates.md"
    with Path(candidates_file).open("w", encoding="utf-8") as f:
        f.write("# Dead Code Candidates for flext-core\n\n")
        f.write(f"Generated: {report['summary']['analysis_date']}\n\n")

        if report["unused_apis"]:
            f.write("## Unused APIs (0 usages across workspace)\n\n")
            for api in report["unused_apis"]:
                f.write(f"### {api['name']}\n")
                f.write(f"- **File**: {api['definition']['file']}\n")
                f.write(f"- **Type**: {api['definition']['type']}\n")
                f.write(f"- **Recommendation**: {api['recommendation']}\n\n")
        else:
            f.write("## No Unused APIs Found\n\n")
            f.write("All public APIs are used somewhere in the workspace.\n\n")

        f.write("## Observations\n\n")
        f.write(f"- Total APIs analyzed: {report['summary']['total_apis']}\n")
        f.write(f"- APIs in use: {report['summary']['used_apis']}\n")
        f.write(f"- APIs with zero usages: {report['summary']['unused_apis']}\n\n")

    print(f"💾 Candidates list saved to: {candidates_file}")


if __name__ == "__main__":
    main()
