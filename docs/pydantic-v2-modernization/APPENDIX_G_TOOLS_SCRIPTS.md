# Appendix G: Tools and Scripts Directory

**Status**: AUTOMATION TOOLS
**Purpose**: Collection of all automated tools and scripts for migration
**Usage**: Copy and run these scripts during migration

---

## Table of Contents

1. [Audit Script (Python)](#audit-script-python)
2. [Pre-Commit Hook](#pre-commit-hook)
3. [Migration Helper Script](#migration-helper-script)
4. [Performance Benchmark Suite](#performance-benchmark-suite)
5. [Quick Fix Scripts](#quick-fix-scripts)

---

## Audit Script (Python)

**Purpose**: Automated audit of Pydantic v2 compliance across all projects

**Location**: `~/flext/scripts/audit_pydantic_v2.py`

```python
#!/usr/bin/env python3
"""Audit FLEXT projects for Pydantic v2 compliance.

Usage:
    python scripts/audit_pydantic_v2.py                # All projects
    python scripts/audit_pydantic_v2.py flext-core     # Single project
    python scripts/audit_pydantic_v2.py --summary      # Summary only
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

LEGACY_PATTERNS = {
    r"class\s+Config:": {
        "message": "Use model_config = ConfigDict()",
        "severity": "error",
        "category": "config",
    },
    r"\.dict\(": {
        "message": "Use .model_dump()",
        "severity": "error",
        "category": "serialization",
    },
    r"\.json\(": {
        "message": "Use .model_dump_json()",
        "severity": "error",
        "category": "serialization",
    },
    r"parse_obj\(": {
        "message": "Use .model_validate()",
        "severity": "error",
        "category": "validation",
    },
    r"@validator\(": {
        "message": "Use @field_validator()",
        "severity": "error",
        "category": "validators",
    },
    r"@root_validator": {
        "message": "Use @model_validator()",
        "severity": "error",
        "category": "validators",
    },
}

PERFORMANCE_PATTERNS = {
    r"json\.loads.*model_validate": {
        "message": "Use model_validate_json() for performance",
        "severity": "warning",
        "category": "performance",
    },
    r"TypeAdapter\(.+\).*\n.*def ": {
        "message": "Move TypeAdapter to module level",
        "severity": "warning",
        "category": "performance",
    },
}

def audit_project(project_path: Path) -> Dict:
    """Audit single project."""
    results = {"errors": [], "warnings": [], "info": []}
    
    if not (project_path / "src").exists():
        results["warnings"].append(f"No src/ directory")
        return results
    
    # Check legacy patterns
    for pattern, config in LEGACY_PATTERNS.items():
        cmd = ["grep", "-rEn", pattern, str(project_path / "src")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    msg = f"[{config['category'].upper()}] {line}: {config['message']}"
                    results[f"{config['severity']}s"].append(msg)
    
    # Check performance patterns
    for pattern, config in PERFORMANCE_PATTERNS.items():
        cmd = ["grep", "-rEn", pattern, str(project_path / "src")]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line:
                    msg = f"[{config['category'].upper()}] {line}: {config['message']}"
                    results["warnings"].append(msg)
    
    return results

def main():
    """Main audit execution."""
    workspace = Path(__file__).parent.parent
    
    if len(sys.argv) > 1 and sys.argv[1] not in ['--summary', '--fix']:
        projects = [workspace / sys.argv[1]]
    else:
        projects = sorted([
            p.parent for p in workspace.glob("flext-*/pyproject.toml")
        ])
    
    summary = {}
    total_errors = 0
    total_warnings = 0
    
    for project in projects:
        print(f"\n{'='*70}")
        print(f"Auditing: {project.name}")
        print(f"{'='*70}")
        
        results = audit_project(project)
        summary[project.name] = results
        
        if results['errors']:
            print(f"\n❌ ERRORS ({len(results['errors'])})")
            for error in results['errors'][:10]:  # Show first 10
                print(f"  {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")
            total_errors += len(results['errors'])
        else:
            print("\n✅ No errors")
        
        if results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(results['warnings'])})")
            for warning in results['warnings'][:5]:  # Show first 5
                print(f"  {warning}")
            if len(results['warnings']) > 5:
                print(f"  ... and {len(results['warnings']) - 5} more")
            total_warnings += len(results['warnings'])
    
    # Summary
    print(f"\n{'='*70}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Projects: {len(summary)}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Warnings: {total_warnings}")
    
    # Project breakdown
    print(f"\nProject Status:")
    for name, results in summary.items():
        status = "✅" if not results['errors'] else "❌"
        print(f"  {status} {name}: {len(results['errors'])} errors, {len(results['warnings'])} warnings")
    
    sys.exit(1 if total_errors > 0 else 0)

if __name__ == "__main__":
    main()
```

**Make executable**:
```bash
chmod +x ~/flext/scripts/audit_pydantic_v2.py
```

**Usage**:
```bash
cd ~/flext
python scripts/audit_pydantic_v2.py                  # All projects
python scripts/audit_pydantic_v2.py flext-core       # Single project
python scripts/audit_pydantic_v2.py > report.txt     # Save report
```

---

## Pre-Commit Hook

**Purpose**: Prevent Pydantic v1 code from being committed

**Location**: `.git/hooks/pre-commit` or use `pre-commit` framework

```bash
#!/bin/bash
# Pre-commit hook for Pydantic v2 compliance

echo "Checking Pydantic v2 compliance..."

# Get list of Python files being committed
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$files" ]; then
    exit 0
fi

# Check for legacy patterns
errors=0

for file in $files; do
    # Check for class Config
    if grep -q "class Config:" "$file"; then
        echo "❌ $file: Found 'class Config:', use model_config = ConfigDict()"
        errors=$((errors+1))
    fi

    # Check for .dict()
    if grep -q "\.dict()" "$file"; then
        echo "❌ $file: Found '.dict()', use .model_dump()"
        errors=$((errors+1))
    fi

    # Check for .json()
    if grep -q "\.json()" "$file"; then
        echo "❌ $file: Found '.json()', use .model_dump_json()"
        errors=$((errors+1))
    fi

    # Check for parse_obj()
    if grep -q "parse_obj(" "$file"; then
        echo "❌ $file: Found 'parse_obj()', use .model_validate()"
        errors=$((errors+1))
    fi

    # Check for @validator (not @field_validator)
    if grep -q "@validator" "$file" && ! grep -q "@field_validator" "$file"; then
        echo "❌ $file: Found '@validator', use @field_validator"
        errors=$((errors+1))
    fi
done

if [ $errors -gt 0 ]; then
    echo ""
    echo "Found $errors Pydantic v1 pattern(s). Please fix before committing."
    echo "See: docs/pydantic-v2-modernization/APPENDIX_C_COMMON_ERRORS.md"
    exit 1
fi

echo "✅ Pydantic v2 compliance check passed!"
exit 0
```

**Install**:
```bash
# Option 1: Direct installation
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Option 2: Using pre-commit framework (recommended)
# Add to .pre-commit-config.yaml
```

---

## Migration Helper Script

**Purpose**: Automated find-and-replace for common patterns

**Location**: `~/flext/scripts/migrate_pydantic_v2.py`

```python
#!/usr/bin/env python3
"""Automated migration helper for Pydantic v1 → v2.

IMPORTANT: Review changes before committing!

Usage:
    python scripts/migrate_pydantic_v2.py path/to/file.py
    python scripts/migrate_pydantic_v2.py src/  # Entire directory
"""

import re
import sys
from pathlib import Path

REPLACEMENTS = [
    # Serialization methods
    (r"\.dict\(", ".model_dump("),
    (r"\.json\(", ".model_dump_json("),
    (r"\.copy\(", ".model_copy("),
    
    # Validation methods
    (r"\.parse_obj\(", ".model_validate("),
    (r"\.parse_raw\(", ".model_validate_json("),
    
    # Validators
    (r"@validator\(", "@field_validator("),
    (r"@root_validator", "@model_validator"),
    
    # Config
    (r"class Config:", "model_config = ConfigDict(  # TODO: convert to dict"),
]

def migrate_file(file_path: Path, dry_run: bool = True):
    """Migrate single file."""
    if not file_path.is_file():
        return
    
    content = file_path.read_text()
    original = content
    
    # Apply replacements
    for pattern, replacement in REPLACEMENTS:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        if dry_run:
            print(f"Would modify: {file_path}")
        else:
            file_path.write_text(content)
            print(f"✅ Modified: {file_path}")
    else:
        print(f"⏭  Skipped (no changes): {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_pydantic_v2.py <path>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    dry_run = "--apply" not in sys.argv
    
    if dry_run:
        print("DRY RUN MODE - Use --apply to make changes")
        print()
    
    if path.is_file():
        migrate_file(path, dry_run)
    elif path.is_dir():
        for py_file in path.rglob("*.py"):
            migrate_file(py_file, dry_run)
    else:
        print(f"Error: {path} not found")
        sys.exit(1)
    
    if dry_run:
        print()
        print("Run with --apply to make changes")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Dry run (preview only)
python scripts/migrate_pydantic_v2.py src/flext_core/models.py

# Apply changes
python scripts/migrate_pydantic_v2.py src/flext_core/models.py --apply

# Entire directory
python scripts/migrate_pydantic_v2.py src/ --apply
```

---

## Performance Benchmark Suite

**Purpose**: Measure performance improvements from Pydantic v2

**Location**: `tests/performance/test_pydantic_benchmarks.py`

```python
import pytest
import json
from pydantic import BaseModel, TypeAdapter

class BenchmarkModel(BaseModel):
    field1: str
    field2: int
    field3: float

# Benchmark 1: JSON Parsing
@pytest.mark.benchmark(group="json-parsing")
def test_model_validate_json_fast(benchmark):
    """model_validate_json() - one-pass Rust parsing."""
    json_str = '{"field1": "value", "field2": 123, "field3": 45.6}'
    result = benchmark(BenchmarkModel.model_validate_json, json_str)
    assert result.field2 == 123

@pytest.mark.benchmark(group="json-parsing")  
def test_json_loads_slow(benchmark):
    """json.loads() + model_validate() - two-pass parsing."""
    json_str = '{"field1": "value", "field2": 123, "field3": 45.6}'
    
    def old_way(s):
        data = json.loads(s)
        return BenchmarkModel.model_validate(data)
    
    result = benchmark(old_way, json_str)
    assert result.field2 == 123

# Benchmark 2: TypeAdapter Reuse
_ADAPTER = TypeAdapter(list[BenchmarkModel])

@pytest.mark.benchmark(group="type-adapter")
def test_type_adapter_reuse_fast(benchmark):
    """Module-level TypeAdapter (created once)."""
    data = [{"field1": "a", "field2": 1, "field3": 1.1}] * 100
    result = benchmark(_ADAPTER.validate_python, data)
    assert len(result) == 100

@pytest.mark.benchmark(group="type-adapter")
def test_type_adapter_recreate_slow(benchmark):
    """TypeAdapter created every call."""
    data = [{"field1": "a", "field2": 1, "field3": 1.1}] * 100
    
    def with_recreate(d):
        adapter = TypeAdapter(list[BenchmarkModel])
        return adapter.validate_python(d)
    
    result = benchmark(with_recreate, data)
    assert len(result) == 100
```

**Run benchmarks**:
```bash
cd flext-core
PYTHONPATH=src poetry run pytest tests/performance/ --benchmark-only

# Compare before/after
poetry run pytest tests/performance/ --benchmark-compare=0001
```

---

## Quick Fix Scripts

### Fix Redundant Type Casts

```python
#!/usr/bin/env python3
"""Remove redundant type casts flagged by Pyrefly."""

import re
import sys
from pathlib import Path

def remove_redundant_casts(file_path: Path):
    """Remove cast() calls for types that don't need them."""
    content = file_path.read_text()
    
    # Remove cast for obvious types
    patterns = [
        (r"cast\(str, (['\"])([^'\"]+)\1\)", r"\1\2\1"),  # cast(str, "value") → "value"
        (r"cast\(int, (\d+)\)", r"\1"),                   # cast(int, 123) → 123
        (r"cast\(bool, (True|False)\)", r"\1"),           # cast(bool, True) → True
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    file_path.write_text(content)
    print(f"✅ Fixed: {file_path}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        remove_redundant_casts(Path(arg))
```

---

### Convert Config to ConfigDict

```python
#!/usr/bin/env python3
"""Convert class Config to model_config = ConfigDict()."""

import re
from pathlib import Path

def convert_config(file_path: Path):
    """Convert class Config pattern."""
    content = file_path.read_text()
    
    # Find class Config blocks
    pattern = r"class Config:\s+(.*?)\n\s*\n"
    
    def replacer(match):
        config_body = match.group(1)
        # Convert to ConfigDict format (manual review needed!)
        return f"model_config = ConfigDict(  # TODO: Review\n        {config_body}\n    )\n\n"
    
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Add import if needed
    if "ConfigDict" in content and "from pydantic import" in content:
        content = re.sub(
            r"from pydantic import (.*?)\n",
            r"from pydantic import \1, ConfigDict\n",
            content
        )
    
    file_path.write_text(content)
    print(f"✅ Converted: {file_path} (MANUAL REVIEW NEEDED!)")

if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        convert_config(Path(arg))
```

---

## All Scripts Summary

| Script | Purpose | Location | Usage |
|--------|---------|----------|-------|
| `audit_pydantic_v2.py` | Audit compliance | `scripts/` | Check all projects |
| `pre-commit` | Prevent v1 code | `.git/hooks/` | Automatic on commit |
| `migrate_pydantic_v2.py` | Auto-replace patterns | `scripts/` | Quick migration |
| `test_pydantic_benchmarks.py` | Performance tests | `tests/performance/` | pytest --benchmark |
| `remove_redundant_casts.py` | Fix type casts | `scripts/` | One-time fix |
| `convert_config.py` | Config → ConfigDict | `scripts/` | One-time fix |

---

**Installation**:
```bash
# Create scripts directory
mkdir -p ~/flext/scripts

# Copy all scripts (use cat commands from this appendix)

# Make executable
chmod +x ~/flext/scripts/*.py
```

**Next**: [Appendix H: References](./APPENDIX_H_REFERENCES.md)
