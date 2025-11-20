# EPIC-00: Baseline & Mapping

**Phase**: 0
**Duration**: Week 0 (2-3 days)
**Risk**: 🟢 Low
**LOC Impact**: 0 (no code changes)
**Dependencies**: None
**Status**: 🔵 READY TO START

---

## 🎯 OBJECTIVE

Establish baseline metrics and create comprehensive maps of:
- Current code metrics (LOC, complexity)
- Circular dependencies
- Business dict usage
- TYPE_CHECKING occurrences

**Why This Matters**: Without baseline, we can't measure success. This phase is purely observational.

---

## 📋 TASKS CHECKLIST

### Task 0.1: Directory Structure Setup

- [ ] Create `docs/metrics/` directory
- [ ] Create `scripts/` directory (if not exists)
- [ ] Verify write permissions for metric files

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
mkdir -p docs/metrics
mkdir -p scripts
```

---

### Task 0.2: Code Metrics Baseline

- [ ] Run `cloc` on entire codebase
- [ ] Save output to `docs/metrics/baseline_loc.txt`
- [ ] Extract key metrics:
  - Total lines of code
  - Lines per module (dispatcher, exceptions, decorators, services)
  - Comment ratio
  - Blank line ratio

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
cloc src/ --by-file --csv > docs/metrics/baseline_loc.csv
cloc src/ > docs/metrics/baseline_loc.txt

# Key modules
cloc src/flext_core/dispatcher.py > docs/metrics/baseline_dispatcher_loc.txt
cloc src/flext_core/exceptions.py > docs/metrics/baseline_exceptions_loc.txt
cloc src/flext_core/decorators.py > docs/metrics/baseline_decorators_loc.txt
cloc src/flext_core/services.py > docs/metrics/baseline_services_loc.txt
```

**Expected Baselines** (approximate):
- Total LOC: 38,000-38,500
- dispatcher.py: ~1,200 LOC
- exceptions.py: ~600 LOC
- decorators.py: ~800 LOC
- services.py: ~1,500 LOC

---

### Task 0.3: Circular Dependency Detection

- [ ] Create or verify `scripts/detect_cycles.py` exists
- [ ] Run cycle detection script
- [ ] Save results to `docs/metrics/cycles_baseline.txt`
- [ ] Document all cycles found with import paths

**Script** (if doesn't exist):
```python
#!/usr/bin/env python3
"""Detect circular import dependencies in flext-core."""

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, List, Tuple

def find_imports(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), str(file_path))
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    return imports

def detect_cycles(src_dir: Path) -> List[Tuple[str, List[str]]]:
    """Detect circular dependencies."""
    # Build dependency graph
    deps = defaultdict(set)
    py_files = list(src_dir.rglob("*.py"))

    for file_path in py_files:
        if file_path.stem == "__init__":
            continue
        module_name = str(file_path.relative_to(src_dir)).replace('/', '.').replace('.py', '')
        imports = find_imports(file_path)
        for imp in imports:
            if imp.startswith('flext_core'):
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
```

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
python scripts/detect_cycles.py > docs/metrics/cycles_baseline.txt 2>&1
```

---

### Task 0.4: Business Dict Analysis

- [ ] Create or verify `scripts/analyze_dicts.py` exists
- [ ] Scan for `dict[str, object]` / `dict[str, Any]` usage
- [ ] Classify each occurrence as:
  - **BUSINESS** (needs model) - configs, options, contexts
  - **DYNAMIC** (can stay) - logging, tracing, free-form
- [ ] Save results to `docs/metrics/dicts_baseline.txt`

**Script** (if doesn't exist):
```python
#!/usr/bin/env python3
"""Analyze dict[str, Any] usage in flext-core."""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

def find_dict_annotations(file_path: Path) -> List[Tuple[int, str]]:
    """Find all dict[str, ...] type annotations."""
    try:
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content, str(file_path))
    except SyntaxError:
        return []

    results = []
    lines = content.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == 'dict':
                if hasattr(node, 'lineno'):
                    line_content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
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

    print(f"Found {total_dicts} dict[str, ...] annotations in {len(results)} files\n")

    for file_path, dicts in sorted(results.items()):
        print(f"\n{file_path} ({len(dicts)} occurrences):")
        for lineno, line in dicts:
            print(f"  Line {lineno}: {line[:80]}...")

    if total_dicts > 0:
        print(f"\n⚠️  Review these dicts and classify as BUSINESS or DYNAMIC")
        sys.exit(1)
    else:
        print("\n✅ No dict annotations found")
        sys.exit(0)
```

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
python scripts/analyze_dicts.py > docs/metrics/dicts_baseline.txt 2>&1
```

---

### Task 0.5: TYPE_CHECKING Analysis

- [ ] Search for all `TYPE_CHECKING` occurrences
- [ ] Document file, line, and context
- [ ] Save to `docs/metrics/type_checking_baseline.txt`

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
grep -rn "TYPE_CHECKING" src/ > docs/metrics/type_checking_baseline.txt
grep -c "TYPE_CHECKING" src/**/*.py | grep -v ":0$" >> docs/metrics/type_checking_baseline.txt
```

**Expected Baseline**: Several occurrences (exact count TBD)

---

### Task 0.6: Protocol Inventory

- [ ] List all existing protocols in `protocols.py` (or equivalent)
- [ ] Document their current usage patterns
- [ ] Identify gaps (protocols that should exist but don't)
- [ ] Save to `docs/metrics/protocols_baseline.txt`

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
find src/ -name "*protocol*.py" -o -name "protocols.py" > docs/metrics/protocols_baseline.txt
grep -r "@runtime_checkable" src/ >> docs/metrics/protocols_baseline.txt
grep -r "class.*Protocol" src/ >> docs/metrics/protocols_baseline.txt
```

---

### Task 0.7: Duplicate Pattern Detection

- [ ] Search for duplicated isinstance patterns
- [ ] Search for duplicated conversion patterns (model ↔ dict)
- [ ] Search for methods with 6+ parameters
- [ ] Save findings to `docs/metrics/duplicates_baseline.txt`

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core

# Find isinstance(*, BaseModel) patterns
grep -rn "isinstance.*BaseModel" src/ > docs/metrics/duplicates_baseline.txt

# Find isinstance(*, dict) patterns
grep -rn "isinstance.*dict" src/ >> docs/metrics/duplicates_baseline.txt

# Find model_dump() calls
grep -rn "model_dump()" src/ >> docs/metrics/duplicates_baseline.txt

# Find methods with many parameters (manual review needed)
echo "\n=== Methods with 6+ parameters (manual review) ===" >> docs/metrics/duplicates_baseline.txt
```

---

### Task 0.8: Classification Report

- [ ] Review all dict occurrences from Task 0.4
- [ ] Manually classify each as BUSINESS or DYNAMIC
- [ ] Create `docs/metrics/dict_classification.md` with rationale
- [ ] Count: How many need to become models?

**Template** (`docs/metrics/dict_classification.md`):
```markdown
# Dict Classification Report

## Summary
- Total dicts found: XX
- Business (need models): XX
- Dynamic (can stay): XX

## Business Dicts (TO BE CONVERTED)

### src/flext_core/dispatcher.py:123
**Current**: `context: dict[str, Any]`
**Rationale**: Used for handler execution context with structured fields (handler_name, timeout, metadata)
**Proposed Model**: `DispatchContext`

### src/flext_core/decorators.py:456
**Current**: `retry_config: dict[str, Any]`
**Rationale**: Retry configuration with max_attempts, backoff, etc.
**Proposed Model**: `RetryOptions`

## Dynamic Dicts (CAN STAY)

### src/flext_core/logging.py:78
**Current**: `extra: dict[str, Any]`
**Rationale**: Free-form logging metadata
**Decision**: KEEP (logging context is inherently dynamic)
```

---

### Task 0.9: Create Baseline Summary

- [ ] Consolidate all metrics into `docs/metrics/baseline_summary.md`
- [ ] Include:
  - Total LOC
  - Circular dependencies count
  - Business dicts count
  - TYPE_CHECKING count
  - Duplicate pattern counts
- [ ] Add timestamp and commit SHA

**Template**:
```markdown
# Baseline Summary

**Date**: 2025-11-20
**Commit**: <git-sha>
**Branch**: main

## Code Metrics
- **Total LOC**: 38,XXX
- **Python LOC**: 35,XXX
- **Comments**: X,XXX
- **Key Modules**:
  - dispatcher.py: X,XXX LOC
  - exceptions.py: XXX LOC
  - decorators.py: XXX LOC
  - services.py: X,XXX LOC

## Architecture Issues
- **Circular Dependencies**: X found
- **Business Dicts**: X found (need models)
- **Dynamic Dicts**: X found (can stay)
- **TYPE_CHECKING**: X occurrences

## Duplication Patterns
- **isinstance(*, BaseModel)**: X occurrences
- **isinstance(*, dict)**: X occurrences
- **model_dump() calls**: X occurrences
- **Methods with 6+ params**: X found (manual review)

## Targets for Improvement
- [ ] Eliminate X circular dependencies
- [ ] Convert X business dicts to models
- [ ] Remove X TYPE_CHECKING blocks
- [ ] Reduce LOC by 500-800 (target: ~37,500-38,000)
```

**Commands**:
```bash
cd /home/marlonsc/flext/flext-core
git rev-parse HEAD > docs/metrics/baseline_commit.txt
date -Iseconds >> docs/metrics/baseline_commit.txt
```

---

## ✅ QUALITY GATES

### Definition of Done

- [ ] All 9 tasks completed
- [ ] All metric files created in `docs/metrics/`
- [ ] Baseline summary document complete
- [ ] Dict classification complete (every dict categorized)
- [ ] Scripts validated (detect_cycles.py, analyze_dicts.py work)
- [ ] Commit created: `chore(core): add baseline metrics for automation plan`

### Validation Steps

```bash
cd /home/marlonsc/flext/flext-core

# Verify all files exist
test -f docs/metrics/baseline_loc.txt && echo "✅ LOC baseline"
test -f docs/metrics/cycles_baseline.txt && echo "✅ Cycles baseline"
test -f docs/metrics/dicts_baseline.txt && echo "✅ Dicts baseline"
test -f docs/metrics/type_checking_baseline.txt && echo "✅ TYPE_CHECKING baseline"
test -f docs/metrics/protocols_baseline.txt && echo "✅ Protocols baseline"
test -f docs/metrics/duplicates_baseline.txt && echo "✅ Duplicates baseline"
test -f docs/metrics/dict_classification.md && echo "✅ Dict classification"
test -f docs/metrics/baseline_summary.md && echo "✅ Baseline summary"

# Verify scripts work
python scripts/detect_cycles.py && echo "✅ Cycle detection works"
python scripts/analyze_dicts.py && echo "✅ Dict analysis works"
```

---

## 📊 SUCCESS METRICS

### Quantitative
- All metric files generated
- All scripts executable and tested
- Dict classification: 100% coverage

### Qualitative
- Clear understanding of baseline state
- Actionable insights for next phases
- No ambiguity in classification

---

## 🔗 DEPENDENCIES

### Required Before Starting
- None (this is Phase 0)

### Outputs Used By
- **Phase 1**: Duplicate patterns inform helper design
- **Phase 2**: Business dicts inform model creation
- **Phase 3**: TYPE_CHECKING informs protocol usage
- **Phase 5**: LOC baselines inform reduction targets
- **Phase 7**: All metrics used for final comparison

---

## ⚠️ RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scripts fail on edge cases | Medium | Test on sample files first |
| Manual classification errors | Low | Peer review classification report |
| Baseline drift (code changes during mapping) | Low | Capture git SHA, re-run if needed |

---

## 📝 NOTES

- This phase is **read-only**: no code changes
- All scripts should be idempotent (safe to re-run)
- If baseline changes during execution, re-capture metrics
- Classification is subjective: document rationale for each decision

---

**Next Phase**: [EPIC-01: Automation Core + Helpers](./EPIC-01-automation-core.md)

**Status**: 🔵 READY TO START
