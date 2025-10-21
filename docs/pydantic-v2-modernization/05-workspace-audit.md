# Part 5: Workspace-Wide Pydantic v2 Audit

**Status**: ECOSYSTEM ROLLOUT
**Priority**: 🟡 MEDIUM (After flext-core complete)
**Estimated Time**: 1 week (distributed across team)
**Impact**: Ensures all 33 projects follow standards

---

## Overview

Audit and modernize all 33 FLEXT projects for Pydantic v2 compliance:
- Automated audit script
- Project prioritization
- Compliance checklist
- Migration coordination

---

## Project Inventory

### Foundation (HIGH PRIORITY)
1. ✅ **flext-core** (THIS PLAN) - Foundation library
2. ⏳ **flext-cli** - CLI foundation
3. ⏳ **flext-ldif** - LDIF processing
4. ⏳ **flext-ldap** - LDAP operations
5. ⏳ **flext-api** - REST API framework

### Domain Libraries (MEDIUM PRIORITY)
6. ⏳ **flext-auth** - Authentication
7. ⏳ **flext-web** - Web framework
8. ⏳ **flext-grpc** - gRPC services
9. ⏳ **flext-meltano** - Meltano integration
10. ⏳ **flext-observability** - Monitoring
11. ⏳ **flext-quality** - Quality tools
12. ⏳ **flext-plugin** - Plugin system

### Singer Platform (MEDIUM PRIORITY)
**Taps** (6 projects):
13-18. ⏳ flext-tap-{ldap, ldif, oracle, oracle-oic, oracle-wms}

**Targets** (6 projects):
19-24. ⏳ flext-target-{ldap, ldif, oracle, oracle-oic, oracle-wms}

**DBT** (4 projects):
25-28. ⏳ flext-dbt-{ldap, ldif, oracle, oracle-wms}

**Database** (3 projects):
29-31. ⏳ flext-db-oracle, flext-dbt-oracle, flext-dbt-oracle-wms

### Enterprise (LOW PRIORITY)
32. ⏳ **client-a-oud-mig** - Oracle migration
33. ⏳ **client-b-meltano-native** - Custom Meltano

---

## Pydantic v2 Compliance Checklist

### For Each Project

```markdown
## Project: _______________
**Date**: _______________
**Auditor**: _______________

### Code Patterns - CRITICAL CHECKS
- [ ] No `class Config:` (Pydantic v1 - FORBIDDEN)
- [ ] No `.dict()` method calls (use `model_dump()`)
- [ ] No `.json()` method calls (use `model_dump_json()`)
- [ ] No `parse_obj()` calls (use `model_validate()`)
- [ ] No `@validator` decorator (use `@field_validator`)
- [ ] No `@root_validator` decorator (use `@model_validator`)
- [ ] All models use `ConfigDict` for configuration
- [ ] All serialization uses `model_dump()` or `model_dump_json()`
- [ ] All deserialization uses `model_validate()` or `model_validate_json()`
- [ ] Uses `@field_validator(mode="before"/"after")` correctly
- [ ] Uses `@model_validator(mode="before"/"after")` correctly

### Code Quality - NO DUPLICATION
- [ ] No custom string validators (use `Field(min_length, max_length, pattern)`)
- [ ] No custom numeric validators (use `Field(ge, le, gt, lt)`)
- [ ] No custom email validators (use `EmailStr` built-in or custom Annotated)
- [ ] No custom URL validators (use `HttpUrl` built-in)
- [ ] Audit output: `grep -r "def validate_" src/ --include="*.py"` returns only business logic

### Type Safety - ANNOTATED PATTERN
- [ ] Uses `Annotated[T, Field(...)]` for constraints (NOT `T = Field(...)`)
- [ ] Uses Pydantic built-in types: `EmailStr`, `HttpUrl`, `FilePath`, etc.
- [ ] Uses FlextTypes domain types: `PortNumber`, `TimeoutSeconds`, etc.
- [ ] Constraint metadata in typings.py, NOT scattered in models

### Performance - RUST OPTIMIZATION
- [ ] All JSON parsing uses `model_validate_json()` (NOT `json.loads()` + `model_validate()`)
- [ ] Module-level TypeAdapter constants (NOT inside functions)
- [ ] Tagged unions use `Discriminator` (NOT plain unions)
- [ ] Benchmark: measure JSON parse time vs previous version

### Documentation - TEAM ENABLEMENT
- [ ] CLAUDE.md includes Pydantic v2 standards section
- [ ] README.md mentions Pydantic v2 adoption
- [ ] Examples use current Pydantic v2 patterns
- [ ] No references to Pydantic v1 patterns

### Quality Gates - AUTOMATED ENFORCEMENT
- [ ] `make lint` passes (0 violations)
- [ ] `make type-check` passes (0 errors)
- [ ] `make test` passes (100% pass rate or documented reason)
- [ ] `make validate` passes (all gates together)

### Audit Output Template

**File**: Project audit report
**Format**: Automated script output with violations listed

```
PROJECT_AUDIT_REPORT
====================
Project: <name>
Status: PASS / FAIL
Violations: <count>

CRITICAL VIOLATIONS (MUST FIX):
- <pattern>: <file>:<line> - <details>

HIGH PRIORITY VIOLATIONS (SHOULD FIX):
- <pattern>: <file>:<line> - <details>

RECOMMENDATIONS (NICE TO HAVE):
- <suggestion>: <file> - <details>

STATISTICS:
- Pydantic v2 adoption: <X%>
- Custom validation methods: <N>
- Performance optimizations: <M/%>
```

### Violations Found
_List any Pydantic v1 patterns found:_
1. _______________
2. _______________
3. _______________

### Action Items
_Migration tasks for this project:_
1. _______________
2. _______________
3. _______________

### Status
- [ ] Audit Complete
- [ ] Fixes Applied
- [ ] Verified
- [ ] Documented
```

---

## Automated Audit Script

**File**: `~/flext/scripts/audit_pydantic_v2.py`

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
# Audit all projects
cd ~/flext
python scripts/audit_pydantic_v2.py

# Audit single project
python scripts/audit_pydantic_v2.py flext-core

# Generate report
python scripts/audit_pydantic_v2.py > audit_report.txt 2>&1
```

---

## Migration Coordination

### Phase 1: Foundation (Week 1)
**Projects**: flext-core, flext-cli, flext-ldif, flext-ldap, flext-api

**Strategy**:
1. Complete flext-core first (Parts 1-4)
2. Use flext-core patterns as reference
3. Apply same checklist to each project
4. Test dependencies between projects

### Phase 2: Domain Libraries (Week 2)
**Projects**: flext-auth, flext-web, flext-grpc, flext-meltano, flext-observability, flext-quality, flext-plugin

**Strategy**:
1. Audit all projects
2. Prioritize by complexity (simpler first)
3. Apply fixes in parallel (different developers)
4. Cross-verify with foundation libraries

### Phase 3: Singer Platform (Week 3)
**Projects**: All taps, targets, DBT, database projects (19 total)

**Strategy**:
1. Audit representative sample (1 tap, 1 target, 1 DBT)
2. Create template fixes
3. Apply to similar projects
4. Automate where possible

### Phase 4: Enterprise (Week 4)
**Projects**: client-a-oud-mig, client-b-meltano-native

**Strategy**:
1. Coordinate with stakeholders
2. Schedule maintenance window
3. Apply fixes
4. Comprehensive testing

---

## Success Criteria

After completing Part 5:
- ✅ **All 33 projects audited**
- ✅ **Zero Pydantic v1 patterns** across ecosystem
- ✅ **Consistent standards** (all use FlextTypes, ConfigDict, etc.)
- ✅ **Documentation updated** (all CLAUDE.md files)
- ✅ **Quality gates pass** in all projects

---

## Next Steps

After completing Part 5:
1. ✅ Generate audit report for all projects
2. ✅ Create issues for each project with violations
3. ➡️ Proceed to Part 6: [Quality Gates](./06-quality-gates.md)

---

**Time Estimate**: 1 week (distributed across team, can be parallelized)
