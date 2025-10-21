# Part 6: Quality Gate Enforcement

**Status**: AUTOMATION & PREVENTION
**Priority**: 🔴 CRITICAL (After Parts 2-4 completion)
**Estimated Time**: 2-3 hours
**Impact**: Prevents regression across all 33 projects

**Related**:
- Audit script: [audit_pydantic_v2.py](./audit_pydantic_v2.py)
- Workspace audit: [05-workspace-audit.md](./05-workspace-audit.md)
- Execution timeline: [08-execution-timeline.md](./08-execution-timeline.md)
- FLEXT CLAUDE.md: `/home/marlonsc/flext/CLAUDE.md` (quality standards)
- flext-core CLAUDE.md: `/home/marlonsc/flext/flext-core/CLAUDE.md` (quality gates section)

**Pydantic v2 Reference**:
- All concepts: `/home/marlonsc/flext/docs/references/pydantic2/concepts/`

---

## Overview

Implement automated quality gates to prevent Pydantic v1 patterns and duplication from being reintroduced:
1. Enhanced Makefile validation target
2. Pre-commit hooks for developers
3. CI/CD integration for main branch
4. Continuous monitoring dashboard (optional)

**Success Criteria**:
- ✅ All automated checks pass locally before commit
- ✅ CI/CD pipeline prevents merging broken code
- ✅ Audit reports generated automatically
- ✅ Zero Pydantic v1 patterns in codebase

---

## Section 6.1: Enhanced Makefile

**File**: `flext-core/Makefile`

Add Pydantic v2 audit target:

```makefile
.PHONY: audit-pydantic-v2
audit-pydantic-v2:  ## Audit Pydantic v2 compliance
	@echo "🔍 Auditing Pydantic v2 compliance..."
	@python $(PWD)/../scripts/audit_pydantic_v2.py $(shell basename $(PWD))

.PHONY: validate
validate:  ## Run complete validation pipeline (ENHANCED)
	@echo "Running complete validation pipeline..."
	make lint
	make type-check
	make security
	make audit-pydantic-v2  # NEW: Pydantic v2 compliance
	make test
	@echo "✅ All validation checks passed!"
```

**Usage**:
```bash
cd flext-core
make audit-pydantic-v2  # Check Pydantic v2 compliance
make validate           # Full pipeline including Pydantic check
```

---

## Section 6.2: Pre-Commit Hooks

### Install Pre-Commit Framework

```bash
cd flext-core
pip install pre-commit
pre-commit install
```

### Configuration

**File**: `.pre-commit-config.yaml`

```yaml
repos:
  # Existing hooks...
  
  - repo: local
    hooks:
      - id: check-pydantic-v2
        name: Check Pydantic v2 Patterns
        entry: python scripts/check_pydantic_v2_precommit.py
        language: system
        types: [python]
        pass_filenames: true
        stages: [commit]
        verbose: true
```

### Pre-Commit Script

**File**: `scripts/check_pydantic_v2_precommit.py`

```python
#!/usr/bin/env python3
"""Pre-commit hook: Block Pydantic v1 patterns."""

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    (r'class\s+Config:', 'Use model_config = ConfigDict()'),
    (r'\.dict\(\)', 'Use .model_dump()'),
    (r'\.json\(\)', 'Use .model_dump_json()'),
    (r'parse_obj\(', 'Use .model_validate()'),
    (r'@validator\(', 'Use @field_validator()'),
    (r'@root_validator', 'Use @model_validator()'),
]

def check_file(filepath: str) -> bool:
    """Check file for forbidden patterns."""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Could not read {filepath}: {e}")
        return True
    
    passed = True
    for line_num, line in enumerate(content.split('\n'), 1):
        for pattern, message in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                print(f"❌ {filepath}:{line_num}: {message}")
                print(f"   {line.strip()}")
                passed = False
    
    return passed

def main():
    if len(sys.argv) < 2:
        return 0
    
    all_passed = all(check_file(f) for f in sys.argv[1:])
    
    if not all_passed:
        print("\n❌ COMMIT BLOCKED: Fix Pydantic v1 patterns")
        print("See: docs/pydantic-v2-modernization/")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Make executable**:
```bash
chmod +x scripts/check_pydantic_v2_precommit.py
```

### Test Pre-Commit Hook

```bash
# Create test file with forbidden pattern
echo "model.dict()" > test_bad.py
git add test_bad.py
git commit -m "test"  # Should BLOCK

# Clean up
rm test_bad.py
git reset
```

---

## Section 6.3: CI/CD Integration

### GitHub Actions Workflow

**File**: `.github/workflows/pydantic-v2-check.yml`

```yaml
name: Pydantic v2 Compliance

on:
  pull_request:
    paths:
      - '**.py'
  push:
    branches:
      - main
      - develop

jobs:
  pydantic-v2-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      
      - name: Install dependencies
        run: |
          pip install pydantic
      
      - name: Run Pydantic v2 Audit
        run: |
          python scripts/audit_pydantic_v2.py
      
      - name: Comment PR if violations found
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Pydantic v1 patterns detected. See audit log above.'
            })
```

### GitLab CI Pipeline

**File**: `.gitlab-ci.yml`

```yaml
pydantic-v2-check:
  stage: test
  image: python:3.13
  script:
    - pip install pydantic
    - python scripts/audit_pydantic_v2.py
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      changes:
        - "**/*.py"
  allow_failure: false  # Block merge if violations found
```

---

## Section 6.4: IDE Integration

### VS Code Settings

**File**: `.vscode/settings.json`

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": [
    "--strict",
    "--show-error-codes"
  ],
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.diagnosticSeverityOverrides": {
    "reportUnknownMemberType": "none",
    "reportGeneralTypeIssues": "error"
  },
  "files.associations": {
    "*.pyi": "python"
  },
  "search.exclude": {
    "**/.venv": true,
    "**/node_modules": true,
    "**/__pycache__": true
  }
}
```

### PyCharm Configuration

**File**: `.idea/inspectionProfiles/Project_Default.xml`

```xml
<component name="InspectionProjectProfileManager">
  <profile version="1.0">
    <option name="myName" value="Project Default" />
    <inspection_tool class="PyPep8Naming" enabled="true" level="WARNING" enabled_by_default="true">
      <option name="ignoredErrors">
        <list>
          <option value="N802" />
        </list>
      </option>
    </inspection_tool>
    <inspection_tool class="PyUnresolvedReferences" enabled="true" level="ERROR" enabled_by_default="true" />
  </profile>
</component>
```

---

## Section 6.5: Monitoring Dashboard

### Quality Metrics Script

**File**: `scripts/quality_dashboard.py`

```python
#!/usr/bin/env python3
"""Generate quality metrics dashboard."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

def get_metrics(project_path: Path) -> dict:
    """Get quality metrics for project."""
    metrics = {
        "project": project_path.name,
        "timestamp": datetime.now().isoformat(),
        "pydantic_v2_compliance": None,
        "test_pass_rate": None,
        "type_check_errors": None,
        "lint_violations": None,
    }
    
    # Pydantic v2 compliance
    result = subprocess.run(
        ["python", "scripts/audit_pydantic_v2.py", project_path.name],
        capture_output=True,
        text=True
    )
    metrics["pydantic_v2_compliance"] = result.returncode == 0
    
    # Test pass rate
    result = subprocess.run(
        ["make", "-C", str(project_path), "test"],
        capture_output=True,
        text=True
    )
    # Parse test output for pass rate
    
    # Type check errors
    result = subprocess.run(
        ["make", "-C", str(project_path), "type-check"],
        capture_output=True,
        text=True
    )
    metrics["type_check_errors"] = result.returncode != 0
    
    return metrics

def main():
    workspace = Path.cwd()
    projects = sorted([p.parent for p in workspace.glob("flext-*/pyproject.toml")])
    
    all_metrics = []
    for project in projects:
        print(f"Collecting metrics for {project.name}...")
        metrics = get_metrics(project)
        all_metrics.append(metrics)
    
    # Save to JSON
    with open("quality_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    compliant = sum(1 for m in all_metrics if m["pydantic_v2_compliance"])
    print(f"\n✅ Pydantic v2 Compliant: {compliant}/{len(all_metrics)}")

if __name__ == "__main__":
    main()
```

---

## Implementation Checklist

### Makefile Enhancement
- [ ] Add `audit-pydantic-v2` target
- [ ] Update `validate` target to include audit
- [ ] Test with `make validate`

### Pre-Commit Hooks
- [ ] Install pre-commit framework
- [ ] Add configuration to `.pre-commit-config.yaml`
- [ ] Create `check_pydantic_v2_precommit.py` script
- [ ] Make script executable
- [ ] Test with sample bad code
- [ ] Install hooks: `pre-commit install`

### CI/CD Integration
- [ ] Create GitHub Actions workflow (if using GitHub)
- [ ] Create GitLab CI config (if using GitLab)
- [ ] Test pipeline with PR/MR

### IDE Configuration
- [ ] Add VS Code settings
- [ ] Add PyCharm configuration
- [ ] Distribute to team

### Monitoring
- [ ] Create quality dashboard script
- [ ] Schedule daily/weekly runs
- [ ] Set up alerts for violations

---

## Success Criteria

After completing Part 6:
- ✅ **Automated enforcement** in place
- ✅ **Pre-commit hooks** prevent bad commits
- ✅ **CI/CD blocks** PRs with violations
- ✅ **IDE integration** provides immediate feedback
- ✅ **Monitoring dashboard** tracks compliance

---

## Next Steps

After completing Part 6:
1. ✅ Test all quality gates
2. ✅ Train team on new processes
3. ➡️ Proceed to Part 7: [Documentation](./07-documentation.md)

---

**Time Estimate**: 2-3 hours setup, ongoing monitoring
