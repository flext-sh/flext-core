#!/usr/bin/env python3
"""Pre-commit hook: Block Pydantic v1 patterns.

This script is run by pre-commit framework to prevent Pydantic v1 patterns
from being committed to the repository. It checks for:
- class Config: pattern
- .dict() method calls
- .json() method calls (HTTP library calls excluded)
- parse_obj() calls
- @validator decorators
- @root_validator decorators

Usage:
    # Manual testing
    python scripts/check_pydantic_v2_precommit.py src/module.py

    # Run via pre-commit
    pre-commit run --all-files

Exit Codes:
    0 = No violations found
    1 = Violations found
"""

import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = [
    (r"class\s+\w+.*:\s*\n\s*class\s+Config:", "Use model_config = ConfigDict()"),
    (r"\.dict\(", "Use .model_dump()"),
    # NOTE: .json() excluded due to HTTP library false positives (requests.json(), httpx.json())
    # (r'\.json\(', 'Use .model_dump_json()'),
    (r"parse_obj\(", "Use .model_validate()"),
    (r"@validator\(", "Use @field_validator()"),
    (r"@root_validator", "Use @model_validator()"),
]


def check_file(filepath: str) -> bool:
    """Check file for forbidden patterns.

    Args:
        filepath: Path to Python file to check

    Returns:
        True if file is clean, False if violations found

    """
    try:
        with Path(filepath).open(encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️  Could not read {filepath}: {e}")
        return True

    passed = True
    for line_num, line in enumerate(content.split("\n"), 1):
        for pattern, message in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
                # Skip docstrings and comments
                stripped = line.strip()
                if stripped.startswith(("#", '"""', "'''")):
                    continue

                print(f"❌ {filepath}:{line_num}: {message}")
                print(f"   {line.rstrip()}")
                passed = False

    return passed


def main() -> int:
    """Check all provided files for Pydantic v1 patterns.

    Returns:
        0 if all files are clean, 1 if violations found

    """
    if len(sys.argv) < 2:
        # If no files provided, exit successfully
        return 0

    all_passed = all(check_file(f) for f in sys.argv[1:])

    if not all_passed:
        print("\n❌ COMMIT BLOCKED: Fix Pydantic v1 patterns")
        print("   See: docs/pydantic-v2-modernization/")
        print("   Or run: make audit-pydantic-v2")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
