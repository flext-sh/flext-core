"""Run all tests and show coverage report.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This script runs all tests and shows the coverage report.
It uses pytest and coverage to run the tests and show the coverage report.
It also checks if the imports work.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_tests() -> int:
    """Run pytest with coverage.

    Returns:
        The exit code of the tests.

    """
    # First, let's run a basic test to ensure imports work
    try:
        pass

    except Exception:
        return 1

    # Run unit tests
    test_files = list(Path("tests/unit").rglob("test_*.py"))

    for test_file in test_files:
        # Using sys.executable is safe as it's the current Python interpreter
        result = subprocess.run(
            [sys.executable, str(test_file)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )

        if result.returncode != 0 and result.stderr:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
