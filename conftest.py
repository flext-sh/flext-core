"""Pytest bootstrap for flext-core local package resolution.

Copyright (c) 2026 FLEXT Team. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

from flext_tests import install_local_packages

install_local_packages(Path(__file__).resolve().parent, ("examples", "tests"))
