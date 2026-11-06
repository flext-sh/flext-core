# AUTO-GENERATED FILE — Regenerate with: make gen
"""Pytest bootstrap for flext_core local package resolution.

Rendered from flext-infra ``templates/project/base/conftest.py.j2``: change the
template or the shared ``install_local_packages`` owner, then run ``make gen``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_tests import install_local_packages


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Register the local packages before collection imports any test module."""
    del config
    install_local_packages(Path(__file__).resolve().parent)
