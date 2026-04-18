"""Tests for FlextTypingProjectMetadata PEP 695 aliases (flat on ``t.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import (
    FlextTypingProjectMetadata as tp,
)

_ALIAS_NAMES = (
    "ProjectAliasName",
    "ProjectTierName",
    "ProjectName",
    "ProjectPackageName",
    "ProjectClassStem",
    "ProjectLibraryName",
    "ProjectAliasToSuffixMap",
    "ProjectTierFacadePrefixMap",
    "ProjectAliasParentSourceMap",
    "ProjectSpecialNameOverrideMap",
    "ProjectManagedKeyName",
    "ProjectManagedKeyTuple",
)


@pytest.mark.parametrize("alias_name", _ALIAS_NAMES)
def test_alias_is_defined(alias_name: str) -> None:
    assert hasattr(tp, alias_name), alias_name


def test_alias_count_stable() -> None:
    declared = {
        name
        for name in vars(tp)
        if not name.startswith("_") and not name.endswith("__")
    }
    assert declared == set(_ALIAS_NAMES)
