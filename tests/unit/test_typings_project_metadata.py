"""Tests for FlextTypingProjectMetadata PEP 695 aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core._typings.project_metadata import (
    FlextTypingProjectMetadata as tp,
)

_ALIAS_NAMES = (
    "AliasName",
    "TierName",
    "ProjectName",
    "PackageName",
    "ClassStem",
    "LibraryName",
    "AliasToSuffixMap",
    "TierFacadePrefixMap",
    "AliasParentSourceMap",
    "SpecialNameOverrideMap",
    "ManagedKeyName",
    "ManagedKeyTuple",
)


@pytest.mark.parametrize("alias_name", _ALIAS_NAMES)
def test_alias_is_defined(alias_name: str) -> None:
    assert hasattr(tp.Project, alias_name), alias_name


def test_alias_count_stable() -> None:
    declared = {
        name
        for name in vars(tp.Project)
        if not name.startswith("_") and not name.endswith("__")
    }
    assert declared == set(_ALIAS_NAMES)
