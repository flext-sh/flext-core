"""Tests for FlextTypingProjectMetadata PEP 695 aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core._typings.project_metadata import (
    FlextTypingProjectMetadata as tp,
)


@pytest.mark.parametrize(
    "alias_name",
    [
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
    ],
)
def test_alias_is_defined(alias_name: str) -> None:
    assert hasattr(tp, alias_name), alias_name


def test_alias_count_stable() -> None:
    declared = {
        name
        for name in vars(tp)
        if not name.startswith("_") and not name.endswith("__")
    }
    assert declared == {
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
    }
