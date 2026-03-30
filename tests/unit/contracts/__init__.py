# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Shared unit-test contracts for behavior reuse across suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.unit.contracts import text_contract as text_contract
    from tests.unit.contracts.text_contract import (
        TextUtilityContract as TextUtilityContract,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "TextUtilityContract": [
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ],
    "text_contract": ["tests.unit.contracts.text_contract", ""],
}

_EXPORTS: Sequence[str] = [
    "TextUtilityContract",
    "text_contract",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
