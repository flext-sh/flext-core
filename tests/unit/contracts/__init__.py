# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Shared unit-test contracts for behavior reuse across suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.unit.contracts.text_contract import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "TextUtilityContract": "tests.unit.contracts.text_contract",
    "text_contract": "tests.unit.contracts.text_contract",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
