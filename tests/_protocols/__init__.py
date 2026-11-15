# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from tests._protocols.mixins import TestsFlextProtocolsMixins


__all__: tuple[str, ...] = ("TestsFlextProtocolsMixins",)

install_lazy_exports(
    __name__,
    globals(),
    {"TestsFlextProtocolsMixins": ".mixins"},
    public_exports=__all__,
    deferred_members={
        "TestsFlextProtocolsMixins": (),
    },
)

defer_lazy_bases(
    __name__,
    {
        "TestsFlextProtocolsMixins": (
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsContainer"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsCoreState"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsCoreErrors"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsCorePublic"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsDomain"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsGuardsMapper"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsSchemaOwnerHeir"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsSchemaOwner"),
            (
                "tests._protocols._mixins",
                "TestsFlextProtocolsMixinsServiceCaseReliability",
            ),
            (
                "tests._protocols._mixins",
                "TestsFlextProtocolsMixinsServiceCaseValidation",
            ),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsServiceCaseCore"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsTestDataValues"),
            ("tests._protocols._mixins", "TestsFlextProtocolsMixinsTestDataIdentity"),
        ),
    },
)
