"""Fixture dictionary model helper namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from tests._models._mixins import (
    TestsFlextModelsFixturePayloadsMixin,
    TestsFlextModelsFixtureSuiteMixin,
)


class TestsFlextModelsFixtureDictsMixin(
    TestsFlextModelsFixturePayloadsMixin,
    TestsFlextModelsFixtureSuiteMixin,
):
    """Fixture dictionary model helpers."""


__all__: list[str] = ["TestsFlextModelsFixtureDictsMixin"]
