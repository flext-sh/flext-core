"""Lossless binary persistence through the public contract-model preset.

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

import pytest

from tests import m, tm


class TestsFlextCoreContractBytes:
    """JSON persistence retains the exact bytes supplied by a caller."""

    @staticmethod
    @pytest.mark.parametrize(
        "payload",
        [b"", b"source content\n", bytes(range(256))],
        ids=["empty", "text", "binary"],
    )
    def test_binary_contract_round_trips_without_reencoding(payload: bytes) -> None:
        """A hydrated public payload remains byte-for-byte consumable."""
        original = m.Tests.BinaryContract(payload=payload)
        restored = m.Tests.BinaryContract.model_validate_json(
            original.model_dump_json()
        )
        tm.that(restored.payload, eq=payload)
