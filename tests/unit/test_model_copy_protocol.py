"""Public model-copy capability without assumptions about serialized payloads.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Self, get_type_hints, is_protocol

from flext_tests import tm

from tests import m, p


class TestsFlextCoreModelCopyProtocol:
    """The minimal copy port preserves native models and the existing Model API."""

    @staticmethod
    def test_native_models_preserve_copy_and_public_model_contract() -> None:
        """Share the native copy signature without inventing a RootModel dump type."""
        tm.ok(is_protocol(p.ModelCopy))
        tm.ok(is_protocol(p.BaseModel))
        tm.ok(p.ModelCopy in p.BaseModel.__mro__)
        tm.ok(p.BaseModel.model_copy is p.ModelCopy.model_copy)
        tm.that(
            tuple(signature(p.ModelCopy.model_copy).parameters),
            eq=tuple(signature(m.BaseModel.model_copy).parameters),
        )
        tm.ok(get_type_hints(p.ModelCopy.model_copy)["return"] is Self)
        for value in (m.Metadata(), m.ConfigMap.model_validate({"value": 1})):
            port: p.ModelCopy = value
            tm.ok(isinstance(port, p.ModelCopy))
            copied = port.model_copy(deep=True)
            tm.ok(copied is not value)
            tm.ok(type(copied) is type(value))
            tm.that(copied, eq=value)
        tm.ok(not isinstance(Path.cwd(), p.ModelCopy))
