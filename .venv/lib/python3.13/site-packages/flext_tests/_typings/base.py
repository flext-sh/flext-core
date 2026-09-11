"""Base tests typing primitives.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types as bt
from collections.abc import (
    ItemsView,
    KeysView,
    MutableMapping,
    Sequence,
    Set as AbstractSet,
    ValuesView,
)
from datetime import datetime, timezone, tzinfo
from enum import Enum
from pathlib import Path
from types import FrameType, GenericAlias, ModuleType
from typing import NotRequired, TypedDict

from flext_cli import t
from flext_core import m, p
from flext_infra import t as it


class FlextTestsBaseTypesMixin:
    """Base generic primitives extending Flext core aliases."""

    type TestobjectAtom = (
        str
        | int
        | float
        | bool
        | bytes
        | datetime
        | tzinfo
        | Path
        | m.BaseModel
        | type
        | frozenset[str]
    )
    type TestobjectCollection = (
        t.SequenceOf[FlextTestsBaseTypesMixin.TestobjectSerializable]
        | t.MappingKV[str, FlextTestsBaseTypesMixin.TestobjectSerializable]
    )
    type TestobjectSerializable = TestobjectAtom | TestobjectCollection | None
    type TestobjectHashable = (
        str | int | float | bool | bytes | datetime | tzinfo | Path | type | None
    )
    type NormalizationInput = (
        TestobjectAtom
        | m.RootModel[FlextTestsBaseTypesMixin.NormalizationInput]
        | t.SequenceOf[FlextTestsBaseTypesMixin.NormalizationInput]
        | t.MappingKV[str, FlextTestsBaseTypesMixin.NormalizationInput]
        | set[FlextTestsBaseTypesMixin.TestobjectHashable]
        | None
    )

    class HandlerCaseSpec(TypedDict):
        handler_id: str
        handler_type: str
        description: str
        expected_result: NotRequired[str]
        should_fail: NotRequired[bool]
        error_message: NotRequired[str]

    type TestResultValue = (
        FlextTestsBaseTypesMixin.TestobjectSerializable
        | t.RegisterableService
        | t.TypeHintSpecifier
        | BaseException
        | Exception
        | Enum
        | t.SecretStr
        | p.Logger
        | p.Container
        | p.Dispatcher
        | p.Settings
        | p.Context
        | p.Registry
        | p.AttributeProbe
        | p.Result[FlextTestsBaseTypesMixin.TestobjectSerializable]
        | it.Infra.RegexMatch
        | bt.UnionType
        | FrameType
        | ModuleType
        | GenericAlias
        | set[FlextTestsBaseTypesMixin.TestobjectHashable]
        | AbstractSet[FlextTestsBaseTypesMixin.TestobjectHashable]
        | ValuesView[FlextTestsBaseTypesMixin.TestobjectSerializable]
        | KeysView[str]
        | ItemsView[str, FlextTestsBaseTypesMixin.TestobjectSerializable]
        | MutableMapping[str, FlextTestsBaseTypesMixin.TestobjectSerializable]
        | tzinfo
        | timezone
    )
    "Type for FLEXT test result payloads."

    type Testobject = (
        FlextTestsBaseTypesMixin.TestResultValue
        | p.Result[FlextTestsBaseTypesMixin.TestResultValue]
    )

    TESTOBJECT_SERIALIZABLE_ADAPTER: m.TypeAdapter[TestobjectSerializable] = (
        m.TypeAdapter(
            TestobjectSerializable, config=m.ConfigDict(arbitrary_types_allowed=True)
        )
    )

    TESTOBJECT_SEQUENCE_ADAPTER: m.TypeAdapter[
        t.SequenceOf[FlextTestsBaseTypesMixin.TestobjectSerializable]
    ] = m.TypeAdapter(
        t.SequenceOf[TestobjectSerializable],
        config=m.ConfigDict(arbitrary_types_allowed=True),
    )
    TESTOBJECT_MAPPING_ADAPTER: m.TypeAdapter[
        t.MappingKV[str, FlextTestsBaseTypesMixin.TestobjectSerializable]
    ] = m.TypeAdapter(
        t.MappingKV[str, TestobjectSerializable],
        config=m.ConfigDict(arbitrary_types_allowed=True),
    )
    STR_MAPPING_SEQUENCE_ADAPTER: m.TypeAdapter[t.SequenceOf[t.StrMapping]] = (
        m.TypeAdapter(t.SequenceOf[t.StrMapping])
    )
    TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER: m.TypeAdapter[
        t.MappingKV[str, FlextTestsBaseTypesMixin.TestobjectSerializable]
    ] = m.TypeAdapter(
        t.MappingKV[str, TestobjectSerializable],
        config=m.ConfigDict(arbitrary_types_allowed=True),
    )
    TESTOBJECT_SERIALIZABLE_SEQUENCE_ADAPTER: m.TypeAdapter[
        t.SequenceOf[FlextTestsBaseTypesMixin.TestobjectSerializable]
    ] = m.TypeAdapter(
        t.SequenceOf[TestobjectSerializable],
        config=m.ConfigDict(arbitrary_types_allowed=True),
    )
    DICT_ADAPTER: m.TypeAdapter[m.Dict] = m.TypeAdapter(m.Dict)
    SCALAR_MAPPING_ADAPTER: m.TypeAdapter[t.ScalarMapping] = m.TypeAdapter(
        t.ScalarMapping
    )
    CONTAINER_MAPPING_SEQUENCE_ADAPTER: m.TypeAdapter[t.SequenceOf[t.JsonMapping]] = (
        m.TypeAdapter(t.SequenceOf[t.JsonMapping])
    )
    STR_MAPPING_MAPPING_ADAPTER: m.TypeAdapter[t.MappingKV[str, t.StrMapping]] = (
        m.TypeAdapter(t.MappingKV[str, t.StrMapping])
    )
    INTEGER_SEQUENCE_ADAPTER: m.TypeAdapter[Sequence[int]] = m.TypeAdapter(
        Sequence[int]
    )
    STR_SEQUENCE_MAPPING_ADAPTER: m.TypeAdapter[t.MappingKV[str, t.StrSequence]] = (
        m.TypeAdapter(t.MappingKV[str, t.StrSequence])
    )
