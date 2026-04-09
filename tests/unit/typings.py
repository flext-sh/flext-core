from __future__ import annotations

from collections.abc import Mapping, MutableSequence

from tests import t

type TestCaseMap = Mapping[str, t.Core.Tests.TestobjectSerializable]

type InputPayloadMap = Mapping[str, t.Core.Tests.TestobjectSerializable]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | MutableSequence[int] | t.MutableStrMapping

type SetGetExpectedValue = t.Primitives
