from __future__ import annotations

from collections.abc import Mapping, MutableSequence

from tests import t

type TestCaseMap = Mapping[str, t.Tests.Testobject]

type InputPayloadMap = Mapping[str, t.Tests.Testobject]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | MutableSequence[int] | t.MutableStrMapping

type SetGetExpectedValue = t.Primitives
