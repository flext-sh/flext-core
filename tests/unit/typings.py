from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence

from tests import t

type TestCaseMap = Mapping[str, t.Tests.Testobject]

type InputPayloadMap = Mapping[str, t.Tests.Testobject]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | MutableSequence[int] | MutableMapping[str, str]

type SetGetExpectedValue = t.Primitives
