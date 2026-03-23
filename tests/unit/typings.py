from __future__ import annotations

from collections.abc import Mapping, MutableMapping, MutableSequence

from tests import t

type TestCaseMap = Mapping[str, t.NormalizedValue]

type InputPayloadMap = Mapping[str, t.NormalizedValue]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | MutableSequence[int] | MutableMapping[str, str]

type SetGetExpectedValue = t.Primitives
