from __future__ import annotations

from collections.abc import Mapping, Sequence

from tests import t

type TestCaseMap = Mapping[str, t.NormalizedValue]

type InputPayloadMap = Mapping[str, t.NormalizedValue]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | Sequence[int] | Mapping[str, str]

type SetGetExpectedValue = t.Primitives
