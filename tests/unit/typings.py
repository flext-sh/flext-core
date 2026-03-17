from __future__ import annotations

from collections.abc import Mapping

from tests import t

type TestCaseMap = Mapping[str, t.Tests.object]

type InputPayloadMap = dict[str, t.Tests.object]

type SampleValue = t.Primitives | None

type SetGetInputValue = t.Primitives | list[int] | dict[str, str]

type SetGetExpectedValue = t.Primitives
