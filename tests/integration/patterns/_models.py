"""Auto-generated centralized models."""

from __future__ import annotations

from pydantic import RootModel


class FixtureCaseDict(RootModel[dict[str, str]]):
    pass


class FixtureDataDict(RootModel[dict[str, t.ContainerValue]]):
    pass


class FixtureFixturesDict(RootModel[dict[str, t.ContainerValue]]):
    pass


class FixtureSuiteDict(RootModel[dict[str, t.ContainerValue]]):
    pass
