"""Auto-generated centralized models."""

from __future__ import annotations

from collections.abc import ItemsView, Mapping, ValuesView

from pydantic import RootModel


class _StringDictModel(RootModel[dict[str, str]]):
    def __getitem__(self, key: str) -> str:
        return self.root[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.root[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def get(self, key: str, default: str = "") -> str:
        return self.root.get(key, default)

    def update(self, payload: Mapping[str, str]) -> None:
        self.root.update(payload)

    def setdefault(self, key: str, default: str) -> str:
        return self.root.setdefault(key, default)

    def values(self) -> ValuesView[str]:
        return self.root.values()

    def items(self) -> ItemsView[str, str]:
        return self.root.items()


class _ContainerDictModel(RootModel[dict[str, object]]):
    def __getitem__(self, key: str) -> object:
        return self.root[key]

    def __setitem__(self, key: str, value: object) -> None:
        self.root[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def get(self, key: str, default: object | None = None) -> object | None:
        return self.root.get(key, default)

    def update(self, payload: Mapping[str, object]) -> None:
        self.root.update(payload)

    def setdefault(self, key: str, default: object) -> object:
        return self.root.setdefault(key, default)

    def values(self) -> ValuesView[object]:
        return self.root.values()

    def items(self) -> ItemsView[str, object]:
        return self.root.items()


class FixtureCaseDict(_StringDictModel):
    pass


class FixtureDataDict(_ContainerDictModel):
    pass


class FixtureFixturesDict(_ContainerDictModel):
    pass


class FixtureSuiteDict(_ContainerDictModel):
    pass
