"""Tests for FlextProtocolsProjectMetadata protocols (flat on ``p.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core import (
    FlextProtocolsProjectMetadata as pm,
)


class _ConcreteReader:
    def read(self, root: Path) -> object:
        return {"root": root}


class _ConcreteStemmer:
    def derive(self, project_name: str) -> str:
        return project_name.title().replace("-", "")


class _ConcreteTierNamer:
    def name_for(self, project_name: str, tier: str) -> str:
        return f"{tier}:{project_name}"


class TestsFlextCoreProtocolsProjectMetadata:
    def test_accepts_reader(self) -> None:
        assert isinstance(_ConcreteReader(), pm.ProjectMetadataReader)

    def test_rejects_unrelated(self) -> None:
        class _Unrelated:
            pass

        assert not isinstance(_Unrelated(), pm.ProjectMetadataReader)

    def test_accepts_deriver(self) -> None:
        assert isinstance(_ConcreteStemmer(), pm.ProjectClassStemDeriver)

    def test_accepts_namer(self) -> None:
        assert isinstance(_ConcreteTierNamer(), pm.ProjectTierFacadeNamer)
