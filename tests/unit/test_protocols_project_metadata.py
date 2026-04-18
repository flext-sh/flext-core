"""Tests for FlextProtocolsProjectMetadata protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core._protocols.project_metadata import (
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


class TestMetadataReaderProtocol:
    def test_runtime_check_accepts_reader(self) -> None:
        assert isinstance(_ConcreteReader(), pm.MetadataReader)

    def test_runtime_check_rejects_unrelated(self) -> None:
        class _Unrelated:
            pass

        assert not isinstance(_Unrelated(), pm.MetadataReader)


class TestClassStemDeriverProtocol:
    def test_runtime_check_accepts_deriver(self) -> None:
        assert isinstance(_ConcreteStemmer(), pm.ClassStemDeriver)

    def test_runtime_check_rejects_unrelated(self) -> None:
        class _Unrelated:
            pass

        assert not isinstance(_Unrelated(), pm.ClassStemDeriver)


class TestTierFacadeNamerProtocol:
    def test_runtime_check_accepts_namer(self) -> None:
        assert isinstance(_ConcreteTierNamer(), pm.TierFacadeNamer)

    def test_runtime_check_rejects_unrelated(self) -> None:
        class _Unrelated:
            pass

        assert not isinstance(_Unrelated(), pm.TierFacadeNamer)
