"""Tests for FlextConstantsProjectMetadata (flat on ``c.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextConstantsProjectMetadata as k,
)


class TestAliasToSuffix:
    def test_c_alias_maps_to_constants(self) -> None:
        assert k.ALIAS_TO_SUFFIX["c"] == "Constants"

    def test_m_alias_maps_to_models(self) -> None:
        assert k.ALIAS_TO_SUFFIX["m"] == "Models"

    def test_p_alias_maps_to_protocols(self) -> None:
        assert k.ALIAS_TO_SUFFIX["p"] == "Protocols"

    def test_t_alias_maps_to_types(self) -> None:
        assert k.ALIAS_TO_SUFFIX["t"] == "Types"

    def test_u_alias_maps_to_utilities(self) -> None:
        assert k.ALIAS_TO_SUFFIX["u"] == "Utilities"

    def test_r_alias_maps_to_result(self) -> None:
        assert k.ALIAS_TO_SUFFIX["r"] == "Result"

    def test_runtime_alias_names_match_alias_to_suffix_keys(self) -> None:
        assert set(k.RUNTIME_ALIAS_NAMES) == set(k.ALIAS_TO_SUFFIX)


class TestTierFacadePrefix:
    def test_src_prefix(self) -> None:
        assert k.TIER_FACADE_PREFIX["src"] == "Flext"

    def test_tests_prefix(self) -> None:
        assert k.TIER_FACADE_PREFIX["tests"] == "TestsFlext"

    def test_examples_prefix(self) -> None:
        assert k.TIER_FACADE_PREFIX["examples"] == "ExamplesFlext"

    def test_scripts_prefix(self) -> None:
        assert k.TIER_FACADE_PREFIX["scripts"] == "ScriptsFlext"

    def test_docs_prefix(self) -> None:
        assert k.TIER_FACADE_PREFIX["docs"] == "DocsFlext"

    def test_scan_directories_cover_all_tiers(self) -> None:
        assert set(k.SCAN_DIRECTORIES) == set(k.TIER_FACADE_PREFIX)


class TestUniversalAliasParentSources:
    def test_r_from_flext_core(self) -> None:
        assert k.UNIVERSAL_ALIAS_PARENT_SOURCES["r"] == "flext_core"

    def test_e_from_flext_core(self) -> None:
        assert k.UNIVERSAL_ALIAS_PARENT_SOURCES["e"] == "flext_core"

    def test_d_from_flext_core(self) -> None:
        assert k.UNIVERSAL_ALIAS_PARENT_SOURCES["d"] == "flext_core"

    def test_x_from_flext_core(self) -> None:
        assert k.UNIVERSAL_ALIAS_PARENT_SOURCES["x"] == "flext_core"


class TestSpecialNameOverrides:
    def test_flext_overrides_to_flextroot(self) -> None:
        assert k.SPECIAL_NAME_OVERRIDES["flext"] == "FlextRoot"

    def test_flext_core_overrides_to_flext(self) -> None:
        assert k.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"


class TestManagedPyprojectKeys:
    def test_project_key(self) -> None:
        assert "tool.flext.project" in k.MANAGED_PYPROJECT_KEYS

    def test_namespace_key(self) -> None:
        assert "tool.flext.namespace" in k.MANAGED_PYPROJECT_KEYS

    def test_docs_key(self) -> None:
        assert "tool.flext.docs" in k.MANAGED_PYPROJECT_KEYS

    def test_aliases_key(self) -> None:
        assert "tool.flext.aliases" in k.MANAGED_PYPROJECT_KEYS
