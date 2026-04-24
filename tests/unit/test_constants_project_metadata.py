"""Behavior contract for c.* project-metadata constants — SSOT for naming law."""

from __future__ import annotations

from tests import c


class TestsFlextCoreConstantsProjectMetadata:
    """Behavior contract for alias→suffix mappings, tier prefixes, and name overrides."""

    def test_alias_to_suffix_maps_every_runtime_alias_to_its_facade_suffix(
        self,
    ) -> None:
        assert c.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.ALIAS_TO_SUFFIX["m"] == "Models"
        assert c.ALIAS_TO_SUFFIX["p"] == "Protocols"
        assert c.ALIAS_TO_SUFFIX["t"] == "Types"
        assert c.ALIAS_TO_SUFFIX["u"] == "Utilities"
        assert c.ALIAS_TO_SUFFIX["r"] == "Result"

    def test_runtime_alias_names_mirror_alias_to_suffix_keys(self) -> None:
        assert set(c.RUNTIME_ALIAS_NAMES) == set(c.ALIAS_TO_SUFFIX)

    def test_tier_facade_prefix_defines_the_five_supported_tiers(self) -> None:
        assert c.TIER_FACADE_PREFIX["src"] == "Flext"
        assert c.TIER_FACADE_PREFIX["tests"] == "TestsFlext"
        assert c.TIER_FACADE_PREFIX["examples"] == "ExamplesFlext"
        assert c.TIER_FACADE_PREFIX["scripts"] == "ScriptsFlext"
        assert c.TIER_FACADE_PREFIX["docs"] == "DocsFlext"

    def test_scan_directories_mirror_tier_facade_prefix_keys(self) -> None:
        assert set(c.SCAN_DIRECTORIES) == set(c.TIER_FACADE_PREFIX)

    def test_universal_alias_parent_sources_anchor_system_aliases_at_flext_core(
        self,
    ) -> None:
        for alias in ("r", "e", "d", "x"):
            assert c.UNIVERSAL_ALIAS_PARENT_SOURCES[alias] == "flext_core"

    def test_special_name_overrides_disambiguate_root_and_core_projects(self) -> None:
        assert c.SPECIAL_NAME_OVERRIDES["flext"] == "FlextRoot"
        assert c.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"
