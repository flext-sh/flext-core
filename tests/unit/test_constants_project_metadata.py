"""Behavior contract for c.* project-metadata constants."""

from __future__ import annotations

from tests import c


class TestsFlextConstantsProjectMetadata:
    """Behavior contract for fixed project-metadata constants."""

    def test_project_metadata_constants_are_flat_on_c(self) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"
        assert c.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.ALIAS_TO_SUFFIX["u"] == "Utilities"
        assert frozenset(c.ALIAS_TO_SUFFIX) == c.RUNTIME_ALIAS_NAMES
        assert frozenset({"c", "m", "p", "t", "u"}) == c.FACADE_ALIAS_NAMES
        assert (
            frozenset({
                "constants",
                "models",
                "protocols",
                "typings",
                "utilities",
            })
            == c.FACADE_MODULE_NAMES
        )
        assert c.TIER_FACADE_PREFIX["tests"] == "TestsFlext"
        assert c.SCAN_DIRECTORIES == ("src", "tests", "examples", "scripts", "docs")
        assert c.TIER_SUB_NAMESPACE["examples"] == "Examples"
        assert c.UNIVERSAL_ALIAS_PARENT_SOURCES["r"] == "flext_core"
        assert c.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"
