"""Tests for FlextModelsProjectMetadata — Tier 3 Pydantic models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flext_core._models.project_metadata import (
    FlextModelsProjectMetadata as pm,
)


class TestProjectModel:
    """m.Project.Project: name/version/license + derived facade names."""

    def test_minimal_project(self) -> None:
        meta = pm.Project(
            name="flext-ldif",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-ldif"),
        )
        assert meta.package_name == "flext_ldif"
        assert meta.class_stem == "FlextLdif"
        assert meta.src_facade_name == "FlextLdif"
        assert meta.tests_facade_name == "TestsFlextLdif"

    def test_flext_core_override(self) -> None:
        meta = pm.Project(
            name="flext-core",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-core"),
        )
        assert meta.class_stem == "Flext"
        assert meta.src_facade_name == "Flext"
        assert meta.tests_facade_name == "TestsFlext"

    def test_flext_root_override(self) -> None:
        meta = pm.Project(
            name="flext",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext"),
        )
        assert meta.class_stem == "FlextRoot"

    def test_tier_facade_name_unknown_tier_raises(self) -> None:
        meta = pm.Project(
            name="flext-ldif",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-ldif"),
        )
        with pytest.raises(ValueError, match="unknown tier"):
            meta.tier_facade_name("nonsense")

    def test_frozen_rejects_mutation(self) -> None:
        meta = pm.Project(
            name="flext-ldif",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-ldif"),
        )
        with pytest.raises(ValidationError):
            meta.name = "other"  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            pm.Project(
                name="",
                version="0.12.0-dev",
                license="MIT",
                root=Path("/tmp/x"),
            )


class TestNamespaceModel:
    """m.Project.Namespace: alias parent source overrides."""

    def test_defaults_include_universal_sources(self) -> None:
        cfg = pm.Namespace(project_name="flext-ldif")
        assert cfg.alias_parent_sources["r"] == "flext_core"
        assert cfg.alias_parent_sources["e"] == "flext_core"
        assert cfg.enabled is True

    def test_custom_override_for_c(self) -> None:
        cfg = pm.Namespace(
            project_name="flext-ldif",
            alias_parent_sources={"c": "flext_cli", "m": "flext_cli"},
        )
        assert cfg.alias_parent_sources["c"] == "flext_cli"
        assert cfg.alias_parent_sources["m"] == "flext_cli"
        assert cfg.alias_parent_sources["r"] == "flext_core"

    def test_unknown_alias_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown alias"):
            pm.Namespace(
                project_name="flext-ldif",
                alias_parent_sources={"zz": "flext_core"},
            )

    def test_universal_override_rejected(self) -> None:
        with pytest.raises(ValidationError, match="universal"):
            pm.Namespace(
                project_name="flext-ldif",
                alias_parent_sources={"r": "flext_ldif"},
            )


class TestToolFlextModel:
    """m.Project.ToolFlext + sub-tables."""

    def test_defaults_round_trip(self) -> None:
        cfg = pm.ToolFlext()
        dumped = cfg.model_dump()
        reloaded = pm.ToolFlext.model_validate(dumped)
        assert reloaded == cfg

    def test_custom_values(self) -> None:
        cfg = pm.ToolFlext(
            project=pm.ToolFlextProject(project_class="platform"),
            namespace=pm.ToolFlextNamespace(
                alias_parent_sources={"c": "flext_cli"},
            ),
            docs=pm.ToolFlextDocs(project_class="platform", site_title="Platform"),
            aliases=pm.ToolFlextAliases(overrides={"c": "FlextCliConstants"}),
        )
        assert cfg.project.project_class == "platform"
        assert cfg.namespace.alias_parent_sources["c"] == "flext_cli"
        assert cfg.docs.site_title == "Platform"
        assert cfg.aliases.overrides["c"] == "FlextCliConstants"

    def test_tool_flext_project_defaults(self) -> None:
        sub = pm.ToolFlextProject()
        assert sub.class_stem_override is None
        assert sub.project_class == "library"

    def test_tool_flext_docs_defaults(self) -> None:
        sub = pm.ToolFlextDocs()
        assert sub.project_class == "library"
        assert sub.site_title is None

    def test_tool_flext_aliases_defaults(self) -> None:
        sub = pm.ToolFlextAliases()
        assert sub.overrides == {}
