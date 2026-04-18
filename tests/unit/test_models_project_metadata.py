"""Tests for FlextModelsProjectMetadata — Tier 3 Pydantic models (flat on ``m.*``).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flext_core import (
    FlextModelsProjectMetadata as pm,
)


class TestProjectMetadata:
    def test_minimal(self) -> None:
        meta = pm.ProjectMetadata(
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
        meta = pm.ProjectMetadata(
            name="flext-core",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-core"),
        )
        assert meta.class_stem == "Flext"
        assert meta.src_facade_name == "Flext"
        assert meta.tests_facade_name == "TestsFlext"

    def test_flext_root_override(self) -> None:
        meta = pm.ProjectMetadata(
            name="flext",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext"),
        )
        assert meta.class_stem == "FlextRoot"

    def test_tier_facade_name_unknown_tier_raises(self) -> None:
        meta = pm.ProjectMetadata(
            name="flext-ldif",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-ldif"),
        )
        with pytest.raises(ValueError, match="unknown tier"):
            meta.tier_facade_name("nonsense")

    def test_frozen_rejects_mutation(self) -> None:
        meta = pm.ProjectMetadata(
            name="flext-ldif",
            version="0.12.0-dev",
            license="MIT",
            root=Path("/tmp/flext-ldif"),
        )
        with pytest.raises(ValidationError):
            setattr(meta, "name", "other")  # Frozen model rejects mutation

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            pm.ProjectMetadata(
                name="",
                version="0.12.0-dev",
                license="MIT",
                root=Path("/tmp/x"),
            )


class TestProjectNamespaceConfig:
    def test_defaults_include_universal_sources(self) -> None:
        cfg = pm.ProjectNamespaceConfig(project_name="flext-ldif")
        assert cfg.alias_parent_sources["r"] == "flext_core"
        assert cfg.alias_parent_sources["e"] == "flext_core"
        assert cfg.enabled is True

    def test_custom_override_for_c(self) -> None:
        cfg = pm.ProjectNamespaceConfig(
            project_name="flext-ldif",
            alias_parent_sources={"c": "flext_cli", "m": "flext_cli"},
        )
        assert cfg.alias_parent_sources["c"] == "flext_cli"
        assert cfg.alias_parent_sources["m"] == "flext_cli"
        assert cfg.alias_parent_sources["r"] == "flext_core"

    def test_unknown_alias_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown alias"):
            pm.ProjectNamespaceConfig(
                project_name="flext-ldif",
                alias_parent_sources={"zz": "flext_core"},
            )

    def test_universal_override_rejected(self) -> None:
        with pytest.raises(ValidationError, match="universal"):
            pm.ProjectNamespaceConfig(
                project_name="flext-ldif",
                alias_parent_sources={"r": "flext_ldif"},
            )


class TestProjectToolFlext:
    def test_defaults_round_trip(self) -> None:
        cfg = pm.ProjectToolFlext()
        dumped = cfg.model_dump()
        reloaded = pm.ProjectToolFlext.model_validate(dumped)
        assert reloaded == cfg

    def test_custom_values(self) -> None:
        cfg = pm.ProjectToolFlext(
            project=pm.ProjectToolFlextProject(project_class="platform"),
            namespace=pm.ProjectToolFlextNamespace(
                alias_parent_sources={"c": "flext_cli"},
            ),
            docs=pm.ProjectToolFlextDocs(
                project_class="platform", site_title="Platform"
            ),
            aliases=pm.ProjectToolFlextAliases(overrides={"c": "FlextCliConstants"}),
        )
        assert cfg.project.project_class == "platform"
        assert cfg.namespace.alias_parent_sources["c"] == "flext_cli"
        assert cfg.docs.site_title == "Platform"
        assert cfg.aliases.overrides["c"] == "FlextCliConstants"

    def test_project_subtable_defaults(self) -> None:
        sub = pm.ProjectToolFlextProject()
        assert sub.class_stem_override is None
        assert sub.project_class == "library"

    def test_docs_subtable_defaults(self) -> None:
        sub = pm.ProjectToolFlextDocs()
        assert sub.project_class == "library"
        assert sub.site_title is None

    def test_aliases_subtable_defaults(self) -> None:
        sub = pm.ProjectToolFlextAliases()
        assert sub.overrides == {}
