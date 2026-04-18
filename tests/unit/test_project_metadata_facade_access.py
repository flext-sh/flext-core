"""Verify c.Project.*, m.Project.*, p.Project.*, t.Project.*, u.Project.* access.

Phase 5 wiring contract: SSOT classes are inherited via MRO into the
public facades, so consumers read the SSOT as sub-namespace access.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, m, p, t, u


class TestFacadeProjectAccess:
    def test_c_project_constants(self) -> None:
        assert c.Project.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.Project.TIER_FACADE_PREFIX["tests"] == "TestsFlext"
        assert c.Project.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"
        assert "tool.flext.project" in c.Project.MANAGED_PYPROJECT_KEYS

    def test_m_project_models(self) -> None:
        assert m.Project.Definition is not None  # type: ignore[truthy-bool]
        assert m.Project.Namespace is not None  # type: ignore[truthy-bool]
        assert m.Project.ToolFlext is not None  # type: ignore[truthy-bool]

    def test_p_project_protocols(self) -> None:
        assert p.Project.MetadataReader is not None  # type: ignore[truthy-bool]
        assert p.Project.ClassStemDeriver is not None  # type: ignore[truthy-bool]
        assert p.Project.TierFacadeNamer is not None  # type: ignore[truthy-bool]

    def test_t_project_types(self) -> None:
        assert hasattr(t.Project, "AliasName")
        assert hasattr(t.Project, "ClassStem")
        assert hasattr(t.Project, "AliasToSuffixMap")

    def test_u_project_utilities(self) -> None:
        assert u.Project.derive_class_stem("flext-ldif") == "FlextLdif"
        assert u.Project.derive_package_name("flext-ldif") == "flext_ldif"
        assert u.Project.derive_tier_facade_name("flext-ldif", "src") == "FlextLdif"
