"""Verify project-metadata SSOT is flat on c/m/p/t/u facades.

Phase 5 wiring contract: SSOT classes are inherited via MRO into the
public facades, so consumers read the SSOT as direct attributes on
``c``/``m``/``p``/``t``/``u`` — never via a ``.Project`` sub-namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c, m, p, t, u


class TestFacadeFlatSsotAccess:
    def test_c_constants_flat(self) -> None:
        assert c.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.TIER_FACADE_PREFIX["tests"] == "TestsFlext"
        assert c.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"
        assert "tool.flext.project" in c.MANAGED_PYPROJECT_KEYS

    def test_m_models_flat(self) -> None:
        assert m.ProjectMetadata is not None  # type: ignore[truthy-bool]
        assert m.ProjectNamespaceConfig is not None  # type: ignore[truthy-bool]
        assert m.ProjectToolFlext is not None  # type: ignore[truthy-bool]

    def test_p_protocols_flat(self) -> None:
        assert p.ProjectMetadataReader is not None  # type: ignore[truthy-bool]
        assert p.ProjectClassStemDeriver is not None  # type: ignore[truthy-bool]
        assert p.ProjectTierFacadeNamer is not None  # type: ignore[truthy-bool]

    def test_t_types_flat(self) -> None:
        assert hasattr(t, "ProjectAliasName")
        assert hasattr(t, "ProjectClassStem")
        assert hasattr(t, "ProjectAliasToSuffixMap")

    def test_u_utilities_flat(self) -> None:
        assert u.derive_class_stem("flext-ldif") == "FlextLdif"
        assert u.derive_package_name("flext-ldif") == "flext_ldif"
        assert u.derive_tier_facade_name("flext-ldif", "src") == "FlextLdif"
        assert u.pascalize("flext-ldif") == "FlextLdif"
