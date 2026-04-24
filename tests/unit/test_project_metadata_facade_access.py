"""Verify project-metadata SSOT is flat on c/m/p/t/u facades.

Phase 5 wiring contract: SSOT classes are inherited via MRO into the
public facades, so consumers read the SSOT as direct attributes on
``c``/``m``/``p``/``t``/``u`` — never via a ``.Project`` sub-namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import c, m, p, u


class TestsFlextCoreFacadeFlatSsotAccess:
    def test_c_constants_flat(self) -> None:
        assert c.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.TIER_FACADE_PREFIX["tests"] == "TestsFlext"
        assert c.SPECIAL_NAME_OVERRIDES["flext-core"] == "Flext"

    def test_m_models_flat(self) -> None:
        assert isinstance(m.ProjectMetadata, type)
        assert isinstance(m.ProjectNamespaceConfig, type)
        assert isinstance(m.ProjectToolFlext, type)

    def test_p_protocols_flat(self) -> None:
        assert isinstance(p.ProjectMetadataReader, type)
        assert isinstance(p.ProjectClassStemDeriver, type)
        assert isinstance(p.ProjectTierFacadeNamer, type)

    def test_u_utilities_flat(self) -> None:
        assert u.derive_class_stem("flext-ldif") == "FlextLdif"
        assert u.derive_package_name("flext-ldif") == "flext_ldif"
        assert u.derive_tier_facade_name("flext-ldif", "src") == "FlextLdif"
        assert u.pascalize("flext-ldif") == "FlextLdif"
