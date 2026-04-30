"""Verify project-metadata SSOT is flat on c/m/p/t/u facades.

Phase 5 wiring contract: SSOT classes are inherited via MRO into the
public facades, so consumers read the SSOT as direct attributes on
``c``/``m``/``p``/``t``/``u`` — never via a ``.Project`` sub-namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import c, m


class TestsFlextFacadeFlatSsotAccess:
    def test_c_constants_flat(self) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"
        assert c.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert c.TIER_FACADE_PREFIX["src"] == "Flext"

    def test_m_models_flat(self) -> None:
        assert isinstance(m.ProjectMetadata, type)
        assert isinstance(m.ProjectNamespaceConfig, type)
        assert isinstance(m.ProjectToolFlext, type)

    def test_u_utilities_flat(self) -> None:
        assert m.derive_class_stem("flext-core") == "Flext"
        assert m.pascalize("flext-ldif") == "FlextLdif"
