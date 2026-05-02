"""Verify project-metadata SSOT is flat on c/m/p/t/u facades.

Phase 5 wiring contract: SSOT classes are inherited via MRO into the
public facades, so consumers read the SSOT as direct attributes on
``c``/``m``/``p``/``t``/``u`` — never via a ``.Project`` sub-namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests import c, m, u


class TestsFlextFacadeFlatSsotAccess:
    def test_c_constants_flat(self) -> None:
        assert c.PYPROJECT_FILENAME == "pyproject.toml"

    def test_m_models_flat(self) -> None:
        assert isinstance(m.ProjectMetadata, type)
        assert isinstance(m.ProjectNamespaceConfig, type)
        assert isinstance(m.ProjectToolFlext, type)

    def test_u_utilities_flat(self) -> None:
        constants = u.read_project_constants("flext-core")
        assert u.derive_class_stem("flext-core") == "Flext"
        assert constants.ALIAS_TO_SUFFIX["c"] == "Constants"
        assert m.pascalize("flext-ldif") == "FlextLdif"
