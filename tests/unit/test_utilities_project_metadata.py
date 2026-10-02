"""Project metadata utility tests.

Covers the surviving public project-metadata utilities: ``u.lazy_alias_suffixes``
(importable-package lazy-alias suffix table) and the ``m.PyprojectDocument``
ingress model (nested PEP 621 ``project`` + ``[tool.flext]`` contract).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from flext_core import u
from tests.models import m


class TestsFlextCoreUtilitiesProjectMetadata:
    """Tests for ``FlextCoreUtilitiesProjectMetadata``."""

    @staticmethod
    def test_lazy_alias_suffixes_reads_the_public_package() -> None:
        """Test lazy alias suffixes reads the public package."""
        package_name = u.__module__.partition(".")[0]
        suffixes = u.lazy_alias_suffixes(package_name)
        tm.that(suffixes, is_=tuple)
        assert suffixes

    @staticmethod
    def test_lazy_alias_suffixes_propagates_missing_package() -> None:
        """Test lazy alias suffixes propagates missing package."""
        package_name = "nonexistent_distribution_xyz"
        with pytest.raises(ModuleNotFoundError) as raised:
            u.lazy_alias_suffixes(package_name)
        assert raised.value.name == package_name

    @staticmethod
    def test_pyproject_document_parses_nested_project_and_tool() -> None:
        """Test pyproject document parses nested project and tool."""
        doc = m.PyprojectDocument.model_validate({
            "project": {"name": "flext-ldif", "version": "1.0.0"},
            "tool": {"flext": {"workspace": {"attached": True}}},
        })
        dumped = doc.model_dump()
        tm.that(dumped["project"]["name"], eq="flext-ldif")
        tm.that(dumped["project"]["version"], eq="1.0.0")
        tm.that(dumped["tool"]["flext"]["workspace"]["attached"], eq=True)
