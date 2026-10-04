"""Project metadata utility tests.

Covers the surviving public project-metadata utilities: ``u.lazy_alias_suffixes``
(importable-package lazy-alias suffix table) and the ``m.PyprojectDocument``
ingress model (nested PEP 621 ``project`` + ``[tool.flext]`` contract).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from importlib.metadata import Distribution, DistributionFinder
from pathlib import Path
from types import ModuleType
from typing import override

import pytest
from flext_tests import tm

from flext_core import u
from tests.models import m


class TestsFlextCoreUtilitiesProjectMetadata:
    """Tests for ``FlextCoreUtilitiesProjectMetadata``."""

    @staticmethod
    def test_installed_distributions_survives_a_finder_inserted_mid_scan(
        tmp_path: Path,
    ) -> None:
        """A finder inserted ahead of the scan never re-yields distributions.

        ``importlib.metadata.distributions()`` re-walks ``sys.meta_path`` when
        an import inserts a finder while it iterates (an import in another gate
        thread did exactly that in CI); the owner scans one snapshot.
        """
        info = tmp_path / "probe_snapshot_dist-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: probe-snapshot-dist\nVersion: 1.0\n",
        )

        class ProbeFinder(DistributionFinder):
            """Reports the probe, then inserts a silent finder ahead of itself.

            The silent instance stands for the finder an import inserted in CI.
            """

            def __init__(self, *, reports: bool) -> None:
                self.reports = reports

            @override
            def find_spec(
                self,
                fullname: str,
                path: Sequence[str] | None = None,
                target: ModuleType | None = None,
            ) -> None:
                return None

            @override
            def find_distributions(
                self,
                context: DistributionFinder.Context | None = None,
            ) -> Iterator[Distribution]:
                if not self.reports or context is None:
                    return
                if context.name not in {None, "probe-snapshot-dist"}:
                    return
                yield Distribution.at(info)
                sys.meta_path.insert(0, ProbeFinder(reports=False))

        original = list(sys.meta_path)
        sys.meta_path.insert(0, ProbeFinder(reports=True))
        try:
            found = u.installed_distributions(name="probe-snapshot-dist")
        finally:
            sys.meta_path[:] = original
        tm.that([d.metadata["Name"] for d in found], eq=["probe-snapshot-dist"])

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

    @staticmethod
    def test_distribution_requirement_names_normalize_to_import_grammar(
        tmp_path: Path,
    ) -> None:
        """Requirement names keep the import grammar: lowercase underscores.

        Version specifiers, extras and environment markers never reach the
        returned name; a malformed line is skipped, never guessed.
        """
        info = tmp_path / "probe_requirement_dist-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            "Name: probe-requirement-dist\n"
            "Version: 1.0\n"
            "Requires-Dist: flext-cli>=0.12\n"
            "Requires-Dist: Flext-Api[extra]~=1.0\n"
            "Requires-Dist: pydantic>=2.11; python_version >= '3.13'\n"
            "Requires-Dist: some_pkg==2.0\n"
            "Requires-Dist: ??? malformed\n",
        )
        names = u.distribution_requirement_names(Distribution.at(info))
        tm.that(
            names,
            eq=("flext_cli", "flext_api", "pydantic", "some_pkg"),
        )
