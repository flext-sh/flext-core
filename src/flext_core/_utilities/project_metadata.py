"""Canonical project metadata boundary and derivation utilities.

Pure data ingress and naming utilities. Result object creation is explicit and
typed through ``p.Result`` contracts, using internal concrete helpers only for
construction.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import sys
import tomllib
from functools import cache
from importlib.metadata import Distribution, DistributionFinder
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants.file import FlextConstantsFile
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
from flext_core._models.project_metadata import FlextModelsProjectMetadata
from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from pathlib import Path


class FlextUtilitiesProjectMetadata(FlextModelsProjectMetadata):
    """Project metadata ingress and canonical name derivation."""

    _DISTRIBUTION_SEPARATOR_RE: ClassVar[FlextTypingBase.RegexPattern] = re.compile(
        r"[-_.]+",
    )
    _REQUIREMENT_NAME_RE: ClassVar[FlextTypingBase.RegexPattern] = re.compile(
        r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
        r"(?=\s*(?:\[|@|[<>=!~;]|$))",
    )

    @classmethod
    def _normalize_distribution_name(cls, distribution_name: str) -> str:
        return cls._DISTRIBUTION_SEPARATOR_RE.sub(
            "-",
            distribution_name.strip().lower(),
        )

    @staticmethod
    @cache
    def read_project_document_cached(
        root: Path,
    ) -> FlextModelsProjectMetadata.PyprojectDocument:
        pyproject = root / FlextConstantsFile.PYPROJECT_FILENAME
        with pyproject.open("rb") as stream:
            return FlextModelsProjectMetadata.PyprojectDocument.model_validate(
                tomllib.load(stream),
            )

    @classmethod
    def build_project_metadata(
        cls,
        root: Path,
        document: FlextModelsProjectMetadata.PyprojectDocument,
    ) -> FlextModelsProjectMetadata.ProjectMetadata:
        project = document.project
        flext = document.tool.flext
        if project is None:
            package_name = (
                flext.docs.package_name or FlextConstantsMixins.IDENTIFIER_UNKNOWN
            )
            class_stem = flext.project.class_stem_override or cls.derive_class_stem(
                package_name,
            )
            resolved_project = FlextModelsProjectMetadata.Project(
                name=package_name,
                version=FlextConstantsProjectMetadata.PROJECT_VERSION_PLACEHOLDER,
            )
            return FlextModelsProjectMetadata.ProjectMetadata(
                root=root,
                package_name=package_name,
                class_stem=class_stem,
                project=resolved_project,
                flext=flext,
            )
        resolved_project = project
        return FlextModelsProjectMetadata.ProjectMetadata(
            root=root,
            package_name=flext.docs.package_name
            or resolved_project.name.replace("-", "_"),
            class_stem=(
                flext.project.class_stem_override
                or cls.derive_class_stem(resolved_project.name)
            ),
            project=resolved_project,
            flext=flext,
        )

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        normalized = project_name.lower()
        override = next(
            (
                value
                for name, value in FlextConstantsProjectMetadata.SPECIAL_NAME_OVERRIDES
                if name == normalized
            ),
            None,
        )
        parts = normalized.replace("-", "_").split("_")
        return override or "".join(
            part[:1].upper() + part[1:] for part in parts if part
        )

    @staticmethod
    def installed_distributions(
        *,
        name: str | None = None,
        path: FlextTypingBase.StrSequence | None = None,
    ) -> FlextTypingBase.VariadicTuple[Distribution]:
        """Enumerate installed distributions over one snapshot of the finder chain.

        ``importlib.metadata.distributions()`` walks the live ``sys.meta_path``
        lazily: an import that inserts a finder ahead of ``PathFinder`` while
        the scan runs (in the same loop or in a concurrent thread) makes it
        yield every distribution twice. Each finder is queried once, from a
        snapshot taken before the scan.

        Returns:
            The distributions every snapshotted finder reports, in finder order.

        """
        context = (
            DistributionFinder.Context(name=name)
            if path is None
            else DistributionFinder.Context(name=name, path=list(path))
        )
        return tuple(
            distribution
            for finder in tuple(sys.meta_path)
            if isinstance(finder, FlextProtocolsProjectMetadata.DistributionSource)
            for distribution in finder.find_distributions(context)
        )

    @classmethod
    def distribution_requirement_names(
        cls,
        distribution: Distribution,
    ) -> FlextTypingBase.VariadicTuple[str]:
        """Return the normalized underscore names of declared requirements.

        Each ``Requires-Dist`` entry is parsed by the same requirement-name
        grammar as ``project_uses_distribution``; version specifiers, markers
        and extras never reach the returned name. The names keep the import
        grammar (underscores), matching the family-surface discovery seed.

        Returns:
            The normalized underscore name of every parseable requirement.
        """
        names: list[str] = []
        for requirement in distribution.requires or ():
            match = cls._REQUIREMENT_NAME_RE.match(requirement)
            if match is not None:
                names.append(match.group("name").lower().replace("-", "_"))
        return tuple(names)

    @classmethod
    def project_uses_distribution(
        cls,
        metadata: FlextProtocolsProjectMetadata.ProjectMetadata,
        distribution_name: str,
    ) -> bool:
        target_name = cls._normalize_distribution_name(distribution_name)
        if not target_name:
            return False
        if cls._normalize_distribution_name(metadata.project.name) == target_name:
            return True
        for dependency in metadata.project.dependencies:
            match = cls._REQUIREMENT_NAME_RE.match(dependency)
            if (
                match is not None
                and cls._normalize_distribution_name(match.group("name")) == target_name
            ):
                return True
        return False


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]
