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
from email.utils import parseaddr
from functools import cache
from importlib.metadata import Distribution, DistributionFinder, PackageMetadata
from typing import TYPE_CHECKING, ClassVar

from packaging.version import Version

from flext_core._constants.file import FlextConstantsFile
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
from flext_core._models import FlextModelsBase, FlextModelsProjectMetadata
from flext_core._protocols import FlextProtocolsProjectMetadata
from flext_core._typings import FlextTypingBase

if TYPE_CHECKING:
    from pathlib import Path


class FlextUtilitiesProjectMetadata(
    FlextModelsBase,
):
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
        """Read and validate the ``pyproject.toml`` document at one root.

        The parsed document is cached per root path.

        Args:
            root: The project root holding ``pyproject.toml``.

        Returns:
            The resulting ``FlextModelsProjectMetadata.PyprojectDocument``.

        """
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
        """Build canonical project metadata from a parsed pyproject document.

        When the document carries no ``[project]`` table, the package name
        falls back to the docs override (or an unknown marker) and the version
        to a placeholder.

        Args:
            root: The project root the metadata describes.
            document: The parsed ``pyproject.toml`` document.

        Returns:
            The resulting ``FlextModelsProjectMetadata.ProjectMetadata``.

        """
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
        """Derive the CamelCase class stem from a project name.

        Underscore-separated segments each capitalize; a configured
        special-name override wins over the derivation.

        Args:
            project_name: The project distribution name.

        Returns:
            The resulting ``str``.

        """
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
        """Report whether a project declares a distribution by name.

        The project's own name and every ``Requires-Dist`` entry participate;
        separators normalize, so ``flext_core`` matches ``flext-core``.

        Args:
            metadata: The project metadata to inspect.
            distribution_name: The distribution name to look for.

        Returns:
            The resulting ``bool``.

        """
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

    @staticmethod
    def package_author(package_metadata: PackageMetadata) -> FlextTypingBase.StrPair:
        """Return the first normalized author identity from package metadata.

        Returns:
            The author name and email address.

        Raises:
            ValueError: If the Author-Email metadata carries no address.

        """
        raw_email = package_metadata.get("Author-Email", "")
        author_name, author_email = parseaddr(raw_email, strict=True)
        if raw_email and not author_email:
            msg = f"invalid Author-Email package metadata: {raw_email!r}"
            raise ValueError(msg)
        return (
            package_metadata.get("Author", "") or author_name or author_email,
            author_email,
        )

    @staticmethod
    def package_homepage(package_metadata: PackageMetadata) -> str:
        """Return the legacy Home-Page or labeled Homepage project URL.

        Returns:
            The homepage URL, empty when the metadata declares none.

        Raises:
            ValueError: If a Homepage project URL entry carries no URL.

        """
        if homepage := package_metadata.get("Home-Page", ""):
            return homepage
        for project_url in package_metadata.get_all("Project-URL") or ():
            project_url_text = str(project_url)
            label, separator, url = project_url_text.partition(",")
            if label.strip().casefold() != "homepage":
                continue
            if not separator or not url.strip():
                msg = f"invalid Homepage project URL metadata: {project_url_text!r}"
                raise ValueError(msg)
            return url.strip()
        return ""

    @staticmethod
    def package_version_info(version: str) -> FlextTypingBase.Triple[int, int, int]:
        """Return the exact three-component PEP 440 release tuple.

        Returns:
            The integer major, minor and patch release components.

        Raises:
            ValueError: If the version is not PEP 440 or its release does not
                have exactly three components.

        """
        try:
            major, minor, patch = Version(version).release
        except ValueError as exc:
            msg = f"invalid three-part semantic version metadata: {version!r}"
            raise ValueError(msg) from exc
        return major, minor, patch

    class PackageVersion:
        """MRO base deriving package metadata for each class declaring ``_metadata``.

        A subclass declares only ``_metadata``; every public value is derived
        once when that subclass is created, so invalid metadata fails at its
        definition.
        """

        _metadata: PackageMetadata
        __version__: str = ""
        __version_info__: FlextTypingBase.Triple[int, int, int]
        __title__: str = ""
        __description__: str = ""
        __author__: str = ""
        __author_email__: str = ""
        __license__: str = ""
        __url__: str = ""

        def __init_subclass__(cls, **kwargs: str | float | bool | None) -> None:
            """Derive the public metadata when a subclass declares ``_metadata``."""
            super().__init_subclass__(**kwargs)
            if "_metadata" not in cls.__dict__:
                return
            owner = FlextUtilitiesProjectMetadata
            package_metadata = cls._metadata
            cls.__version__ = package_metadata["Version"]
            cls.__version_info__ = owner.package_version_info(cls.__version__)
            cls.__title__ = package_metadata.get("Name", "")
            cls.__description__ = package_metadata.get("Summary", "")
            cls.__author__, cls.__author_email__ = owner.package_author(
                package_metadata,
            )
            cls.__license__ = package_metadata.get(
                "License-Expression",
                "",
            ) or package_metadata.get("License", "")
            cls.__url__ = owner.package_homepage(package_metadata)


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]
