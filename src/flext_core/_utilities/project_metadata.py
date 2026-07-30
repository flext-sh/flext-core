"""Canonical project metadata boundary and derivation utilities.

Pure data ingress and naming utilities. Result object creation is explicit and
typed through ``p.Result`` contracts, using internal concrete helpers only for
construction.
"""

from __future__ import annotations

import re
import tomllib
from functools import cache
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants.file import FlextConstantsFile as cf
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata as ppm
from flext_core._protocols.result import FlextProtocolsResult as p
from flext_core.result import FlextResult as _Result
from flext_core._typings.base import FlextTypingBase as t

if TYPE_CHECKING:
    from pathlib import Path


class FlextUtilitiesProjectMetadata(mpm):
    """Project metadata ingress and canonical name derivation."""

    _DISTRIBUTION_SEPARATOR_RE: ClassVar[t.RegexPattern] = re.compile(r"[-_.]+")
    _REQUIREMENT_NAME_RE: ClassVar[t.RegexPattern] = re.compile(
        r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
        r"(?=\s*(?:\[|@|[<>=!~;]|$))"
    )

    @classmethod
    def _normalize_distribution_name(cls, distribution_name: str) -> str:
        return cls._DISTRIBUTION_SEPARATOR_RE.sub(
            "-", distribution_name.strip().lower()
        )

    @staticmethod
    @cache
    def read_project_document_cached(root: Path) -> mpm.PyprojectDocument:
        pyproject = root / cf.PYPROJECT_FILENAME
        with pyproject.open("rb") as stream:
            return mpm.PyprojectDocument.model_validate(tomllib.load(stream))

    @classmethod
    def build_project_metadata(
        cls, root: Path, document: mpm.PyprojectDocument
    ) -> ppm.ProjectMetadata:
        project = document.project
        flext = document.tool.flext
        if project is None:
            package_name = flext.docs.package_name or "unknown"
            class_stem = flext.project.class_stem_override or cls.derive_class_stem(
                package_name
            )
            resolved_project = mpm.Project(name=package_name, version="0.0.0")
            return mpm.ProjectMetadata(
                root=root,
                package_name=package_name,
                class_stem=class_stem,
                project=resolved_project,
                flext=flext,
            )
        resolved_project = project
        return mpm.ProjectMetadata(
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
    def read_project_metadata(root: Path) -> p.Result[ppm.ProjectMetadata]:
        """Read project metadata from ``pyproject.toml`` and return a protocol result.

        The method remains as a compatibility surface for existing consumers while
        delegating the canonical behavior to:

        - ``read_project_document_cached``
        - ``build_project_metadata``
        """
        try:
            project_root = root.resolve()
            document = FlextUtilitiesProjectMetadata.read_project_document_cached(
                project_root
            )
            return _Result[ppm.ProjectMetadata].ok(
                FlextUtilitiesProjectMetadata.build_project_metadata(
                    project_root, document
                )
            )
        except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
            msg = f"cannot read project metadata from {root}: {exc}"
            return _Result[ppm.ProjectMetadata].fail(msg, exception=exc)

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        normalized = project_name.lower()
        override = next(
            (value for name, value in cpm.SPECIAL_NAME_OVERRIDES if name == normalized),
            None,
        )
        parts = normalized.replace("-", "_").split("_")
        return override or "".join(
            part[:1].upper() + part[1:] for part in parts if part
        )

    @classmethod
    def project_uses_distribution(
        cls, metadata: ppm.ProjectMetadata, distribution_name: str
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
