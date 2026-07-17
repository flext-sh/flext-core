"""Canonical project metadata boundary and derivation utilities."""

from __future__ import annotations

import re
import tomllib
from typing import ClassVar, TYPE_CHECKING

from flext_core._constants.file import FlextConstantsFile as cf
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata as ppm
from flext_core import FlextResult as r


if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t
    from flext_core._protocols.result import FlextProtocolsResult as pr
    from pathlib import Path


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): this utility
# retains only the useful ingress and naming owners. Lazy alias discovery stays
# on the existing beartype utility facade; ProjectConstants and scan copies are
# deleted instead of being transported through another model.
class FlextUtilitiesProjectMetadata(mpm):
    """Project metadata ingress and canonical name derivation."""

    _DISTRIBUTION_SEPARATOR_RE: ClassVar[t.RegexPattern] = re.compile(r"[-_.]+")
    _REQUIREMENT_NAME_RE: ClassVar[t.RegexPattern] = re.compile(
        r"^\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
        r"(?=\s*(?:\[|@|[<>=!~;]|$))"
    )

    @classmethod
    def _normalize_distribution_name(cls, distribution_name: str) -> str:
        """Return the PEP 503 normalized form of one distribution name."""
        return cls._DISTRIBUTION_SEPARATOR_RE.sub(
            "-", distribution_name.strip().lower()
        )

    @staticmethod
    def _read_project_document(root: Path) -> pr.Result[mpm.PyprojectDocument]:
        """Parse and validate one canonical pyproject document exactly once."""
        pyproject = root / cf.PYPROJECT_FILENAME
        try:
            with pyproject.open("rb") as stream:
                document = mpm.PyprojectDocument.model_validate(tomllib.load(stream))
        except (OSError, ValueError) as exc:
            return r[mpm.PyprojectDocument].fail(
                f"cannot load project metadata from {pyproject}: {exc}"
            )
        return r[mpm.PyprojectDocument].ok(document)

    @classmethod
    def _retain_project_metadata(
        cls, root: Path, document: mpm.PyprojectDocument
    ) -> pr.Result[ppm.ProjectMetadata]:
        """Retain exact validated declarations as canonical project metadata."""
        project = document.project
        if project is None:
            return r[ppm.ProjectMetadata].fail(
                f"cannot load project metadata from {root / cf.PYPROJECT_FILENAME}: "
                "PEP 621 [project] table is required"
            )
        flext = document.tool.flext
        metadata = mpm.ProjectMetadata(
            root=root,
            package_name=flext.docs.package_name or project.name.replace("-", "_"),
            class_stem=(
                flext.project.class_stem_override or cls.derive_class_stem(project.name)
            ),
            project=project,
            flext=flext,
        )
        return r[ppm.ProjectMetadata].ok(metadata)

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        """Return the total canonical class-stem projection for a project name."""
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
        """Return whether a project is or depends on a distribution.

        Project and dependency names use PEP 503 normalization. Dependency
        declarations retain their exact PEP 508 strings on the metadata model;
        only their leading distribution name is compared here.
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

    @classmethod
    def project_uses_distribution_at(
        cls, root: Path, distribution_name: str
    ) -> pr.Result[bool]:
        """Return whether a canonical project declares one distribution.

        A valid TOML document without a PEP 621 ``[project]`` table is not a
        package project and therefore returns ``False``. Missing or malformed
        documents remain failures at the filesystem ingress boundary.
        """
        document_result = cls._read_project_document(root)
        if document_result.failure:
            return r[bool].from_failure(document_result)
        document = document_result.value
        if document.project is None:
            return r[bool].ok(False)
        return cls._retain_project_metadata(root, document).map(
            lambda metadata: cls.project_uses_distribution(metadata, distribution_name)
        )

    @classmethod
    def read_project_metadata(cls, root: Path) -> pr.Result[ppm.ProjectMetadata]:
        """Validate one project document and retain its exact nested models."""
        document_result = cls._read_project_document(root)
        if document_result.failure:
            return r[ppm.ProjectMetadata].from_failure(document_result)
        return cls._retain_project_metadata(root, document_result.value)


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]
