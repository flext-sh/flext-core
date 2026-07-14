"""Canonical project metadata boundary and derivation utilities."""

from __future__ import annotations

import tomllib

from flext_core._constants.file import FlextConstantsFile as cf
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata as ppm
from flext_core._protocols.result import FlextProtocolsResult as pr
from flext_core.result import FlextResult as r

from pathlib import Path


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): this utility
# retains only the useful ingress and naming owners. Lazy alias discovery stays
# on the existing beartype utility facade; ProjectConstants and scan copies are
# deleted instead of being transported through another model.
class FlextUtilitiesProjectMetadata(mpm):
    """Project metadata ingress and canonical name derivation."""

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

    @staticmethod
    def read_project_metadata(root: Path) -> pr.Result[ppm.ProjectMetadata]:
        """Validate one project document and retain its exact nested models."""
        pyproject = root / cf.PYPROJECT_FILENAME
        try:
            with pyproject.open("rb") as stream:
                document = mpm.PyprojectDocument.model_validate(tomllib.load(stream))
            project = document.project
            flext = document.tool.flext
            metadata = mpm.ProjectMetadata(
                root=root,
                package_name=(
                    flext.docs.package_name or project.name.replace("-", "_")
                ),
                class_stem=(
                    flext.project.class_stem_override
                    or FlextUtilitiesProjectMetadata.derive_class_stem(project.name)
                ),
                project=project,
                flext=flext,
            )
        except (OSError, ValueError) as exc:
            return r[ppm.ProjectMetadata].fail(
                f"cannot load project metadata from {pyproject}: {exc}"
            )
        return r[ppm.ProjectMetadata].ok(metadata)


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]
