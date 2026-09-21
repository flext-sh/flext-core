"""Fixed project-metadata constants."""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import t


class FlextConstantsProjectMetadata:
    """Fixed project-metadata constants exposed flat on `c.*`."""

    # NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): immutable
    # pairs replace the model-less mapping while retaining the naming policy.
    SPECIAL_NAME_OVERRIDES: ClassVar[t.StrPairTuple] = (
        ("flext", "FlextRoot"),
        ("flext-core", "Flext"),
    )
    PYPROJECT_FILENAME: ClassVar[str] = "pyproject.toml"
    PROJECT_VERSION_PLACEHOLDER: ClassVar[str] = "0.0.0"
    METADATA_SCHEMA_VERSION_DEFAULT: ClassVar[str] = "1.0.0"
