"""Fixed project-metadata constants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_core import FlextTypes as t


class FlextConstantsProjectMetadata:
    """Fixed project-metadata constants exposed flat on `c.*`."""

    # NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): immutable
    # pairs replace the model-less mapping while retaining the naming policy.
    SPECIAL_NAME_OVERRIDES: Final[t.StrPairTuple] = (
        ("flext", "FlextRoot"),
        ("flext-core", "Flext"),
    )
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
