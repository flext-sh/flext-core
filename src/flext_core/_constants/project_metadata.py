"""Fixed project-metadata constants."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsProjectMetadata:
    """Fixed project-metadata constants exposed flat on `c.*`."""

    SPECIAL_NAME_OVERRIDES: Final[t.StrMapping] = MappingProxyType({
        "flext": "FlextRoot",
        "flext-core": "Flext",
    })
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
