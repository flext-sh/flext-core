"""Fixed project-metadata constants.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsProjectMetadata:
    """Fixed project-metadata constants exposed flat on `c.*`."""

    # NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): immutable
    # pairs replace the model-less mapping while retaining the naming policy.
    SPECIAL_NAME_OVERRIDES: Final[t.StrPairTuple] = (
        ("flext", "FlextRoot"),
        ("flext-core", "Flext"),
    )
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
    PROJECT_VERSION_PLACEHOLDER: Final[str] = "0.0.0"
    METADATA_SCHEMA_VERSION_DEFAULT: Final[str] = "1.0.0"
