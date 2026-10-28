"""Settings helpers for parameter access and manipulation.

Only the helpers used by ``src/`` (any FLEXT project) are retained.  Dead
validators (URL scheme / trace-debug), log-level bootstraps, and the
``_try_get_*`` attribute-resolution ladders were removed after the
fire-test audit - callers that need narrowing must go through Pydantic
models directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableSequence


class FlextUtilitiesSettings:
    """Settings utilities resolving the inherited process environment for consumers.

    ``resolve_env_file`` was deleted: it duplicated the settings-layer owner
    (``FlextSettings.resolve_env_file``) without namespace support. Chain law:
    the algorithm and its protocol constants live once in ``_settings.py``.
    ``register_factory`` was deleted: no runtime consumer called it; the
    container's own ``factory``/``resolve`` contract is the registration owner.
    """

    @staticmethod
    def resolve_process_environment() -> Mapping[str, str]:
        """Resolve the inherited process environment as a plain string mapping.

        Returns:
            The resulting ``Mapping[str, str]``.

        """
        return dict(os.environ)


__all__: MutableSequence[str] = ["FlextUtilitiesSettings"]
