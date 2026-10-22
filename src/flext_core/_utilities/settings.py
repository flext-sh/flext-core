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

# AGENT-COORDINATION (2026-07-11, ai-hub-mkzg): t MUST stay a RUNTIME import.
# beartype.claw evaluates annotations at runtime; moving FlextTypes
# under TYPE_CHECKING raises NameError at import (test_beartype_engine_claw_packages).
# Do NOT "optimize" this into a TYPE_CHECKING block. Same precedent: model_options.py,
# model_runtime.py. Contact owner of bead ai-hub-mkzg before touching this line.
from flext_core import t


class FlextUtilitiesSettings:
    """Settings utilities resolving the inherited process environment for consumers.

    ``resolve_env_file`` was deleted: it duplicated the settings-layer owner
    (``FlextSettings.resolve_env_file``) without namespace support. Chain law:
    the algorithm and its protocol constants live once in ``_settings.py``.
    ``register_factory`` was deleted: no runtime consumer called it; the
    container's own ``factory``/``resolve`` contract is the registration owner.
    """

    @staticmethod
    def resolve_process_environment() -> t.StrMapping:
        """Resolve the inherited process environment as a plain string mapping.

        Returns:
            The resulting ``t.StrMapping``.

        """
        return dict(os.environ)


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesSettings"]
