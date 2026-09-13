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

# AGENT-COORDINATION (2026-07-11, ai-hub-mkzg): p/t MUST stay a RUNTIME import.
# beartype.claw evaluates annotations at runtime; moving FlextProtocols/FlextTypes
# under TYPE_CHECKING raises NameError at import (test_beartype_engine_claw_packages).
# Do NOT "optimize" this into a TYPE_CHECKING block. Same precedent: model_options.py,
# model_runtime.py. Contact owner of bead ai-hub-mkzg before touching this line.
from flext_core import FlextProtocols as p, FlextTypes as t, r


class FlextUtilitiesSettings:
    """Settings utilities for environment resolution and DI registration.

    ``resolve_env_file`` was deleted: it duplicated the settings-layer owner
    (``FlextSettings.resolve_env_file``) without namespace support. Chain law:
    the algorithm and its protocol constants live once in ``_settings.py``.
    """

    @staticmethod
    def resolve_process_environment() -> dict[str, str]:
        """Resolve the inherited process environment as a plain string mapping."""
        return dict(os.environ)

    @staticmethod
    def register_factory(
        container: p.Container, name: str, factory: t.FactoryCallable
    ) -> p.Result[bool]:
        """Register factory in DI container, verifying resolution succeeds."""
        _ = container.factory(name, factory)
        resolved = container.resolve(name)
        if resolved.failure:
            return r[bool].from_failure(resolved)
        return r[bool].ok(True)


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesSettings"]
