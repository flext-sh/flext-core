"""Compatibility shim for tests patching flext_core._config_base.

Provides aliases redirecting to the real base configuration implementation
in flext_core.config_base, so test patches like
"flext_core._config_base._BaseConfigOps..." resolve safely.
"""

from __future__ import annotations

from flext_core.config_base import (
    FlextConfigOperations as _BaseConfigDefaults,
    FlextConfigOperations as _BaseConfigOps,
    FlextConfigOperations as _BaseConfigValidation,
)

# Common helpers sometimes patched in tests


# Test compatibility shim removed due to type checking issues


__all__: list[str] = [
    "_BaseConfigDefaults",
    "_BaseConfigOps",
    "_BaseConfigValidation",
    # "dict", # Removed due to type issues
]
