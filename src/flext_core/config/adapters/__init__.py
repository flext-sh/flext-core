"""Configuration adapters package - DEPRECATED.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

🚨 ARCHITECTURAL VIOLATION FIXED:
These concrete adapters violated Clean Architecture by making flext-core
depend on concrete implementations (Singer, CLI, Django).

🎯 PROPER ARCHITECTURE:
- flext-core provides ONLY abstract interfaces and base classes
- Concrete projects implement their own adapters
- Use Dependency Injection (DI) for runtime discovery

📦 MIGRATION PATHS:
- CLI adapters: Moved to CLI implementation projects
- Singer adapters: Moved to orchestration projects
- Django adapters: Moved to web implementation projects
"""

from __future__ import annotations

import warnings
from typing import Any


class _DeprecatedAdapterWarning(DeprecationWarning):
    """Deprecated adapter access warning."""


def __getattr__(name: str) -> Any:
    """Handle deprecated adapter access with migration guidance."""
    # CLI adapters -> flext-cli
    cli_items = {"CLIConfig", "CLISettings", "cli_config_to_dict"}

    # Singer adapters -> orchestration projects
    singer_items = {
        "SingerConfig",
        "SingerSettings",
        "SingerTapConfig",
        "SingerTargetConfig",
        "singer_config_adapter",
    }

    # Django adapters -> flext-web
    django_items = {"DjangoSettings", "django_settings_adapter"}

    if name in cli_items:
        warnings.warn(
            f"❌ ARCHITECTURAL VIOLATION: '{name}' moved from flext-core to CLI projects.\n"
            f"🎯 CLEAN ARCHITECTURE: Use appropriate CLI project imports\n"
            f"💡 flext-core is abstract foundation, use DI for concrete implementations.\n"
            f"📖 This import will be removed in flext-core v0.8.0",
            _DeprecatedAdapterWarning,
            stacklevel=2,
        )
        msg = f"{name} moved to CLI projects for Clean Architecture compliance"
        raise ImportError(
            msg,
        )

    if name in singer_items:
        warnings.warn(
            f"❌ ARCHITECTURAL VIOLATION: '{name}' moved from flext-core to orchestration projects.\n"
            f"🎯 CLEAN ARCHITECTURE: Use appropriate orchestration project imports\n"
            f"💡 flext-core is abstract foundation, use DI for concrete implementations.\n"
            f"📖 This import will be removed in flext-core v0.8.0",
            _DeprecatedAdapterWarning,
            stacklevel=2,
        )
        msg = f"{name} moved to orchestration projects for Clean Architecture compliance"
        raise ImportError(
            msg,
        )

    if name in django_items:
        warnings.warn(
            f"❌ ARCHITECTURAL VIOLATION: '{name}' moved from flext-core to web projects.\n"
            f"🎯 CLEAN ARCHITECTURE: Use appropriate web project imports\n"
            f"💡 flext-core is abstract foundation, use DI for concrete implementations.\n"
            f"📖 This import will be removed in flext-core v0.8.0",
            _DeprecatedAdapterWarning,
            stacklevel=2,
        )
        msg = f"{name} moved to web projects for Clean Architecture compliance"
        raise ImportError(
            msg,
        )

    msg = f"module 'flext_core.config.adapters' has no attribute '{name}'"
    raise AttributeError(
        msg,
    )


# Export nothing - adapters moved to concrete projects
__all__: list[str] = []
