"""Public API for assertion and validation guards.

Provides type checking and assertion utilities for validating argument types
and runtime conditions. This is the main entry point for guard functionality.
"""

from __future__ import annotations

from flext_core import FlextUtilitiesGuardsEnsure


class FlextUtilitiesGuards(FlextUtilitiesGuardsEnsure):
    """Assertion and validation guard utilities.

    Inherits all guard methods from FlextUtilitiesGuardsEnsure and serves
    as the public API for type checking and runtime assertions.
    """
