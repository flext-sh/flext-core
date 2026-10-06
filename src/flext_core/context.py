"""FlextContext — scoped key-value store + contextvar facade.

Pure Pydantic v2 model (m.ManagedModel). Zero utility-chain inheritance.
Context-variable I/O (correlation IDs, service metadata) exclusively via u.*.
Auto-injected transparently by FlextService via __init_subclass__.

Per AGENTS.md §0.7: zero nested classes; all methods flat on the facade.
Per AGENTS.md §3.1: single concern — context data model + contextvar ops.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import t
from flext_core._models.flext_context import FlextContext

# NOTE (multi-agent): mro-i6nq.12 — Generator is annotation-only; importing it
# under TYPE_CHECKING keeps the public runtime facade graph lazy.


__all__: t.StrSequence = ("FlextContext",)
