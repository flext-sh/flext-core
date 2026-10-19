"""Original decorated callables and a port-bearing registry consumer.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from examples import h, m, p, r, s, t


@h.handler(command=m.Examples.CommandA, priority=3)
def discovered_handler(message: p.Routable) -> t.JsonPayload:
    """Return the original owned payload without serializing it.

    Raises:
        TypeError: If the message is not CommandA.

    """
    if not isinstance(message, m.Examples.CommandA):
        msg = "Expected CommandA"
        raise TypeError(msg)
    return message


@h.handler(command=m.Examples.CommandB)
def amount_handler(message: p.Routable) -> p.Result[t.JsonPayload]:
    """Execute a second route through the canonical Result boundary.

    Returns:
        The command amount or an invalid-message failure.

    """
    if not isinstance(message, m.Examples.CommandB):
        return r[t.JsonPayload].fail("Expected CommandB")
    return r[t.JsonPayload].ok(message.amount)


class RegistryLookup(s[p.RegisterableService], m.Examples.RegistryLookupOptions):
    """Consume a registry supplied explicitly by the example composition root."""

    @override
    def execute(self) -> p.Result[p.RegisterableService]:
        """Resolve the requested plugin without constructing another registry.

        Returns:
            The registry's plugin lookup result.

        """
        return self.registry.fetch_plugin(self.category, self.plugin_name)
