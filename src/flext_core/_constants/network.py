"""FlextConstantsNetwork - host, port, and hostname constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsNetwork:
    """SSOT for all network-related constants (hosts, ports, hostname limits)."""

    LOCALHOST: Final[str] = "localhost"
    LOOPBACK_IP: Final[str] = "127.0.0.1"
    MIN_PORT: Final[int] = 1
    MAX_PORT: Final[int] = 65535
    MAX_HOSTNAME_LENGTH: Final[int] = 253
