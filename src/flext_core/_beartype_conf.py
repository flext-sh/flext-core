"""Beartype configuration for flext-core type checking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from beartype import BeartypeConf, BeartypeStrategy

BEARTYPE_CONF = BeartypeConf(
    strategy=BeartypeStrategy.Ologn,
    is_color=True,
    claw_is_pep526=False,
    warning_cls_on_decorator_exception=UserWarning,
)
