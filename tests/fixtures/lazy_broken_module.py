"""Broken fixture — its body fails while executing, like a stale constants module.

The lazy loader must surface this failure as an ``ImportError`` chained to the
original ``AttributeError``: CPython's from-import discards an ``AttributeError``
escaping a module ``__getattr__`` together with its cause.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

msg = "fixture constants body references a removed attribute"
raise AttributeError(msg)
