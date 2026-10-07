"""_UniqueKeySafeLoader part.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from yaml import SafeLoader


class _UniqueKeySafeLoader(SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""
