"""Shared beartype engine test helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing

from tests import t

type AnyAlias = str | typing.Any
type CleanAlias = str | int
type NestedAnyAlias = t.MappingKV[str, typing.Any]
