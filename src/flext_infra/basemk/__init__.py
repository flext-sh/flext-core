"""Base.mk template engine service.

Provides services for managing, validating, and rendering base.mk templates
for workspace build orchestration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.basemk.engine import TemplateEngine
from flext_infra.basemk.generator import BaseMkGenerator

__all__ = ["BaseMkGenerator", "TemplateEngine"]
