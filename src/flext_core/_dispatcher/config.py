"""Configuration patterns specifically for the dispatcher.

This module provides the DispatcherConfig alias which is used throughout
the dispatcher components for configuration access.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.settings import FlextModelsConfig

DispatcherConfig = FlextModelsConfig.DispatcherConfig
