"""FLEXT infra tests module.

Provides MRO-based test infrastructure classes extending flext_tests base classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests.infra.constants import FlextInfraTestConstants as FlextInfraTestConstants
from tests.infra.constants import c as c
from tests.infra.models import FlextInfraTestModels as FlextInfraTestModels
from tests.infra.models import m as m
from tests.infra.protocols import FlextInfraTestProtocols as FlextInfraTestProtocols
from tests.infra.protocols import p as p
from tests.infra.typings import FlextInfraTestTypes as FlextInfraTestTypes
from tests.infra.typings import t as t
from tests.infra.utilities import FlextInfraTestUtilities as FlextInfraTestUtilities
from tests.infra.utilities import u as u

__all__ = [
    "FlextInfraTestConstants",
    "c",
    "FlextInfraTestModels",
    "m",
    "FlextInfraTestProtocols",
    "p",
    "FlextInfraTestTypes",
    "t",
    "FlextInfraTestUtilities",
    "u",
]
