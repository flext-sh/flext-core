"""Type system foundation for flext-core tests.

Provides TestsFlextTypes, extending FlextTestsTypes with flext-core-specific types.
All generic test types come from flext_tests, only flext-core-specific additions here.

Architecture:
- FlextTestsTypes (flext_tests) = Generic types for all FLEXT projects
- TestsFlextTypes (tests/) = flext-core-specific types extending FlextTestsTypes
- All fixture models live in tests/models.py (TestsFlextModels.Fixtures)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from flext_core import T, T_co, T_contra, t
from flext_tests import (
    FlextTestsTypes,
)

from .models import TestsFlextModels


class TestsFlextTypes(FlextTestsTypes):
    """Type system foundation for flext-core tests - extends FlextTestsTypes.

    Architecture: Extends FlextTestsTypes with flext-core-specific type definitions.
    All generic types from FlextTestsTypes are available through inheritance.
    Fixture models are Pydantic v2 BaseModel from TestsFlextModels.Fixtures.

    Rules:
    - NEVER redeclare types from FlextTestsTypes
    - Only flext-core-specific types allowed (not generic for other projects)
    - All generic types come from FlextTestsTypes
    - All fixture models come from TestsFlextModels.Fixtures (Pydantic v2)
    """

    class Core:
        """Flext-core-specific type definitions for testing.

        Uses composition of t for type safety and consistency.
        Only defines types that are truly flext-core-specific.
        """

        type ServiceConfigMapping = Mapping[
            str,
            t.ContainerValue | Sequence[str] | Mapping[str, str | int] | None,
        ]
        """Service configuration mapping specific to flext-core services."""

        type HandlerConfigMapping = Mapping[
            str,
            t.ContainerValue | Sequence[str] | Mapping[str, str] | None,
        ]
        """Handler configuration mapping specific to flext-core handlers."""

    class Fixtures:
        """Aliases to Pydantic v2 fixture models from TestsFlextModels.Fixtures.

        All fixture models are Pydantic v2 BaseModel defined in tests/models.py.
        These aliases maintain backward compatibility for existing consumers.
        """

        # Direct aliases to Pydantic models in TestsFlextModels.Fixtures
        GenericFieldsDict = TestsFlextModels.Fixtures.GenericFieldsModel
        GenericTestCaseDict = TestsFlextModels.Fixtures.GenericTestCaseModel
        BddPhaseDict = TestsFlextModels.Fixtures.BddPhaseModel
        BddPhaseData = TestsFlextModels.Fixtures.BddPhaseData
        MockScenarioData = TestsFlextModels.Fixtures.MockScenarioData
        NestedDataDict = TestsFlextModels.Fixtures.NestedDataModel
        FixtureDataDict = TestsFlextModels.Fixtures.FixtureDataModel
        FixtureCaseDict = TestsFlextModels.Fixtures.FixtureCaseModel
        SuccessCaseDict = TestsFlextModels.Fixtures.SuccessCaseModel
        FailureCaseDict = TestsFlextModels.Fixtures.FailureCaseModel
        SetupDataDict = TestsFlextModels.Fixtures.SetupDataModel
        FixtureSuiteDict = TestsFlextModels.Fixtures.FixtureSuiteModel
        UserDataFixtureDict = TestsFlextModels.Fixtures.UserDataFixtureModel
        RequestDataFixtureDict = TestsFlextModels.Fixtures.RequestDataFixtureModel
        FixtureFixturesDict = TestsFlextModels.Fixtures.FixtureFixturesModel
        UserProfileDict = TestsFlextModels.Fixtures.UserProfileModel
        ConfigTestCaseDict = TestsFlextModels.Fixtures.ConfigTestCaseModel
        PerformanceMetricsDict = TestsFlextModels.Fixtures.PerformanceMetricsModel
        StressTestResultDict = TestsFlextModels.Fixtures.StressTestResultModel
        AsyncPayloadDict = TestsFlextModels.Fixtures.AsyncPayloadModel
        AsyncTestDataDict = TestsFlextModels.Fixtures.AsyncTestDataModel
        UserPayloadDict = TestsFlextModels.Fixtures.UserPayloadModel
        UpdateFieldDict = TestsFlextModels.Fixtures.UpdateFieldModel
        UpdatePayloadDict = TestsFlextModels.Fixtures.UpdatePayloadModel
        UserDataDict = TestsFlextModels.Fixtures.UserDataModel
        UpdateResultDict = TestsFlextModels.Fixtures.UpdateResultModel
        CommandPayloadDict = TestsFlextModels.Fixtures.CommandPayloadModel


__all__ = [
    "T",
    "T_co",
    "T_contra",
    "TestsFlextTypes",
]
