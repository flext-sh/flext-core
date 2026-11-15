# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Constants package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests._constants.domain import TestsFlextConstantsDomain
    from tests._constants.enums import TestsFlextConstantsEnums
    from tests._constants.errors import TestsFlextConstantsErrors
    from tests._constants.fixtures import TestsFlextConstantsFixtures
    from tests._constants.other import TestsFlextConstantsOther
    from tests._constants.result import TestsFlextConstantsResult
    from tests._constants.services import TestsFlextConstantsServices


__all__: tuple[str, ...] = (
    "TestsFlextConstantsDomain",
    "TestsFlextConstantsEnums",
    "TestsFlextConstantsErrors",
    "TestsFlextConstantsFixtures",
    "TestsFlextConstantsOther",
    "TestsFlextConstantsResult",
    "TestsFlextConstantsServices",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "TestsFlextConstantsDomain": ".domain",
        "TestsFlextConstantsEnums": ".enums",
        "TestsFlextConstantsErrors": ".errors",
        "TestsFlextConstantsFixtures": ".fixtures",
        "TestsFlextConstantsOther": ".other",
        "TestsFlextConstantsResult": ".result",
        "TestsFlextConstantsServices": ".services",
    },
    public_exports=__all__,
    deferred_members={
        "TestsFlextConstantsDomain": (
            "REPO_ROOT_PARENT_DEPTH",
            "SRC_DIR",
            "CORE_PACKAGE_DIR",
            "EXAMPLES_DIR",
            "PYPROJECT_FILENAME",
            "StatusEnum",
            "STATUS_ENUM",
            "STATUS_ACTIVE",
            "STATUS_PENDING",
            "STATUS_INACTIVE",
        ),
        "TestsFlextConstantsEnums": ("Color",),
        "TestsFlextConstantsErrors": (
            "USER_NOT_FOUND",
            "INVALID_EMAIL",
            "VALUE_TOO_LOW",
            "VALUE_TOO_HIGH",
            "TEST_ERROR",
            "NO_USER_IDS_PROVIDED",
            "SUBCLASSES_MUST_IMPLEMENT_EXECUTE",
            "HANDLER_ID_CANNOT_BE_EMPTY",
            "PROCESSING_ERROR_DEFAULT",
            "BAD_DICT_GET",
            "BAD_LIST_ITERATION",
            "CANNOT_INSTANTIATE",
            "UNEXPECTED_MESSAGE_TYPE",
            "VALIDATION_FAILED_FOR_TEST",
            "MIN_LENGTH_DEFAULT",
            "MAX_VALUE",
        ),
        "TestsFlextConstantsFixtures": (
            "Status",
            "MAX_VALUE",
            "MAX_RETRIES",
            "SAMPLE_PROJECT_NAME",
            "SAMPLE_PROJECT_VERSION",
            "SAMPLE_PROJECT_LICENSE",
            "SAMPLE_PROJECT_CLASS_STEM",
            "SAMPLE_ALIAS_PARENT_SOURCE",
            "SAMPLE_PROJECT_CLASS_PLATFORM",
            "SAMPLE_AUTHOR_ALICE",
            "SAMPLE_AUTHOR_BOB",
            "SAMPLE_PROJECT_NAME_MIGRATION",
        ),
        "TestsFlextConstantsOther": (
            "EXAMPLE_CHECK_COUNT_PATTERN",
            "DOCUMENTED_COMBINED_SUM",
            "DOCUMENTED_FLAT_MAP_VALUE",
            "DOCUMENTED_FLOW_VALUE",
            "DOCUMENTED_INCREMENTED_VALUE",
            "DOCUMENTED_RECOVERED_VALUE",
            "NON_GOLDEN_EXAMPLE_MODULES",
            "PUBLIC_EXAMPLES",
            "LAZY_BENCHMARK_REAL_SYMBOLS",
            "LAZY_BENCHMARK_EXTRA_INSTALL_MAPS",
            "CORE_PACKAGE_NAME",
            "PACKAGE_INFO_REQUIRED_KEYS",
            "AT_LEAST_CASES",
            "AT_LEAST_CASE_IDS",
            "CORE_SAFE_STRING_VALID_CASES",
            "CORE_SAFE_STRING_INVALID_CASES",
            "CORE_FORMAT_APP_ID_CASES",
        ),
        "TestsFlextConstantsResult": (
            "RailwayOperation",
            "RAILWAY_OPERATION_GET_EMAIL",
            "RAILWAY_OPERATION_SEND_EMAIL",
            "RAILWAY_OPERATION_GET_STATUS",
            "RAILWAY_OPERATION_DOUBLE",
            "RAILWAY_OPERATION_SQUARE",
            "RAILWAY_OPERATION_NEGATE",
            "RAILWAY_OPERATIONS",
        ),
        "TestsFlextConstantsServices": (
            "ServiceTestType",
            "ServiceType",
            "SERVICE_TEST_TYPE_GET_USER",
            "SERVICE_TEST_TYPE_VALIDATE",
            "SERVICE_TEST_TYPE_FAIL",
            "DEFAULT_USER_NAME_PREFIX",
            "DEFAULT_EMAIL_DOMAIN",
            "DEFAULT_ERROR_MESSAGE",
            "USER_IDS_SUCCESS",
            "USER_IDS_INVALID",
            "OPERATION_RESULT_KEY",
            "OPERATION_NAME_KEY",
            "OPERATION_FACTORS",
            "UNKNOWN_OPERATION_PREFIX",
        ),
    },
)
