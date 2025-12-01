"""Test helpers for flext-core.

Provides reusable test utilities and helpers for all test modules.
Consolidates typings, constants, models, and protocols in unified classes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests.helpers.constants import TestConstants
from tests.helpers.data_mapper_helpers import (
    BadDict,
    BadDictGet,
    BadList,
    DataMapperTestCase,
    DataMapperTestHelpers,
    DataMapperTestType,
)
from tests.helpers.domain_helpers import (
    BadConfig,
    BadConfigTypeError,
    BadModelDump,
    ComplexValue,
    CustomEntity,
    DomainTestCase,
    DomainTestEntity,
    DomainTestHelpers,
    DomainTestType,
    DomainTestValue,
    ImmutableObj,
    MutableObj,
    NoConfigNoSetattr,
    NoDict,
    NoSetattr,
    SimpleValue,
)
from tests.helpers.models import TestModels
from tests.helpers.protocols import TestProtocols
from tests.helpers.result_helpers import (
    ResultCreationCase,
    ResultOperation,
    ResultTestCase,
    ResultTestHelpers,
)
from tests.helpers.typings import TestTypings

__all__ = [
    # Helper classes
    "BadConfig",
    "BadConfigTypeError",
    "BadDict",
    "BadDictGet",
    "BadList",
    "BadModelDump",
    "ComplexValue",
    "CustomEntity",
    "DataMapperTestCase",
    "DataMapperTestHelpers",
    "DataMapperTestType",
    "DomainTestCase",
    "DomainTestEntity",
    "DomainTestHelpers",
    "DomainTestType",
    "DomainTestValue",
    "ImmutableObj",
    "MutableObj",
    "NoConfigNoSetattr",
    "NoDict",
    "NoSetattr",
    "ResultCreationCase",
    "ResultOperation",
    "ResultTestCase",
    "ResultTestHelpers",
    "SimpleValue",
    # Unified test classes (herdando de flext-core)
    "TestConstants",
    "TestModels",
    "TestProtocols",
    "TestTypings",
]
