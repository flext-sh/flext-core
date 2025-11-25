"""Test helpers for flext-core.

Provides reusable test utilities and helpers for all test modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
from tests.helpers.result_helpers import (
    ResultCreationCase,
    ResultOperation,
    ResultTestCase,
    ResultTestHelpers,
)

__all__ = [
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
]
