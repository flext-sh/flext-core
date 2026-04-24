"""Smoke tests for c — production constants facade.

Per YAGNI: StrEnum @unique already enforces uniqueness at class creation,
Final type annotations enforce immutability, and the class body IS the
source of truth. Only semantic invariants (MIN < MAX) and facade
completeness add real coverage beyond Python's own guarantees.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests import c


class TestsFlextCoreConstantsNew:
    """Semantic invariants over the c facade."""

    def test_port_range_invariant(self) -> None:
        """MIN_PORT < MAX_PORT within valid TCP range."""
        tm.that(c.MIN_PORT, gt=0)
        tm.that(c.MAX_PORT, lte=65535)
        tm.that(c.MIN_PORT, lt=c.MAX_PORT)

    def test_timeout_range_invariant(self) -> None:
        """MIN <= DEFAULT <= MAX for timeout seconds."""
        tm.that(c.MIN_TIMEOUT_SECONDS, gt=0)
        tm.that(c.DEFAULT_TIMEOUT_SECONDS, gte=c.MIN_TIMEOUT_SECONDS)
        tm.that(c.MAX_TIMEOUT_SECONDS, gt=c.DEFAULT_TIMEOUT_SECONDS)

    def test_page_size_range_invariant(self) -> None:
        """MIN <= DEFAULT <= MAX for page sizes."""
        tm.that(c.MIN_PAGE_SIZE, gt=0)
        tm.that(c.DEFAULT_PAGE_SIZE, gte=c.MIN_PAGE_SIZE)
        tm.that(c.MAX_PAGE_SIZE, gte=c.DEFAULT_PAGE_SIZE)

    def test_retry_range_invariant(self) -> None:
        """MIN_RETRIES <= DEFAULT_RETRIES <= MAX_RETRIES."""
        tm.that(c.MIN_RETRIES, lte=c.DEFAULT_RETRIES)
        tm.that(c.DEFAULT_RETRIES, lte=c.MAX_RETRIES)

    @pytest.mark.parametrize(
        "attr",
        [
            # Critical facade members actively consumed in src/
            "NAME",
            "LOCALHOST",
            "LOOPBACK_IP",
            "DEFAULT_TIMEOUT_SECONDS",
            "DEFAULT_ENCODING",
            "DEFAULT_PAGE_SIZE",
            "HandlerType",
            "Status",
            "HealthStatus",
            "ErrorCode",
            "ErrorType",
            "ErrorDomain",
            "FailureLevel",
            "ContextScope",
            "ContextKey",
            "MetadataKey",
            "LogLevel",
            "Environment",
            "RegistrationScope",
            "MethodName",
            "BackoffStrategy",
            "SerializationFormat",
            "Compression",
            "ParserCase",
            "ParserBooleanToken",
            "STRING_METHOD_MAP",
            "PARSER_BOOLEAN_TRUTHY",
            "PARSER_BOOLEAN_FALSY",
            "HANDLER_ATTR",
            "FACTORY_ATTR",
        ],
    )
    def test_facade_attribute_accessible(self, attr: str) -> None:
        """All MRO-composed members accessible through c facade."""
        tm.that(hasattr(c, attr), eq=True, msg=f"c.{attr} missing")


__all__: list[str] = ["TestsFlextCoreConstantsNew"]
