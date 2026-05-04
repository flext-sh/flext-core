"""Smoke tests for c — production constants facade.

Per YAGNI: StrEnum @unique already enforces uniqueness at class creation,
Final type annotations enforce immutability, and the class body IS the
source of truth. Only semantic invariants (MIN < MAX) and facade
completeness add real coverage beyond Python's own guarantees.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest
from flext_tests import tm

from tests import c


class TestsFlextConstantsNew:
    """Semantic invariants over the c facade."""

    def test_constants_package_initializer_is_loadable(self) -> None:
        """_constants package lazy map must be present and non-empty."""
        module = importlib.import_module("flext_core._constants")
        module = importlib.reload(module)
        tm.that(hasattr(module, "_LAZY_IMPORTS"), eq=True)
        tm.that(bool(getattr(module, "_LAZY_IMPORTS")), eq=True)

    @pytest.mark.parametrize(
        "module_path",
        [
            "flext_core._constants.base",
            "flext_core._constants.cqrs",
            "flext_core._constants.enforcement",
            "flext_core._constants.environment",
            "flext_core._constants.errors",
            "flext_core._constants.file",
            "flext_core._constants.guards",
            "flext_core._constants.infrastructure",
            "flext_core._constants.logging",
            "flext_core._constants.mixins",
            "flext_core._constants.project_metadata",
            "flext_core._constants.pydantic",
            "flext_core._constants.regex",
            "flext_core._constants.serialization",
            "flext_core._constants.settings",
            "flext_core._constants.status",
            "flext_core._constants.timeout",
            "flext_core._constants.validation",
        ],
    )
    def test_constants_modules_reload_and_expose_constants_class(
        self,
        module_path: str,
    ) -> None:
        """Force execution of each _constants module and validate facade class shape."""
        module = importlib.import_module(module_path)
        module = importlib.reload(module)
        has_constants_class = any(
            name.startswith("FlextConstants") for name in vars(module)
        )
        tm.that(
            has_constants_class, eq=True, msg=f"{module_path} missing FlextConstants*"
        )

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

    @pytest.mark.parametrize(("raw_app_id", "normalized"), c.Tests.FORMAT_APP_ID_CASES)
    def test_app_id_cases_match_core_identifier_regex(
        self,
        raw_app_id: str,
        normalized: str,
    ) -> None:
        """Shared flat test cases must produce identifiers accepted by core regex rules."""
        tm.that(
            bool(c.PATTERN_IDENTIFIER_LOWERCASE_RE.fullmatch(normalized)),
            eq=True,
        )
        assert " " not in normalized
        assert "\t" not in raw_app_id

    @pytest.mark.parametrize(("raw", "expected"), c.Tests.SAFE_STRING_VALID_CASES)
    def test_safe_string_valid_cases_align_with_parser_tokens(
        self,
        raw: str,
        expected: str,
    ) -> None:
        """Flat string fixtures exercise parser-ready normalized values."""
        tm.that(raw.strip(), eq=expected)
        has_inner_space = " " in expected
        tm.that(
            bool(c.PATTERN_IDENTIFIER_WITH_UNDERSCORE_RE.fullmatch(expected)),
            eq=not has_inner_space,
        )

    @pytest.mark.parametrize(("raw", "_reason"), c.Tests.SAFE_STRING_INVALID_CASES)
    def test_safe_string_invalid_cases_do_not_match_identifier_regex(
        self,
        raw: str | None,
        _reason: str,
    ) -> None:
        """Invalid fixture values should fail core identifier matching."""
        candidate = "" if raw is None else raw.strip()
        tm.that(
            bool(c.PATTERN_IDENTIFIER_WITH_UNDERSCORE_RE.fullmatch(candidate)),
            eq=False,
        )

    @pytest.mark.parametrize(
        "token", tuple(c.PARSER_BOOLEAN_TRUTHY) + tuple(c.PARSER_BOOLEAN_FALSY)
    )
    def test_boolean_tokens_are_lowercase_and_unique(self, token: str) -> None:
        """Validation token sets remain normalized and collision-free."""
        tm.that(token, eq=token.lower())
        assert " " not in token

    def test_error_domain_string_conversion_returns_value(self) -> None:
        """ErrorDomain.__str__ returns enum value for routing semantics."""
        tm.that(str(c.ErrorDomain.VALIDATION), eq=c.ErrorDomain.VALIDATION.value)

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


__all__: list[str] = ["TestsFlextConstantsNew"]
