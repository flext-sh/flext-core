"""Behavioral tests for c — the public production constants facade.

These tests assert the OBSERVABLE public contract of the constants facade:
semantic range invariants (MIN < MAX), StrEnum wire-value semantics that
callers rely on for routing/serialization, the public token/method catalogs,
and facade completeness (every advertised member is reachable). They do NOT
poke private internals (`_constants` package, `_LAZY_IMPORTS`, module `vars()`).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import enum

import pytest
from flext_tests import tm

from tests.constants import c


class TestsFlextConstantsNew:
    """Observable public-contract behavior over the c facade."""

    # ----------------------------------------------------------------- ranges
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

    # -------------------------------------------------- StrEnum wire semantics
    @pytest.mark.parametrize(
        "domain_enum",
        [
            c.Status,
            c.HealthStatus,
            c.ErrorDomain,
            c.LogLevel,
            c.Environment,
            c.HandlerType,
            c.SerializationFormat,
            c.BackoffStrategy,
            c.ParserCase,
        ],
    )
    def test_domain_enums_are_string_valued_for_routing(
        self, domain_enum: type[enum.Enum]
    ) -> None:
        """Every routing enum is a StrEnum whose str() equals its wire value.

        Callers serialize these members to strings and compare against raw wire
        tokens, so ``str(member) == member.value`` is a load-bearing contract.
        """
        tm.that(issubclass(domain_enum, enum.StrEnum), eq=True)
        members = list(domain_enum)
        tm.that(bool(members), eq=True, msg=f"{domain_enum.__name__} is empty")
        for member in members:
            tm.that(isinstance(member, str), eq=True)
            tm.that(str(member), eq=member.value)

    @pytest.mark.parametrize(
        "domain_enum",
        [c.Status, c.ErrorDomain, c.LogLevel, c.Environment, c.SerializationFormat],
    )
    def test_domain_enum_value_lookup_roundtrips(
        self, domain_enum: type[enum.StrEnum]
    ) -> None:
        """A caller can reconstruct any member from its public wire value."""
        for member in domain_enum:
            tm.that(domain_enum(member.value), eq=member)

    def test_error_domain_string_conversion_returns_value(self) -> None:
        """ErrorDomain.__str__ returns enum value for routing semantics."""
        tm.that(str(c.ErrorDomain.VALIDATION), eq=c.ErrorDomain.VALIDATION.value)

    # ----------------------------------------------------- parser token tables
    def test_boolean_truthy_and_falsy_sets_are_disjoint(self) -> None:
        """A token can never be simultaneously truthy and falsy."""
        truthy = set(c.PARSER_BOOLEAN_TRUTHY)
        falsy = set(c.PARSER_BOOLEAN_FALSY)
        tm.that(bool(truthy), eq=True)
        tm.that(bool(falsy), eq=True)
        tm.that(truthy.isdisjoint(falsy), eq=True)

    @pytest.mark.parametrize(
        "token", tuple(c.PARSER_BOOLEAN_TRUTHY) + tuple(c.PARSER_BOOLEAN_FALSY)
    )
    def test_boolean_tokens_are_lowercase_and_spaceless(self, token: str) -> None:
        """Validation token sets remain normalized (lowercase, no whitespace)."""
        tm.that(token, eq=token.lower())
        tm.that(" " not in token, eq=True)

    def test_string_method_map_exposes_core_type_predicates(self) -> None:
        """STRING_METHOD_MAP publishes the type-guard token vocabulary callers use."""
        tokens = set(c.STRING_METHOD_MAP)
        expected = {"str", "int", "float", "bool", "list", "dict", "tuple", "none"}
        tm.that(expected.issubset(tokens), eq=True, msg=f"missing: {expected - tokens}")

    # -------------------------------------------------- identifier regex rules
    @pytest.mark.parametrize(("raw_app_id", "normalized"), c.Tests.FORMAT_APP_ID_CASES)
    def test_app_id_cases_match_core_identifier_regex(
        self, raw_app_id: str, normalized: str
    ) -> None:
        """Shared flat test cases must produce identifiers accepted by core regex rules."""
        tm.that(bool(c.PATTERN_IDENTIFIER_LOWERCASE_RE.fullmatch(normalized)), eq=True)
        tm.that(" " not in normalized, eq=True)
        tm.that("\t" not in raw_app_id, eq=True)

    @pytest.mark.parametrize(("raw", "expected"), c.Tests.SAFE_STRING_VALID_CASES)
    def test_safe_string_valid_cases_align_with_parser_tokens(
        self, raw: str, expected: str
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
        self, raw: str | None, _reason: str
    ) -> None:
        """Invalid fixture values should fail core identifier matching."""
        candidate = "" if raw is None else raw.strip()
        tm.that(
            bool(c.PATTERN_IDENTIFIER_WITH_UNDERSCORE_RE.fullmatch(candidate)), eq=False
        )

    @pytest.mark.parametrize(
        "version",
        [
            "0.12.0",
            "0.12.0rc0",
            "0.12.0.rc0",
            "1.2.3a1",
            "1.2.3b2",
            "1.2.3.dev4",
            "1.2.3.post5",
            "1.2.3-rc.1",
            "1.2.3+linux.x86_64",
        ],
    )
    def test_version_pattern_accepts_semver_and_pep440(self, version: str) -> None:
        """Version validation accepts published and normalized metadata forms."""
        tm.that(bool(c.PATTERN_SEMVER_RE.fullmatch(version)), eq=True)

    @pytest.mark.parametrize(
        "version", ["1.2", "1.2.3rc", "1.2.3+", "1.2.3..rc0", "1.2.3-"]
    )
    def test_version_pattern_rejects_incomplete_versions(self, version: str) -> None:
        """Version validation rejects incomplete prerelease and local segments."""
        tm.that(bool(c.PATTERN_SEMVER_RE.fullmatch(version)), eq=False)

    @pytest.mark.parametrize(
        "distinguished_name",
        [
            "CN=John Doe",
            "CN=John Doe, OU=People, DC=example, DC=com",
            "uid=user-123,dc=example",
        ],
    )
    def test_ldap_dn_pattern_accepts_complete_components(
        self, distinguished_name: str
    ) -> None:
        """LDAP DN validation accepts complete comma-delimited components."""
        tm.that(bool(c.PATTERN_LDAP_DN_RE.fullmatch(distinguished_name)), eq=True)

    @pytest.mark.parametrize(
        "distinguished_name", ["", "=value", "CN=", "CN=   ", "CN=value,", "1CN=value"]
    )
    def test_ldap_dn_pattern_rejects_incomplete_components(
        self, distinguished_name: str
    ) -> None:
        """LDAP DN validation rejects missing names, values, and components."""
        tm.that(bool(c.PATTERN_LDAP_DN_RE.fullmatch(distinguished_name)), eq=False)

    def test_ldap_dn_pattern_rejects_adversarial_component_chain(self) -> None:
        """LDAP DN validation rejects the CodeQL adversarial shape."""
        adversarial = "A=+" + ",A=+ " * 256 + ",A="
        tm.that(bool(c.PATTERN_LDAP_DN_RE.fullmatch(adversarial)), eq=False)

    # ---------------------------------------------------- facade completeness
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
