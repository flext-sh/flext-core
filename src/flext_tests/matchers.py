r"""Test matchers and assertions for FLEXT ecosystem tests.

Provides unified assertion API with powerful generalist methods.
Short alias: tm (test matchers)

Core Philosophy:
- MINIMAL API: Only 5 core methods (ok, fail, check, that, scope)
- POWERFUL: Each method handles multiple scenarios via optional parameters
- UNIVERSAL: tm.that() does ALL validations (equality, type, length, containment, etc.)

Core Methods (5 main methods):
    tm.ok(result, **kw)     # Assert FlextResult success, optional validation
    tm.fail(result, **kw)   # Assert FlextResult failure, optional validation
    tm.check(result)        # Railway-pattern chained assertions
    tm.that(value, **kw)    # Universal assertion - ALL validations in ONE method
    tm.scope()              # Isolated test context (context manager)

Usage Examples:
    # FlextResult assertions
    value = tm.ok(result)                    # Assert success, return value
    tm.ok(result, eq="expected")            # Assert success and equals
    tm.fail(result, contains="error")       # Assert failure with error check

    # Universal assertions (tm.that() does EVERYTHING)
    tm.that(x, gt=0, lt=100)                 # Comparisons
    tm.that(v, is_=str, none=False)          # Type and None
    tm.that(d, contains="key")               # Containment (dict/list/str)
    tm.that(lst, length=5, length_gt=0)      # Length checks
    tm.that(text, starts="http", ends="/")   # String validation
    tm.that(text, match="[0-9]{4}-[0-9]{2}")    # Regex match

    # Chained assertions
    tm.check(result).ok().eq(5).done()       # Railway pattern

Deprecated Methods (use tm.that() instead):
    tm.that() -> tm.that(actual, eq=eq=expected)
    tm.true() -> tm.that(condition, eq=True)
    tm.assert_contains() -> tm.that(container, contains=item)
    tm.str_() -> tm.that(text, contains/starts/ends/match/excludes/empty=...)
    tm.is_() -> tm.that(value, is_=type, none=...)
    tm.len() -> tm.that(items, length/length_gt/length_gte/empty=...)
    tm.hasattr() -> tm.that(hasattr(obj, attr), eq=True)
    tm.method() -> tm.that(hasattr(...), eq=True) + tm.that(callable(...), eq=True)
    tm.not_none() -> tm.that(value, none=False)

Note: For test data creation, use tb() (FlextTestsBuilders) instead:
    data = tb().with_users(10).with_configs(production=True).build()

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence, Sized
from contextlib import contextmanager
from pathlib import Path
from typing import TypeGuard, TypeVar

from flext_core import r
from flext_core.typings import t as core_t
from pydantic import BaseModel

from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u

TK = TypeVar("TK")
TV = TypeVar("TV")


def _is_key_value_pair[TK, TV](
    key_equals: tuple[TK, TV] | Sequence[tuple[TK, TV]] | None,
) -> TypeGuard[tuple[TK, TV]]:
    """Return True if key_equals is a single (key, value) tuple."""
    return isinstance(key_equals, tuple) and len(key_equals) == 2


def _is_non_string_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _to_test_payload(value: object) -> t.Tests.PayloadValue:
    if value is None or isinstance(value, str | int | float | bool | bytes | BaseModel):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_test_payload(v) for k, v in value.items()}
    if _is_non_string_sequence(value):
        return [_to_test_payload(seq_item) for seq_item in value]
    return str(value)


def _as_guard_input(value: object) -> core_t.GuardInputValue:
    if isinstance(value, BaseModel | str | int | float | bool | Path):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _as_guard_input(v) for k, v in value.items()}
    if _is_non_string_sequence(value):
        return [_as_guard_input(seq_item) for seq_item in value]
    return str(value)


def _check_has_lacks(
    value: object,
    has: object | Sequence[object] | None,
    lacks: object | Sequence[object] | None,
    msg: str | None,
    *,
    as_str: bool = False,
) -> None:
    """Shared has/lacks containment check for ok(), fail(), and that()."""
    if has is not None:
        items = list(has) if _is_non_string_sequence(has) else [has]
        for item in items:
            if as_str:
                check_str = str(item)
                target = str(value)
                if check_str not in target:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value, item=item
                        ),
                    )
            else:
                check_val = _as_guard_input(item)
                target = _as_guard_input(value)
                # Handle RootModel (e.g. ConfigMap) by extracting root dict
                if isinstance(target, BaseModel) and hasattr(target, "root"):
                    target = target.root
                if not isinstance(target, Mapping | str | list):
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value, item=item
                        ),
                    )

                if isinstance(target, str):
                    if str(check_val) not in target:
                        raise AssertionError(
                            msg
                            or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                                container=value, item=item
                            ),
                        )
                elif check_val not in target:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value, item=item
                        ),
                    )
    if lacks is not None:
        items = list(lacks) if _is_non_string_sequence(lacks) else [lacks]
        for item in items:
            if as_str:
                check_str = str(item)
                target = str(value)
                if check_str in target:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value, item=item
                        ),
                    )
            else:
                check_val = _as_guard_input(item)
                target = _as_guard_input(value)
                # Handle RootModel (e.g. ConfigMap) by extracting root dict
                if isinstance(target, BaseModel) and hasattr(target, "root"):
                    target = target.root
                if not isinstance(target, Mapping | str | list):
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value, item=item
                        ),
                    )

                if isinstance(target, str):
                    if str(check_val) in target:
                        raise AssertionError(
                            msg
                            or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                                container=value, item=item
                            ),
                        )
                elif check_val in target:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value, item=item
                        ),
                    )


class FlextTestsMatchers:
    """Test matchers with powerful generalist methods.

    Short alias: tm

    Core Methods (5 main methods):
        tm.ok(result, **kw)     - Assert FlextResult success, optional validation
        tm.fail(result, **kw)   - Assert FlextResult failure, optional validation
        tm.check(result)        - Railway-pattern chained assertions
        tm.that(value, **kw)    - Universal assertion - ALL validations in ONE method
        tm.scope()              - Isolated test context (context manager)

    The tm.that() method handles ALL assertion types:
        - Comparisons: eq, ne, gt, gte, lt, lte
        - Type/None: is_, none
        - Containment: contains (works for dict/list/str)
        - Strings: starts, ends, match, excludes
        - Length: length, length_gt, length_gte, length_lt, length_lte, empty

    Deprecated Methods (all redirect to tm.that()):
        tm.that() -> tm.that(actual, eq=eq=expected)
        tm.true() -> tm.that(condition, eq=True)
        tm.assert_contains() -> tm.that(container, contains=item)
        tm.str_() -> tm.that(text, contains/starts/ends/match/excludes/empty=...)
        tm.is_() -> tm.that(value, is_=type, none=...)
        tm.len() -> tm.that(items, length/length_gt/length_gte/empty=...)
        tm.hasattr() -> tm.that(hasattr(obj, attr), eq=True)
        tm.method() -> tm.that(hasattr(...), eq=True) + tm.that(callable(...), eq=True)
        tm.not_none() -> tm.that(value, none=False)
        tm.dict_() -> tm.that(data, contains=...) or tm.that(data, length=...)
        tm.that() -> tm.that(items, has=...) or tm.that(items, length=...)
        tm.that() -> tm.that(value, is_=is_=type, none=False, none=False)
    """

    # =========================================================================
    # CORE ASSERTIONS
    # =========================================================================

    @staticmethod
    def ok[TResult](
        result: r[TResult],
        **kwargs: t.Tests.PayloadValue,
    ) -> TResult | t.Tests.PayloadValue:
        """Enhanced assertion for FlextResult success with optional value validation.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated via m.Tests.Matcher.OkParams model.

        Examples:
            # Basic success assertions
            tm.ok(result)                      # Assert success
            tm.ok(result, eq=5)               # Success and value == 5
            tm.ok(result, is_=str, len=(1,100))  # Success, is string, len 1-100
            tm.ok(result, has=["a", "b"])     # Success and value contains both

            # Deep structural matching on result value
            tm.ok(result, deep={
                "user.name": "John",
                "user.email": lambda e: "@" in e,
            })

            # Path extraction first
            tm.ok(result, path="data.value", eq=42)

            # Custom validation
            tm.ok(result, where=lambda x: x.status == "active")

        Args:
            result: FlextResult to validate
            **kwargs: Parameters validated via m.Tests.Matcher.OkParams model
                - eq, ne: Equality/inequality check
                - is_: Runtime type check against single type or tuple
                - none, empty: Nullability checks
                - gt, gte, lt, lte: Comparison checks (numeric or length)
                - has, lacks: Unified containment (replaces contains)
                - starts, ends, match: String assertions
                - len: Length spec - exact int or (min, max) tuple
                - deep: Deep structural matching specification
                - path: Extract nested value via dot notation before validation
                - where: Custom predicate function for validation
                - msg: Custom error message
                - contains, starts, ends: Legacy parameters (deprecated)

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure or value doesn't satisfy constraints
            ValueError: If parameter validation fails (via Pydantic model)

        """
        try:
            params = m.Tests.Matcher.OkParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc

        if not result.is_success:
            raise AssertionError(
                params.msg or c.Tests.Matcher.ERR_OK_FAILED.format(error=result.error),
            )
        # Start with TResult, may be reassigned to extracted value (t.Tests.PayloadValue)
        result_value: TResult | t.Tests.PayloadValue = result.value

        # Path extraction first (if specified)
        if params.path is not None:
            # u.Mapper.extract expects str, not PathSpec
            # Type narrowing
            if isinstance(params.path, str):
                path_str: str = params.path
            else:
                path_str = ".".join(params.path)
            if not (
                (
                    hasattr(type(result_value), "__mro__")
                    and BaseModel in type(result_value).__mro__
                )
                or (hasattr(result_value, "keys") and hasattr(result_value, "items"))
            ):
                raise AssertionError(
                    params.msg
                    or f"Path extraction requires dict or model, got {type(result_value).__name__}",
                )
            extract_source: BaseModel | core_t.ConfigMapValue
            if isinstance(result_value, BaseModel):
                extract_source = result_value
            elif isinstance(result_value, Mapping):
                extract_source = {
                    str(k): _as_guard_input(v) for k, v in result_value.items()
                }
            else:
                raise AssertionError(
                    params.msg
                    or f"Path extraction requires dict or model, got {type(result_value).__name__}",
                )
            extracted = u.Mapper.extract(extract_source, path_str)
            if extracted.is_failure:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_SCOPE_PATH_NOT_FOUND.format(
                        path=path_str,
                        error=extracted.error,
                    ),
                )
            result_value = _to_test_payload(extracted.value)

        # Validate value with u.chk() - pass parameters directly for type safety
        # Note: u.chk() doesn't support tuple types for is_/not_, handle separately
        has_validation = (
            params.eq is not None
            or params.ne is not None
            or params.none is not None
            or params.empty is not None
            or params.gt is not None
            or params.gte is not None
            or params.lt is not None
            or params.lte is not None
            or params.starts is not None
            or params.ends is not None
            or params.match is not None
        )
        if has_validation:
            # u.chk() only accepts single type, not tuple
            is_type = params.is_ if not isinstance(params.is_, tuple) else None
            if not u.chk(
                _as_guard_input(result_value),
                eq=_as_guard_input(params.eq) if params.eq is not None else None,
                ne=_as_guard_input(params.ne) if params.ne is not None else None,
                is_=is_type,
                none=params.none,
                empty=params.empty,
                gt=params.gt,
                gte=params.gte,
                lt=params.lt,
                lte=params.lte,
                starts=params.starts,
                ends=params.ends,
                match=params.match,
            ):
                error_msg = (
                    params.msg or f"Value {result_value!r} did not satisfy constraints"
                )
                raise AssertionError(error_msg)
        # Handle tuple types separately
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not (
                type(result_value) in params.is_
                or (
                    hasattr(type(result_value), "__mro__")
                    and any(t in type(result_value).__mro__ for t in params.is_)
                )
            )
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=params.is_,
                    actual=type(result_value).__name__,
                ),
            )

        # Handle unified has/lacks (works for str, list, dict, set, tuple)
        _check_has_lacks(result_value, params.has, params.lacks, params.msg)
        # Length validation (delegate to u.Tests.Length)
        result_payload = _to_test_payload(result_value)
        if params.len is not None and not u.Tests.Length.validate(
            result_payload, params.len
        ):
            # Type guard: result_value has __len__ if it passed validation
            # Type narrow for __len__
            actual_len = len(result_value) if isinstance(result_value, Sized) else 0
            if isinstance(params.len, int):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_LEN_EXACT_FAILED.format(
                        expected=params.len,
                        actual=actual_len,
                    ),
                )
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_LEN_RANGE_FAILED.format(
                    min=params.len[0],
                    max=params.len[1],
                    actual=actual_len,
                ),
            )

        # Deep matching (delegate to u.Tests.DeepMatch)
        if params.deep is not None:
            # Type narrow for DeepMatch.match
            if not isinstance(result_value, BaseModel | Mapping):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(result_value).__name__}",
                )
            deep_input: BaseModel | Mapping[str, t.Tests.PayloadValue]
            if isinstance(result_value, BaseModel):
                deep_input = result_value
            else:
                deep_input = {
                    str(k): _to_test_payload(v) for k, v in result_value.items()
                }
            match_result = u.Tests.DeepMatch.match(deep_input, params.deep)
            if not match_result.matched:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_DEEP_PATH_FAILED.format(
                        path=match_result.path,
                        reason=match_result.reason,
                    ),
                )

        # Custom predicate
        if params.where is not None and not params.where(
            _to_test_payload(result_value)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_PREDICATE_FAILED.format(value=result_value),
            )

        # Type guard: result_value is not None after all validations
        if result_value is None:
            raise AssertionError(
                params.msg
                or "Value is None but validation passed - this should not happen",
            )
        # Return value - if no path extraction, this is TResult from result.value
        # If path extraction was used, this is the extracted payload value
        # Both paths are validated at this point
        if params.path is None:
            # No path extraction - return original result.value (TResult)
            return result.value
        # Path extraction case - result_value is payload value
        # Return type is TResult | payload value to accurately reflect this
        return result_value

    @staticmethod
    def fail[TResult](
        result: r[TResult],
        **kwargs: t.Tests.PayloadValue,
    ) -> str:
        r"""Enhanced assertion for FlextResult failure with optional error validation.

        Examples:
            # Basic failure assertions
            tm.fail(result)                   # Assert failure
            tm.fail(result, has="not found")  # Failure with error containing
            tm.fail(result, code="VALIDATION")  # Failure with specific code
            tm.fail(result, match=r"Error: \\d+")  # Error matches regex

            # Multiple error checks
            tm.fail(result, has=["invalid", "required"], lacks="internal")

            # Error metadata checks
            tm.fail(result, code="VALIDATION", data={"field": "email"})

        Args:
            result: FlextResult to check
            error: Expected error substring (legacy parameter, use has=)
            msg: Optional custom error message
            has: Unified containment - error contains substring(s) (replaces contains)
            lacks: Unified non-containment - error does NOT contain substring(s) (replaces excludes)
            starts: Assert error starts with prefix
            ends: Assert error ends with suffix
            match: Assert error matches regex
            code: Assert error code equals
            code_has: Assert error code contains substring(s)
            data: Assert error data contains key-value pairs
            contains: Legacy parameter (deprecated, use has=)
            excludes: Legacy parameter (deprecated, use lacks=)

        Returns:
            Error message from result

        Raises:
            AssertionError: If result is success or error doesn't satisfy constraints

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated via m.Tests.Matcher.FailParams model.

        Args:
            result: FlextResult to check
            error: Expected error substring (legacy parameter, use has=)
            **kwargs: Parameters validated via m.Tests.Matcher.FailParams model

        Returns:
            Error message from result

        Raises:
            AssertionError: If result is success or error doesn't satisfy constraints
            ValueError: If parameter validation fails (via Pydantic model)

        """
        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs accepts payload kwargs - Pydantic validates types
        # Legacy 'error' parameter is handled by FailParams model validator
        try:
            params = m.Tests.Matcher.FailParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc

        if result.is_success:
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_FAIL_EXPECTED.format(value=result.value),
            )
        err = result.error or ""

        # Apply error message validation if any check parameters provided
        # Legacy parameters (error, contains, excludes) already converted to has/lacks by model validator
        if params.has or params.lacks or params.starts or params.ends or params.match:
            _check_has_lacks(err, params.has, params.lacks, params.msg, as_str=True)
            # String assertions
            if params.starts is not None and not u.chk(err, starts=params.starts):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_STARTSWITH.format(
                        text=err,
                        prefix=params.starts,
                    ),
                )
            if params.ends is not None and not u.chk(err, ends=params.ends):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_ENDSWITH.format(
                        text=err,
                        suffix=params.ends,
                    ),
                )
            if params.match is not None and not u.chk(err, match=params.match):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_MATCHES.format(
                        text=err,
                        pattern=params.match,
                    ),
                )

        # Error code validation
        if params.code is not None:
            actual_code = result.error_code
            if actual_code != params.code:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_ERROR_CODE_MISMATCH.format(
                        expected=params.code,
                        actual=actual_code,
                    ),
                )

        if params.code_has is not None:
            actual_code = result.error_code or ""
            # ErrorCodeSpec is str | Sequence[str] - need to handle both cases
            # Convert to list[str] for uniform processing
            code_has_value = params.code_has
            if isinstance(code_has_value, str):
                items_list: list[str] = [code_has_value]
            else:
                # Sequence[str] case - convert to list
                items_list = [str(x) for x in code_has_value]
            for item in items_list:
                if item not in actual_code:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_CODE_NOT_CONTAINS.format(
                            expected=item,
                            actual=actual_code,
                        ),
                    )

        # Error data validation
        if params.data is not None:
            actual_raw = result.error_data
            actual_data: MutableMapping[str, t.Tests.PayloadValue] = {}
            if actual_raw is not None:
                root_value: object = (
                    actual_raw.root
                    if isinstance(actual_raw, m.ConfigMap)
                    else actual_raw
                )
                if isinstance(root_value, Mapping):
                    actual_data = {
                        str(k): _to_test_payload(v) for k, v in root_value.items()
                    }
            for key, expected_value in params.data.items():
                if key not in actual_data:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_DATA_KEY_MISSING.format(key=key),
                    )
                if actual_data[key] != expected_value:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_DATA_VALUE_MISMATCH.format(
                            key=key,
                            expected=expected_value,
                            actual=actual_data[key],
                        ),
                    )

        return err

    @staticmethod
    def that(
        value: object,
        **kwargs: object,
    ) -> None:
        r"""Super-powered universal value assertion - ALL validations in ONE method.

        This is the PRIMARY assertion method. All other assertion methods
        (eq, true, assert_contains, str_, is_, len, etc.) are convenience
        wrappers that delegate to this method.

        Supports unlimited depth for deep structural matching, comprehensive
        collection assertions, mapping validations, and custom predicates.

        Examples:
            # Basic assertions
            tm.that(x, eq=5)                    # x == 5
            tm.that(x, is_=str, len=(1, 50))    # is string, len 1-50
            tm.that(x, gt=0, lt=100)            # 0 < x < 100

            # String assertions
            tm.that(text, starts="Hello", ends="!", len=(5, 100))
            tm.that(email, match=r"^[\w.]+@[\w.]+$")

            # Sequence assertions
            tm.that(items, len=5, first="a", last="z", unique=True)
            tm.that(items, all_=str, sorted=True)
            tm.that(items, has=["required1", "required2"])

            # Mapping assertions
            tm.that(data, keys=["id", "name"], kv={"status": "active"})
            tm.that(config, attrs=["debug", "timeout"], attr_eq={"debug": True})

            # FlextResult in tm.that() (auto-detected)
            tm.that(result, ok=True, eq="expected")

            # Deep structural matching (unlimited depth)
            tm.that(response, deep={
                "user.name": "John",
                "user.profile.address.city": "NYC",
                "user.email": lambda e: "@" in e,
                "items": lambda i: len(i) > 0,
            })

            # Custom validation
            tm.that(user, where=lambda u: u.age >= 18 and u.verified)

        Args:
            value: Value to validate
            msg: Custom error message
            eq, ne: Equality/inequality
            is_, not_: Type checks - supports single type or tuple
            none: None check (True=must be None, False=must not be None)
            empty: Empty check (True=must be empty, False=must not be empty)
            gt, gte, lt, lte: Comparisons (numeric or length)
            len: Unified length spec - exact int or (min, max) tuple
            has: Unified containment - value contains item(s) (replaces contains)
            lacks: Unified non-containment - value does NOT contain item(s) (replaces excludes)
            starts, ends: String prefix/suffix
            match: Regex pattern (for strings)
            first, last: Sequence first/last item equals
            all_: All items match type or predicate
            any_: At least one item matches type or predicate
            sorted: Is sorted (True=ascending, or key function)
            unique: All items unique
            keys: Mapping has all keys
            lacks_keys: Mapping missing keys
            values: Mapping has all values
            kv: Key-value pairs (single tuple or mapping)
            attrs: Object has attribute(s)
            methods: Object has method(s)
            attr_eq: Attribute equals (single tuple or mapping)
            ok: For FlextResult: assert success
            error: For FlextResult: error contains
            deep: Deep structural matching specification
            where: Custom predicate function
            contains, excludes, length, length_gt, etc.: Legacy parameters (deprecated)

        Raises:
            AssertionError: If value doesn't satisfy constraints
            ValueError: If parameter validation fails (via Pydantic model)

        """
        raw_eq = kwargs.get("eq") if "eq" in kwargs else None
        raw_ne = kwargs.get("ne") if "ne" in kwargs else None

        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs accepts payload kwargs - Pydantic validates types
        try:
            params = m.Tests.Matcher.ThatParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            filtered_kwargs = {
                key: val for key, val in kwargs.items() if key not in {"eq", "ne"}
            }
            if filtered_kwargs == kwargs:
                raise ValueError(f"Parameter validation failed: {exc}") from exc
            try:
                params = m.Tests.Matcher.ThatParams.model_validate(filtered_kwargs)
            except (TypeError, ValueError, AttributeError) as filtered_exc:
                raise ValueError(
                    f"Parameter validation failed: {filtered_exc}"
                ) from filtered_exc

        # FlextResult auto-detection and handling
        if isinstance(value, r):
            result_obj: r[t.Tests.PayloadValue] = value
            if params.ok is not None:
                if params.ok and not result_obj.is_success:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_OK_FAILED.format(error=result_obj.error),
                    )
                if not params.ok and result_obj.is_success:
                    # Type narrowing: value.is_success is True, so .value returns the actual value
                    # Convert to string for format() - no cast needed, str() accepts any object
                    unwrapped_value_error = result_obj.value
                    value_str: str = str(unwrapped_value_error)
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_FAIL_EXPECTED.format(
                            value=value_str,
                        ),
                    )
            # Legacy error parameter already converted to has by model validator
            # Use params.has instead of params.error
            # Unwrap FlextResult for further validation
            # value is r[TResult], we need to extract the actual value for validation
            actual_value: t.Tests.PayloadValue | str
            if result_obj.is_success:
                # Type narrowing: .value property returns the actual value
                # No cast needed - value.value is already the correct type
                unwrapped_value: t.Tests.PayloadValue = result_obj.value
                actual_value = unwrapped_value
            # If result is failure, check if we're validating the error
            # params.has (converted from error) means we want to validate the error message
            elif params.has is not None:
                err = result_obj.error or ""
                _check_has_lacks(err, params.has, None, params.msg, as_str=True)
                actual_value = err
            elif params.ok is None:
                # If result is failure and no ok/error checks, fail
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_OK_FAILED.format(error=result_obj.error),
                )
            else:
                # params.ok is False, which means we expect failure - continue validation
                actual_value = result_obj.error or ""

            # Use actual_value for all further validations
            value = actual_value

        # Apply basic validations via u.chk() - pass parameters directly for type safety
        # Note: u.chk() doesn't support tuple types for is_/not_, handle separately
        has_validation = (
            (raw_eq is not None)
            or (raw_ne is not None)
            or params.eq is not None
            or params.ne is not None
            or params.gt is not None
            or params.gte is not None
            or params.lt is not None
            or params.lte is not None
            or params.none is not None
            or params.empty is not None
            or params.starts is not None
            or params.ends is not None
            or params.match is not None
        )
        if has_validation:
            eq_value = raw_eq if "eq" in kwargs else params.eq
            ne_value = raw_ne if "ne" in kwargs else params.ne
            if not u.chk(
                _as_guard_input(value),
                eq=_as_guard_input(eq_value) if eq_value is not None else None,
                ne=_as_guard_input(ne_value) if ne_value is not None else None,
                gt=params.gt,
                gte=params.gte,
                lt=params.lt,
                lte=params.lte,
                is_=None,
                not_=None,
                none=params.none,
                empty=params.empty,
                starts=params.starts,
                ends=params.ends,
                match=params.match,
            ):
                error_msg = (
                    params.msg
                    or f"Assertion failed: {value!r} did not satisfy constraints"
                )
                raise AssertionError(error_msg)
        if (
            params.is_ is not None
            and not isinstance(params.is_, tuple)
            and not isinstance(value, params.is_)
        ):
            raise AssertionError(
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=params.is_, actual=type(value).__name__)}",
            )
        if (
            params.not_ is not None
            and not isinstance(params.not_, tuple)
            and isinstance(value, params.not_)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=f"not {params.not_}",
                    actual=type(value).__name__,
                ),
            )
        # Handle tuple types separately
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not (
                type(value) in params.is_
                or (
                    hasattr(type(value), "__mro__")
                    and any(t in type(value).__mro__ for t in params.is_)
                )
            )
        ):
            raise AssertionError(
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=params.is_, actual=type(value).__name__)}",
            )
        if (
            params.not_ is not None
            and isinstance(params.not_, tuple)
            and (
                type(value) in params.not_
                or (
                    hasattr(type(value), "__mro__")
                    and any(t in type(value).__mro__ for t in params.not_)
                )
            )
        ):
            error_msg = (
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=f'not {params.not_}', actual=type(value).__name__)}"
            )
            raise AssertionError(error_msg)

        _check_has_lacks(value, params.has, params.lacks, params.msg)
        # Length validation (delegate to u.Tests.Length)
        # model_validator already converts legacy length_* params to unified len
        value_payload = _to_test_payload(value)
        if params.len is not None and not u.Tests.Length.validate(
            value_payload, params.len
        ):
            # Type guard: value has __len__ if it passed validation
            # Type narrow for __len__
            # Type narrow for __len__
            actual_len = len(value) if isinstance(value, Sized) else 0
            if isinstance(params.len, int):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_LEN_EXACT_FAILED.format(
                        expected=params.len,
                        actual=actual_len,
                    ),
                )
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_LEN_RANGE_FAILED.format(
                    min=params.len[0],
                    max=params.len[1],
                    actual=actual_len,
                ),
            )

        # Sequence assertions
        if isinstance(value, (list, tuple)):
            seq_value: Sequence[object] = value
            if params.first is not None:
                if not seq_value:
                    raise AssertionError(
                        params.msg or "Sequence is empty, cannot check first",
                    )
                if seq_value[0] != params.first:
                    raise AssertionError(
                        params.msg
                        or f"First item: expected {params.first!r}, got {seq_value[0]!r}",
                    )

            if params.last is not None:
                if not seq_value:
                    raise AssertionError(
                        params.msg or "Sequence is empty, cannot check last",
                    )
                if seq_value[-1] != params.last:
                    raise AssertionError(
                        params.msg
                        or f"Last item: expected {params.last!r}, got {seq_value[-1]!r}",
                    )

            if params.all_ is not None:
                if isinstance(params.all_, type):

                    def _all_match(t: type, seq: Sequence[object]) -> bool:
                        return all(
                            isinstance(x, t)
                            or (hasattr(type(x), "__mro__") and t in type(x).__mro__)
                            for x in seq
                        )

                    if not _all_match(params.all_, seq_value):
                        failed_idx = next(
                            (
                                i
                                for i, item in enumerate(list(seq_value))
                                if not (
                                    isinstance(item, params.all_)
                                    or (
                                        hasattr(type(item), "__mro__")
                                        and params.all_ in type(item).__mro__
                                    )
                                )
                            ),
                            None,
                        )
                        raise AssertionError(
                            params.msg
                            or c.Tests.Matcher.ERR_ALL_ITEMS_FAILED.format(
                                index=failed_idx,
                            ),
                        )
                elif callable(params.all_) and not all(
                    params.all_(_to_test_payload(item)) for item in seq_value
                ):
                    failed_idx = next(
                        (
                            i
                            for i, item in enumerate(list(seq_value))
                            if not params.all_(_to_test_payload(item))
                        ),
                        None,
                    )
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ALL_ITEMS_FAILED.format(
                            index=failed_idx,
                        ),
                    )

            if params.any_ is not None:
                if isinstance(params.any_, type):
                    any_type = params.any_
                    if not any(
                        isinstance(item, any_type)
                        or (
                            hasattr(type(item), "__mro__")
                            and any_type in type(item).__mro__
                        )
                        for item in seq_value
                    ):
                        raise AssertionError(
                            params.msg or c.Tests.Matcher.ERR_ANY_ITEMS_FAILED,
                        )
                elif callable(params.any_) and not any(
                    params.any_(_to_test_payload(item)) for item in seq_value
                ):
                    raise AssertionError(
                        params.msg or c.Tests.Matcher.ERR_ANY_ITEMS_FAILED,
                    )

            sorted_param = params.sorted
            if sorted_param is not None:
                value_list = list(seq_value)
                if sorted_param is True:
                    # sorted() requires SupportsRichComparison - use str key for any object
                    sorted_list = sorted(
                        value_list,
                        key=lambda x: (type(x).__name__, str(x)),
                    )
                    if value_list != sorted_list:
                        raise AssertionError(params.msg or "Sequence is not sorted")
                elif callable(sorted_param):
                    # callable() builtin narrows type for pyrefly/mypy
                    # Wrap user key function to return comparable string representation
                    # sorted_param is Callable[[object], object] but sorted needs comparable return
                    user_key_fn = sorted_param

                    def comparable_key(x: object) -> tuple[str, str]:
                        """Wrap user key to return comparable tuple."""
                        result = user_key_fn(_to_test_payload(x))
                        type_name = type(result).__name__
                        return (str(type_name), str(result))

                    sorted_list = sorted(value_list, key=comparable_key)
                    if value_list != sorted_list:
                        raise AssertionError(
                            params.msg or "Sequence is not sorted by key function",
                        )

            if params.unique is not None and params.unique:
                # Type guard: seq_value is list|tuple, so it has __len__
                value_len = len(seq_value)
                value_set_len = len(set(seq_value))
                if value_len != value_set_len:
                    raise AssertionError(
                        params.msg or "Sequence contains duplicate items",
                    )

        # Mapping assertions
        if isinstance(value, Mapping):
            mapping_value = value
            if params.keys is not None:
                key_set: set[object] = set(params.keys)
                missing = key_set - set(mapping_value.keys())
                if missing:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_KEYS_MISSING.format(keys=list(missing)),
                    )

            if params.lacks_keys is not None:
                lacks_key_set: set[object] = set(params.lacks_keys)
                present = lacks_key_set & set(mapping_value.keys())
                if present:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_KEYS_EXTRA.format(keys=list(present)),
                    )

            if params.values is not None:
                value_list = list(mapping_value.values())
                for expected_val in params.values:
                    if expected_val not in value_list:
                        raise AssertionError(
                            params.msg
                            or f"Expected value {expected_val!r} not found in mapping",
                        )

            if params.kv is not None:
                # KeyValueSpec is tuple[str, object] | Mapping[str, object]
                if isinstance(params.kv, tuple) and len(params.kv) == 2:
                    key, expected_val = params.kv
                    if key not in mapping_value:
                        raise AssertionError(
                            params.msg or f"Key {key!r} not found in mapping",
                        )
                    if mapping_value[key] != expected_val:
                        raise AssertionError(
                            params.msg
                            or f"Key {key!r}: expected {expected_val!r}, got {mapping_value[key]!r}",
                        )
                elif hasattr(params.kv, "keys") and hasattr(params.kv, "items"):
                    mapping_kv: Mapping[str, object] = params.kv
                    for key, expected_obj in mapping_kv.items():
                        if key not in mapping_value:
                            raise AssertionError(
                                params.msg or f"Key {key!r} not found in mapping",
                            )
                        if mapping_value[key] != expected_obj:
                            raise AssertionError(
                                params.msg
                                or f"Key {key!r}: expected {expected_obj!r}, got {mapping_value[key]!r}",
                            )

        # Object/Class assertions
        if params.attrs is not None:
            if isinstance(params.attrs, str):
                attr_list: list[str] = [params.attrs]
            else:
                attr_list = list(params.attrs)
            for attr in attr_list:
                if not hasattr(value, attr):
                    raise AssertionError(
                        params.msg or f"Object missing attribute: {attr}",
                    )

        if params.methods is not None:
            if isinstance(params.methods, str):
                method_list: list[str] = [params.methods]
            else:
                method_list = list(params.methods)
            for method in method_list:
                if not hasattr(value, method):
                    raise AssertionError(
                        params.msg or f"Object missing method: {method}",
                    )
                if not callable(getattr(value, method)):
                    raise AssertionError(
                        params.msg or f"Object attribute {method} is not callable",
                    )

        if params.attr_eq is not None:
            if isinstance(params.attr_eq, tuple) and len(params.attr_eq) == 2:
                attr, expected_val = params.attr_eq
                if not hasattr(value, attr):
                    raise AssertionError(
                        params.msg or f"Object missing attribute: {attr}",
                    )
                actual_val = getattr(value, attr)
                if actual_val != expected_val:
                    raise AssertionError(
                        params.msg
                        or f"Attribute {attr}: expected {expected_val!r}, got {actual_val!r}",
                    )
            elif u.is_type(params.attr_eq, "mapping"):
                for attr, expected_val in params.attr_eq.items():
                    if not hasattr(value, attr):
                        raise AssertionError(
                            params.msg or f"Object missing attribute: {attr}",
                        )
                    actual_val = getattr(value, attr)
                    if actual_val != expected_val:
                        raise AssertionError(
                            params.msg
                            or f"Attribute {attr}: expected {expected_val!r}, got {actual_val!r}",
                        )

        # Deep matching (delegate to u.Tests.DeepMatch)
        if params.deep is not None:
            if not (
                (hasattr(type(value), "__mro__") and BaseModel in type(value).__mro__)
                or (hasattr(value, "keys") and hasattr(value, "items"))
            ):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(value).__name__}",
                )
            deep_value: BaseModel | Mapping[str, t.Tests.PayloadValue]
            if isinstance(value, BaseModel):
                deep_value = value
            elif isinstance(value, Mapping):
                deep_value = {str(k): _to_test_payload(v) for k, v in value.items()}
            else:
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(value).__name__}",
                )
            match_result = u.Tests.DeepMatch.match(deep_value, params.deep)
            if not match_result.matched:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_DEEP_PATH_FAILED.format(
                        path=match_result.path,
                        reason=match_result.reason,
                    ),
                )

        # Custom predicate
        if params.where is not None and not params.where(_to_test_payload(value)):
            raise AssertionError(
                params.msg or c.Tests.Matcher.ERR_PREDICATE_FAILED.format(value=value),
            )

    @staticmethod
    def check[TResult](result: r[TResult]) -> m.Tests.Matcher.Chain[TResult]:
        """Start chained assertions on result (railway pattern)."""
        return m.Tests.Matcher.Chain[TResult](result=result)

    @staticmethod
    @contextmanager
    def scope(
        **kwargs: t.Tests.PayloadValue,
    ) -> Iterator[m.Tests.Matcher.TestScope]:
        """Enhanced isolated test execution scope.

        Uses Pydantic 2 model (ScopeParams) for parameter validation and computation.
        All parameters are validated automatically via u.Model.from_kwargs.

        Provides isolated configuration, container, and context for tests.
        Supports temporary environment variables, working directory changes,
        and automatic cleanup functions.

        Args:
            **kwargs: Parameters validated via m.Tests.Matcher.ScopeParams model
                - config: Initial configuration values
                - container: Initial container/service mappings
                - context: Initial context values
                - cleanup: Sequence of cleanup functions to call on exit
                - env: Temporary environment variables (restored on exit)
                - cwd: Temporary working directory (restored on exit)

        Yields:
            TestScope with config, container, and context dicts

        Examples:
            with tm.scope() as s:
                s.container["service"] = mock_service
                result = operation()
                tm.ok(result)

            with tm.scope(config={"debug": True}, env={"API_KEY": "test"}) as s:
                # Test with specific config and env vars
                pass

            with tm.scope(cleanup=[lambda: cleanup_resource()]) as s:
                # Auto-cleanup on exit
                pass

        Raises:
            ValueError: If parameter validation fails (via Pydantic model)

        """
        # Create and validate parameters using Pydantic 2 model
        # u.Model.from_kwargs accepts payload kwargs - Pydantic validates types
        try:
            params = m.Tests.Matcher.ScopeParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc

        # Save original environment and working directory
        original_env: dict[str, str | None] = {}
        original_cwd: Path | None = None

        try:
            # Set temporary environment variables
            if params.env is not None:
                for key, value in params.env.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value

            # Change working directory
            if params.cwd is not None:
                original_cwd = Path.cwd()
                cwd_path = (
                    Path(params.cwd) if u.is_type(params.cwd, "str") else params.cwd
                )
                os.chdir(cwd_path)

            # Create scope - use dict[str, t.Tests.PayloadValue] to match TestScope field types
            cfg: dict[str, t.Tests.PayloadValue] = {}
            if params.config:
                cfg = {str(key): value for key, value in params.config.items()}
            # Filter container to payload values - services may be arbitrary objects
            container_dict: dict[str, t.Tests.PayloadValue] = {
                k: v
                for k, v in (params.container or {}).items()
                if t.Guards.is_general_value(v)
            }
            context_map: dict[str, t.Tests.PayloadValue] = {}
            if params.context:
                context_map = {str(key): value for key, value in params.context.items()}

            scope = m.Tests.Matcher.TestScope.model_validate({
                "config": cfg,
                "container": container_dict,
                "context": context_map,
            })
            yield scope

        finally:
            # Restore environment variables
            if params.env is not None:
                for key, original_value in original_env.items():
                    if original_value is None:
                        _ = os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

            # Restore working directory
            if original_cwd is not None:
                os.chdir(original_cwd)

            # Run cleanup functions
            if params.cleanup is not None:
                for cleanup_func in params.cleanup:
                    try:
                        cleanup_func()
                    except (
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        AttributeError,
                    ) as e:
                        # Log but don't fail on cleanup errors
                        warnings.warn(
                            c.Tests.Matcher.ERR_SCOPE_CLEANUP_FAILED.format(
                                error=str(e),
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )

    @staticmethod
    def assert_length_equals(
        value: object,
        expected: int,
        msg: str | None = None,
    ) -> None:
        """Assert value length equals expected.

        Args:
            value: Value to check length of (must have __len__)
            expected: Expected length
            msg: Optional custom error message

        Raises:
            AssertionError: If length doesn't match

        """
        FlextTestsMatchers.that(value, length=expected, msg=msg)

    @staticmethod
    def assert_length_greater_than(
        value: object,
        min_length: int,
        msg: str | None = None,
    ) -> None:
        """Assert value length is greater than min_length.

        Args:
            value: Value to check length of (must have __len__)
            min_length: Minimum expected length
            msg: Optional custom error message

        Raises:
            AssertionError: If length is not greater than min_length

        """
        FlextTestsMatchers.that(value, length_gt=min_length, msg=msg)

    @staticmethod
    def assert_result_success[TResult](
        result: r[TResult],
        msg: str | None = None,
    ) -> TResult:
        """Assert result is success and return unwrapped value.

        Args:
            result: FlextResult to check
            msg: Optional custom error message

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure

        """
        # Direct implementation to preserve exact TResult type
        if not result.is_success:
            error_msg = msg or f"Expected success but got failure: {result.error}"
            raise AssertionError(error_msg)
        return result.value


tm = FlextTestsMatchers

__all__ = ["FlextTestsMatchers", "tm"]
