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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

from flext_core import r
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.typings import t
from flext_tests.utilities import u


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
        **kwargs: t.GeneralValueType,
    ) -> TResult:
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
                - is_: Type check (isinstance) - supports single type or tuple
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
        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs accepts **kwargs: GeneralValueType - Pydantic validates types
        params_result = u.Model.from_kwargs(m.Tests.Matcher.OkParams, **kwargs)
        if params_result.is_failure:
            raise ValueError(f"Parameter validation failed: {params_result.error}")
        params = params_result.value

        if not result.is_success:
            raise AssertionError(
                params.msg or c.Tests.Matcher.ERR_OK_FAILED.format(error=result.error),
            )
        # Start with TResult, may be reassigned to extracted value (t.GeneralValueType)
        result_value: TResult | t.GeneralValueType = result.value

        # Path extraction first (if specified)
        if params.path is not None:
            # u.Mapper.extract expects str, not PathSpec
            # Use isinstance for proper type narrowing
            if isinstance(params.path, str):
                path_str: str = params.path
            else:
                path_str = ".".join(params.path)
            # Mapper.extract requires BaseModel or Mapping - type narrow first
            if not isinstance(result_value, (BaseModel, Mapping)):
                raise AssertionError(
                    params.msg
                    or f"Path extraction requires dict or model, got {type(result_value).__name__}",
                )
            extracted = u.Mapper.extract(result_value, path_str)
            if extracted.is_failure:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_SCOPE_PATH_NOT_FOUND.format(
                        path=path_str,
                        error=extracted.error,
                    ),
                )
            # Reassign to extracted value - now type is t.GeneralValueType
            result_value = extracted.value

        # Validate value with u.chk() - pass parameters directly for type safety
        # Note: u.chk() doesn't support tuple types for is_/not_, handle separately
        has_validation = (
            params.eq is not None
            or params.ne is not None
            or (params.is_ is not None and not isinstance(params.is_, tuple))
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
                result_value,
                eq=params.eq,
                ne=params.ne,
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
        # Handle tuple types separately (isinstance supports tuples)
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not isinstance(result_value, params.is_)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=params.is_,
                    actual=type(result_value).__name__,
                ),
            )

        # Handle unified has/lacks (works for str, list, dict, set, tuple)
        # params.has is t.Tests.Matcher.ContainmentSpec | None
        # ContainmentSpec = object | Sequence[object]
        if params.has is not None:
            # Type narrowing: params.has is object | Sequence[object]
            has_value: object | Sequence[object] = params.has
            # Use isinstance for proper type narrowing
            if isinstance(has_value, (list, tuple)):
                # Type narrowing: has_value is already list | tuple which is Sequence[object]
                has_items: Sequence[object] = has_value
            else:
                # Single item - wrap in list for uniform processing
                has_item: object = has_value
                # list[object] is already Sequence[object], no cast needed
                has_items = [has_item]
            for item in has_items:
                # Type narrowing: item is object from Sequence[object]
                item_typed: object = item
                if not u.chk(result_value, contains=item_typed):
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=result_value,
                            item=item_typed,
                        ),
                    )

        if params.lacks is not None:
            # Use isinstance for proper type narrowing (pyrefly requires it)
            if isinstance(params.lacks, (list, tuple)):
                lacks_items: Sequence[object] = params.lacks
            else:
                # Single item - wrap in list for uniform processing
                lacks_item: object = params.lacks
                # list[object] is already Sequence[object], no cast needed
                lacks_items = [lacks_item]
            for item in lacks_items:
                if u.chk(result_value, contains=item):
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=result_value,
                            item=item,
                        ),
                    )

        # Length validation (delegate to u.Tests.Length)
        if params.len is not None and not u.Tests.Length.validate(
            result_value,
            params.len,
        ):
            # Type guard: result_value has __len__ if it passed validation
            # Use isinstance to help type checker understand result_value has __len__
            if isinstance(result_value, (str, bytes, Sequence, Mapping)):
                actual_len = len(result_value)
            else:
                actual_len = 0
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
            # Type narrow for DeepMatch.match which requires BaseModel or Mapping
            if not isinstance(result_value, (BaseModel, Mapping)):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(result_value).__name__}",
                )
            match_result = u.Tests.DeepMatch.match(result_value, params.deep)
            if not match_result.matched:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_DEEP_PATH_FAILED.format(
                        path=match_result.path,
                        reason=match_result.reason,
                    ),
                )

        # Custom predicate
        if params.where is not None and not params.where(result_value):
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
        # If path extraction was used, this is the extracted GeneralValueType
        # Both paths are validated at this point
        if params.path is None:
            # No path extraction - return original result.value (TResult)
            return result.value
        # Path extraction case - result_value is GeneralValueType
        # Caller expects TResult but path extraction changes the type
        # Return validated value, caller must handle type appropriately
        validated_result: TResult = result.value  # Return original TResult
        return validated_result

    @staticmethod
    def fail[TResult](
        result: r[TResult],
        **kwargs: t.GeneralValueType,
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
        # u.Model.from_kwargs accepts **kwargs: GeneralValueType - Pydantic validates types
        # Legacy 'error' parameter is handled by FailParams model validator
        params_result = u.Model.from_kwargs(m.Tests.Matcher.FailParams, **kwargs)
        if params_result.is_failure:
            raise ValueError(f"Parameter validation failed: {params_result.error}")
        params = params_result.value

        if result.is_success:
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_FAIL_EXPECTED.format(value=result.value),
            )
        err = result.error or ""

        # Apply error message validation if any check parameters provided
        # Legacy parameters (error, contains, excludes) already converted to has/lacks by model validator
        if params.has or params.lacks or params.starts or params.ends or params.match:
            # Handle unified has/lacks
            if params.has is not None:
                # ExclusionSpec is str | Sequence[str] - need to handle both cases
                # Convert to list[str] for uniform processing
                has_value = params.has
                if isinstance(has_value, str):
                    items_has: list[str] = [has_value]
                else:
                    # Sequence[str] case - convert to list
                    items_has = [str(x) for x in has_value]
                for item in items_has:
                    if not u.chk(err, contains=item):
                        raise AssertionError(
                            params.msg
                            or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                                container=err,
                                item=item,
                            ),
                        )

            if params.lacks is not None:
                # ExclusionSpec is str | Sequence[str] - need to handle both cases
                # Convert to list[str] for uniform processing
                lacks_value = params.lacks
                if isinstance(lacks_value, str):
                    items_lacks: list[str] = [lacks_value]
                else:
                    # Sequence[str] case - convert to list
                    items_lacks = [str(x) for x in lacks_value]
                for item in items_lacks:
                    if u.chk(err, contains=item):
                        raise AssertionError(
                            params.msg
                            or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                                container=err,
                                item=item,
                            ),
                        )

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
            actual_data = result.error_data or {}
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
        **kwargs: t.GeneralValueType,
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
            is_, not_: Type checks (isinstance) - supports single type or tuple
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
        # Convert kwargs to validated model using FlextUtilities
        # u.Model.from_kwargs accepts **kwargs: GeneralValueType - Pydantic validates types
        params_result = u.Model.from_kwargs(m.Tests.Matcher.ThatParams, **kwargs)
        if params_result.is_failure:
            raise ValueError(f"Parameter validation failed: {params_result.error}")
        params = params_result.value

        # FlextResult auto-detection and handling
        if isinstance(value, r):
            if params.ok is not None:
                if params.ok and not value.is_success:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_OK_FAILED.format(error=value.error),
                    )
                if not params.ok and value.is_success:
                    # Type narrowing: value.is_success is True, so .value returns the actual value
                    # Convert to string for format() - no cast needed, str() accepts any object
                    unwrapped_value_error = value.value
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
            actual_value: object
            if value.is_success:
                # Type narrowing: .value property returns the actual value
                # No cast needed - value.value is already the correct type
                unwrapped_value: t.GeneralValueType = value.value
                actual_value = unwrapped_value
            # If result is failure, check if we're validating the error
            # params.has (converted from error) means we want to validate the error message
            elif params.has is not None:
                # Validate error message using has parameter
                err = value.error or ""
                # params.has is t.Tests.Matcher.ContainmentSpec | None
                # ContainmentSpec = object | Sequence[object]
                # For error validation, convert to list of strings for checking
                # Build list of strings directly to avoid type narrowing issues
                error_has_items: list[str]
                if isinstance(params.has, str):
                    error_has_items = [params.has]
                elif isinstance(params.has, Sequence):
                    # Convert all items to strings
                    error_has_items = [str(x) for x in params.has]
                else:
                    # Convert single object to string
                    error_has_items = [str(params.has)]
                for item in error_has_items:
                    if item not in err:
                        raise AssertionError(
                            params.msg
                            or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                                container=err,
                                item=item,
                            ),
                        )
                # Error validated, use error string for further validation
                actual_value = err
            elif params.ok is None:
                # If result is failure and no ok/error checks, fail
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_OK_FAILED.format(error=value.error),
                )
            else:
                # params.ok is False, which means we expect failure - continue validation
                actual_value = value.error or ""

            # Use actual_value for all further validations
            value = actual_value

        # Apply basic validations via u.chk() - pass parameters directly for type safety
        # Note: u.chk() doesn't support tuple types for is_/not_, handle separately
        has_validation = (
            params.eq is not None
            or params.ne is not None
            or (params.is_ is not None and not isinstance(params.is_, tuple))
            or (params.not_ is not None and not isinstance(params.not_, tuple))
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
            # u.chk() only accepts single type, not tuple
            is_type = params.is_ if not isinstance(params.is_, tuple) else None
            not_type = params.not_ if not isinstance(params.not_, tuple) else None
            if not u.chk(
                value,
                eq=params.eq,
                ne=params.ne,
                gt=params.gt,
                gte=params.gte,
                lt=params.lt,
                lte=params.lte,
                is_=is_type,
                not_=not_type,
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
        # Handle tuple types separately (isinstance supports tuples)
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not isinstance(value, params.is_)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=params.is_,
                    actual=type(value).__name__,
                ),
            )
        if (
            params.not_ is not None
            and isinstance(params.not_, tuple)
            and isinstance(value, params.not_)
        ):
            error_msg = params.msg or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                expected=f"not {params.not_}",
                actual=type(value).__name__,
            )
            raise AssertionError(error_msg)

        # Handle unified has/lacks (works for str, list, dict, set, tuple)
        # params.has is t.Tests.Matcher.ContainmentSpec | None
        # ContainmentSpec = object | Sequence[object]
        if params.has is not None:
            # Type narrowing: params.has is object | Sequence[object]
            has_value: object | Sequence[object] = params.has
            # Use isinstance for proper type narrowing
            if isinstance(has_value, (list, tuple)):
                # Type narrowing: has_value is already list | tuple which is Sequence[object]
                has_items: Sequence[object] = has_value
            else:
                # Single item - wrap in list for uniform processing
                has_item: object = has_value
                # list[object] is already Sequence[object], no cast needed
                has_items = [has_item]
            for has_item_obj in has_items:
                # Type narrowing: has_item_obj is object from Sequence[object]
                has_item_val: object = has_item_obj
                if not u.chk(value, contains=has_item_val):
                    # str() already returns str
                    item_str: str = str(has_item_val)
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value,
                            item=item_str,
                        ),
                    )

        if params.lacks is not None:
            # Use isinstance for proper type narrowing (pyrefly requires it)
            if isinstance(params.lacks, (list, tuple)):
                lacks_items: Sequence[object] = params.lacks
            else:
                # Single item - wrap in list for uniform processing
                lacks_item: object = params.lacks
                # list[object] is already Sequence[object], no cast needed
                lacks_items = [lacks_item]
            for lacks_item_obj in lacks_items:
                lacks_item_val: object = lacks_item_obj
                if u.chk(value, contains=lacks_item_val):
                    # str() already returns str, so cast is redundant
                    lacks_item_str: str = str(lacks_item_val)
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value,
                            item=lacks_item_str,
                        ),
                    )

        # Legacy excludes support (deprecated)
        # Legacy excludes parameter already converted to lacks by model validator
        # Use params.lacks instead of params.excludes

        # Length validation (delegate to u.Tests.Length)
        # model_validator already converts legacy length_* params to unified len
        if params.len is not None and not u.Tests.Length.validate(value, params.len):
            # Type guard: value has __len__ if it passed validation
            # Use isinstance to help type checker understand value has __len__
            # Type narrowing for len() call
            if isinstance(value, (str, bytes, Sequence, Mapping)):
                # Type narrowing: value is already narrowed by isinstance to have __len__
                # No cast needed - isinstance narrows the type automatically
                actual_len = len(value)
            else:
                actual_len = 0
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
        # Use isinstance for proper type narrowing (pyrefly requires it)
        if isinstance(value, (list, tuple)):
            # Type narrowing: value is already narrowed to list | tuple by isinstance
            # No cast needed - isinstance already provides type narrowing
            seq_value: list[object] | tuple[object, ...] = value
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
                    if not all(isinstance(item, params.all_) for item in seq_value):
                        failed_idx = next(
                            (
                                i
                                for i, item in enumerate(seq_value)
                                if not isinstance(item, params.all_)
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
                    params.all_(item) for item in seq_value
                ):
                    failed_idx = next(
                        (
                            i
                            for i, item in enumerate(seq_value)
                            if not params.all_(item)
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
                    if not any(isinstance(item, params.any_) for item in seq_value):
                        raise AssertionError(
                            params.msg or c.Tests.Matcher.ERR_ANY_ITEMS_FAILED,
                        )
                elif callable(params.any_) and not any(
                    params.any_(item) for item in seq_value
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
                    user_key_fn: Callable[[object], object] = sorted_param

                    def comparable_key(x: object) -> tuple[str, str]:
                        """Wrap user key to return comparable tuple."""
                        result = user_key_fn(x)
                        return (type(result).__name__, str(result))

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
            # Type narrowing: value is already narrowed to Mapping by isinstance
            # No cast needed - isinstance already provides type narrowing
            mapping_value: Mapping[object, object] = value
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
                # Handle tuple case first (single key-value pair)
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
                # Handle Mapping case (multiple key-value pairs)
                elif isinstance(params.kv, Mapping):
                    # Type narrowing: params.kv is Mapping[str, object]
                    mapping_kv: Mapping[str, object] = params.kv
                    for key, expected_val in mapping_kv.items():
                        if key not in mapping_value:
                            raise AssertionError(
                                params.msg or f"Key {key!r} not found in mapping",
                            )
                        if mapping_value[key] != expected_val:
                            raise AssertionError(
                                params.msg
                                or f"Key {key!r}: expected {expected_val!r}, got {mapping_value[key]!r}",
                            )

        # Object/Class assertions
        if params.attrs is not None:
            # Type narrowing: create list[str] for proper hasattr typing
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
            # Type narrowing: create list[str] for proper hasattr typing
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
            # Type narrow for DeepMatch.match which requires BaseModel or Mapping
            if not isinstance(value, (BaseModel, Mapping)):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(value).__name__}",
                )
            # Explicit type binding after isinstance check for pyrefly
            deep_value: BaseModel | Mapping[str, t.GeneralValueType] = value
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
        if params.where is not None and not params.where(value):
            raise AssertionError(
                params.msg or c.Tests.Matcher.ERR_PREDICATE_FAILED.format(value=value),
            )

    @staticmethod
    def check[TResult](result: r[TResult]) -> m.Tests.Matcher.Chain:
        """Start chained assertions on result (railway pattern)."""
        # Chain expects r[t.GeneralValueType], TResult is compatible at runtime
        # FlextResult[TResult] is covariant with FlextResult[t.GeneralValueType]
        # No cast needed - generic type parameters are compatible
        return m.Tests.Matcher.Chain(result=result)

    # =========================================================================
    # NEW GENERALIST METHODS
    # =========================================================================

    @staticmethod
    def assert_contains[TItem](
        container: Mapping[object, object] | Sequence[TItem] | str,
        item: TItem | str,
        *,
        msg: str | None = None,
    ) -> None:
        """DEPRECATED: Use tm.that(container, contains=item) instead.

        Migration:
            tm.assert_contains(data, "key") -> tm.that(data, contains="key")
            tm.assert_contains(items, value) -> tm.that(items, contains=value)
            tm.assert_contains(text, "sub") -> tm.that(text, contains="sub")

        """
        warnings.warn(
            "assert_contains() is deprecated. Use tm.that(container, contains=item) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = FlextTestsMatchers.that(container, contains=item, msg=msg)

    @staticmethod
    def str_(
        text: str,
        contains: str | None = None,
        *,
        starts: str | None = None,
        ends: str | None = None,
        match: str | None = None,
        excludes: str | None = None,
        empty: bool | None = None,
        msg: str | None = None,
    ) -> None:
        """DEPRECATED: Use tm.that(text, contains/starts/ends/match/excludes/empty=...) instead.

        Migration:
            tm.str_(url, starts="http", ends="/") -> tm.that(url, starts="http", ends="/")
            tm.str_(text, contains="success") -> tm.that(text, contains="success")
            tm.str_(text, match="[0-9]{4}-[0-9]{2}") -> tm.that(text, match="[0-9]{4}-[0-9]{2}")
            tm.str_(text, excludes="error") -> tm.that(text, excludes="error")
            tm.str_(name, empty=False) -> tm.that(name, empty=False)

        """
        warnings.warn(
            "str_() is deprecated. Use tm.that(text, contains/starts/ends/match/excludes/empty=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        FlextTestsMatchers.that(
            text,
            contains=contains,
            starts=starts,
            ends=ends,
            match=match,
            excludes=excludes,
            empty=empty,
            msg=msg,
        )

    @staticmethod
    def is_(
        value: object,
        expected_type: type[object] | None = None,
        *,
        none: bool | None = None,
        msg: str | None = None,
    ) -> None:
        """DEPRECATED: Use tm.that(value, is_=type, none=...) instead.

        Migration:
            tm.is_(config, FlextSettings) -> tm.that(config, is_=FlextSettings)
            tm.is_(value, none=False) -> tm.that(value, none=False)
            tm.is_(value, none=True) -> tm.that(value, none=True)
            tm.is_(value, str, none=False) -> tm.that(value, is_=str, none=False)

        """
        warnings.warn(
            "is_() is deprecated. Use tm.that(value, is_=type, none=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        FlextTestsMatchers.that(value, is_=expected_type, none=none, msg=msg)

    @staticmethod
    @contextmanager
    def scope(
        **kwargs: t.GeneralValueType,
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
        # u.Model.from_kwargs accepts **kwargs: GeneralValueType - Pydantic validates types
        params_result = u.Model.from_kwargs(m.Tests.Matcher.ScopeParams, **kwargs)
        if params_result.is_failure:
            raise ValueError(f"Parameter validation failed: {params_result.error}")
        params = params_result.value

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

            # Create scope
            cfg: t.ConfigurationDict = dict(params.config) if params.config else {}
            scope = m.Tests.Matcher.TestScope(
                config=cfg,
                container=dict(params.container or {}),
                context=dict(params.context or {}),
            )
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
                    except Exception as e:
                        # Log but don't fail on cleanup errors
                        warnings.warn(
                            c.Tests.Matcher.ERR_SCOPE_CLEANUP_FAILED.format(
                                error=str(e),
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )

    # =========================================================================
    # SPECIALIZED ASSERTIONS (keep for backward compatibility)
    # =========================================================================

    @staticmethod
    def dict_[TK, TV](
        data: Mapping[TK, TV],
        *,
        has_key: TK | Sequence[TK] | None = None,
        not_has_key: TK | Sequence[TK] | None = None,
        has_value: TV | Sequence[TV] | None = None,
        not_has_value: TV | Sequence[TV] | None = None,
        key_equals: tuple[TK, TV] | Sequence[tuple[TK, TV]] | None = None,
        contains: dict[TK, TV] | None = None,
        not_contains: dict[TK, TV] | None = None,
        empty: bool | None = None,
        length: int | None = None,
        length_gt: int | None = None,
        length_gte: int | None = None,
        length_lt: int | None = None,
        length_lte: int | None = None,
        msg: str | None = None,
    ) -> dict[TK, TV]:
        """DEPRECATED: Use tm.has() for key checks, tm.len() for length checks.

        Migration:
            tm.that(d, keys=["x"]) -> tm.has(d, "x")
            tm.dict_(d, length=5)   -> tm.that(d, length=5)
            tm.dict_(d, empty=False) -> tm.len(d, empty=False)

        This method remains for complex cases requiring key_equals, contains, etc.

        """
        warnings.warn(
            (
                "dict_() is deprecated. Use tm.has() for key checks, "
                "tm.len() for length. Kept for complex validations."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        if has_key is not None:
            keys_to_check = (
                [has_key] if not isinstance(has_key, Sequence) else list(has_key)
            )
            for key in keys_to_check:
                if key not in data:
                    raise AssertionError(msg or f"Key {key!r} not found in dict")

        if not_has_key is not None:
            keys_to_check = (
                [not_has_key]
                if not isinstance(not_has_key, Sequence)
                else list(not_has_key)
            )
            for key in keys_to_check:
                if key in data:
                    raise AssertionError(msg or f"Key {key!r} should not be in dict")

        if has_value is not None:
            values_to_check = (
                [has_value] if not isinstance(has_value, Sequence) else list(has_value)
            )
            for val in values_to_check:
                if val not in data.values():
                    raise AssertionError(msg or f"Value {val!r} not found in dict")

        if not_has_value is not None:
            values_to_check = (
                [not_has_value]
                if not isinstance(not_has_value, Sequence)
                else list(not_has_value)
            )
            for val in values_to_check:
                if val in data.values():
                    raise AssertionError(msg or f"Value {val!r} should not be in dict")

        if key_equals is not None:
            pairs_to_check: list[tuple[TK, TV]]
            # Check for single tuple first (tuple is a Sequence, so order matters)
            if isinstance(key_equals, tuple) and len(key_equals) == 2:
                # Single tuple: (key, value) - already tuple[TK, TV] from type annotation
                # No cast needed - type is guaranteed by function signature
                pairs_to_check = [key_equals]
            else:
                # Sequence of tuples - mypy understands Sequence[tuple[TK, TV]] from type annotation
                # Since tuple is a Sequence, this branch handles all non-single-tuple cases
                # Type annotation ensures items are tuple[TK, TV], no cast needed
                pairs_to_check = list(key_equals)
            for key, expected_value in pairs_to_check:
                if key not in data:
                    raise AssertionError(msg or f"Key {key!r} not found in dict")
                if data[key] != expected_value:
                    raise AssertionError(
                        msg
                        or f"Key '{key}': expected {expected_value}, got {data[key]}",
                    )

        if contains is not None:
            for key, val in contains.items():
                if key not in data:
                    raise AssertionError(msg or f"Key {key!r} not found in dict")
                if data.get(key) != val:
                    raise AssertionError(
                        msg or f"Key '{key}': expected {val}, got {data.get(key)}",
                    )

        if empty is not None:
            FlextTestsMatchers.that(data, msg=msg, empty=empty)

        if (
            length is not None
            or length_gt is not None
            or length_gte is not None
            or length_lt is not None
            or length_lte is not None
        ):
            # Convert Mapping to list for length check using tm.that()
            length_kwargs: dict[str, t.GeneralValueType] = {}
            if length is not None:
                length_kwargs["length"] = length
            if length_gt is not None:
                length_kwargs["length_gt"] = length_gt
            if length_gte is not None:
                length_kwargs["length_gte"] = length_gte
            if length_lt is not None:
                length_kwargs["length_lt"] = length_lt
            if length_lte is not None:
                length_kwargs["length_lte"] = length_lte
            _ = FlextTestsMatchers.that(list(data.keys()), msg=msg, **length_kwargs)

        return dict(data)

    @staticmethod
    def list_[TItem](
        items: Sequence[TItem],
        *,
        contains: TItem | None = None,
        not_contains: TItem | None = None,
        first_equals: TItem | None = None,
        last_equals: TItem | None = None,
        equals: Sequence[TItem] | None = None,
        empty: bool | None = None,
        length: int | None = None,
        length_gt: int | None = None,
        length_gte: int | None = None,
        length_lt: int | None = None,
        length_lte: int | None = None,
        all_match: Callable[[TItem], bool] | None = None,
        any_matches: Callable[[TItem], bool] | None = None,
        msg: str | None = None,
    ) -> list[TItem]:
        """DEPRECATED: Use tm.has() for containment, tm.len() for length checks.

        Migration:
            tm.that(l, has="x") -> tm.has(l, "x")
            tm.that(l, length=5)    -> tm.that(l, length=5)
            tm.list_(l, empty=False) -> tm.len(l, empty=False)

        This method remains for complex cases like all_match, any_matches, etc.

        """
        warnings.warn(
            (
                "list_() is deprecated. Use tm.has() for containment, "
                "tm.len() for length. Kept for predicates (all_match, any_matches)."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        if contains is not None:
            FlextTestsMatchers.that(items, msg=msg, contains=contains)

        if not_contains is not None and not_contains in items:
            raise AssertionError(msg or f"{not_contains} should not be in list")

        if first_equals is not None:
            FlextTestsMatchers.that(items, msg="List must not be empty", empty=False)
            FlextTestsMatchers.that(items[0], msg=msg, eq=first_equals)

        if last_equals is not None:
            FlextTestsMatchers.that(items, msg="List must not be empty", empty=False)
            FlextTestsMatchers.that(items[-1], msg=msg, eq=last_equals)

        if equals is not None:
            FlextTestsMatchers.that(list(items), msg=msg, eq=list(equals))

        if empty is not None:
            FlextTestsMatchers.that(items, msg=msg, empty=empty)

        if (
            length is not None
            or length_gt is not None
            or length_gte is not None
            or length_lt is not None
            or length_lte is not None
        ):
            # Use tm.that() directly instead of deprecated tm.len()
            length_kwargs: dict[str, t.GeneralValueType] = {}
            if length is not None:
                length_kwargs["length"] = length
            if length_gt is not None:
                length_kwargs["length_gt"] = length_gt
            if length_gte is not None:
                length_kwargs["length_gte"] = length_gte
            if length_lt is not None:
                length_kwargs["length_lt"] = length_lt
            if length_lte is not None:
                length_kwargs["length_lte"] = length_lte
            _ = FlextTestsMatchers.that(items, msg=msg, **length_kwargs)

        if all_match is not None and not all(all_match(item) for item in items):
            raise AssertionError(msg or "Not all items matched predicate")

        if any_matches is not None and not any(any_matches(item) for item in items):
            raise AssertionError(msg or "No item matched predicate")

        return list(items)

    @staticmethod
    def len(
        items: Sequence[object] | str | Mapping[object, object],
        expected: int | None = None,
        *,
        gt: int | None = None,
        gte: int | None = None,
        lt: int | None = None,
        lte: int | None = None,
        empty: bool | None = None,
        msg: str | None = None,
    ) -> None:
        """DEPRECATED: Use tm.that(items, length/length_gt/length_gte/empty=...) instead.

        Migration:
            tm.that(items, length=5) -> tm.that(items, length=5)
            tm.that(items, length_gt=0, lt=10) -> tm.that(items, length_gt=0, length_lt=10)
            tm.that(items, length_gte=1) -> tm.that(items, length_gte=1)
            tm.len(items, empty=False) -> tm.that(items, empty=False)

        """
        warnings.warn(
            "len() is deprecated. Use tm.that(items, length/length_gt/length_gte/empty=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        length_kwargs: dict[str, t.GeneralValueType] = {}
        if expected is not None:
            length_kwargs["length"] = expected
        if gt is not None:
            length_kwargs["length_gt"] = gt
        if gte is not None:
            length_kwargs["length_gte"] = gte
        if lt is not None:
            length_kwargs["length_lt"] = lt
        if lte is not None:
            length_kwargs["length_lte"] = lte
        if empty is not None:
            length_kwargs["empty"] = empty
        FlextTestsMatchers.that(items, msg=msg, **length_kwargs)

    @staticmethod
    def assert_is_type(
        value: object,
        type_spec: type[object],
        msg: str | None = None,
    ) -> object:
        """DEPRECATED: Use tm.is_(value, type, none=False) instead.

        Note: Renamed from is_type to avoid signature conflict with parent class.

        Migration:
            tm.assert_is_type(v, str) -> tm.is_(v, str, none=False)

        """
        warnings.warn(
            "assert_is_type() is deprecated. Use tm.is_(value, type, none=False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FlextTestsMatchers.is_(value, type_spec, none=False, msg=msg)

    @staticmethod
    def hasattr(
        obj: object,
        *attrs: str,
        msg: str | None = None,
    ) -> None:
        """DEPRECATED: Use tm.that(hasattr(obj, attr), eq=True) for each attribute.

        Migration:
            tm.hasattr(obj, "attr1", "attr2") ->
                tm.that(hasattr(obj, "attr1"), eq=True)
                tm.that(hasattr(obj, "attr2"), eq=True)

        """
        warnings.warn(
            "hasattr() is deprecated. Use tm.that(hasattr(obj, attr), eq=True) for each attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not attrs:
            msg = "At least one attribute name required"
            raise ValueError(msg)
        for attr in attrs:
            _ = FlextTestsMatchers.that(hasattr(obj, attr), msg=msg, eq=True)

    @staticmethod
    def method(obj: object, name: str, msg: str | None = None) -> None:
        """DEPRECATED: Use tm.that(hasattr(obj, name), eq=True) and tm.that(callable(...), eq=True).

        Migration:
            tm.that(hasattr(api, "connect"), eq=True) and tm.that(callable(getattr(api, "connect", None)), eq=True) ->
                tm.that(hasattr(api, "connect"), eq=True)
                tm.that(callable(getattr(api, "connect")), eq=True)

        """
        warnings.warn(
            "method() is deprecated. Use tm.that(hasattr(obj, name), eq=True) and tm.that(callable(...), eq=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = FlextTestsMatchers.that(hasattr(obj, name), msg=msg, eq=True)
        _ = FlextTestsMatchers.that(callable(getattr(obj, name)), msg=msg, eq=True)

    @staticmethod
    def not_none(*values: object, msg: str | None = None) -> None:
        """DEPRECATED: Use tm.that(value, none=False) for each value.

        Migration:
            tm.that(value1, value2, value3, none=False) ->
                tm.that(value1, none=False)
                tm.that(value2, none=False)
                tm.that(value3, none=False)

        """
        warnings.warn(
            "not_none() is deprecated. Use tm.that(value, none=False) for each value.",
            DeprecationWarning,
            stacklevel=2,
        )
        for value in values:
            _ = FlextTestsMatchers.that(value, msg=msg, none=False)

    # =========================================================================
    # CONVENIENCE ALIASES (DEPRECATED - Use tm.that() instead)
    # =========================================================================

    @staticmethod
    def eq(actual: object, expected: object, msg: str | None = None) -> None:
        """DEPRECATED: Use tm.that(actual, eq=expected) instead.

        Migration:
            tm.that(a, eq=b) -> tm.that(a, eq=b)

        """
        warnings.warn(
            "eq() is deprecated. Use tm.that(actual, eq=expected) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = FlextTestsMatchers.that(actual, msg=msg, eq=expected)

    @staticmethod
    def true(condition: bool, msg: str | None = None) -> None:
        """DEPRECATED: Use tm.that(condition, eq=True) instead.

        Migration:
            tm.that(callable(func, eq=True)) -> tm.that(callable(func), eq=True)

        """
        warnings.warn(
            "true() is deprecated. Use tm.that(condition, eq=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _ = FlextTestsMatchers.that(condition, msg=msg, eq=True)

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
        return FlextTestsMatchers.ok(result, msg=msg)


tm = FlextTestsMatchers

__all__ = ["FlextTestsMatchers", "tm"]
