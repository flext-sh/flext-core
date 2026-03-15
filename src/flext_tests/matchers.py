"""Test matchers and assertions for FLEXT ecosystem tests.

Provides unified assertion API with powerful generalist methods.
Short alias: tm (test matchers)

Core Philosophy:
- MINIMAL API: Only 5 core methods (ok, fail, check, that, scope)
- POWERFUL: Each method handles multiple scenarios via optional parameters
- UNIVERSAL: tm.that() does ALL validations (equality, type, length, containment, etc.)

Core Methods (5 main methods):
    tm.ok(result, **kw)     # Assert r success, optional validation
    tm.fail(result, **kw)   # Assert r failure, optional validation
    tm.check(result)        # Railway-pattern chained assertions
    tm.that(value, **kw)    # Universal assertion - ALL validations in ONE method
    tm.scope()              # Isolated test context (context manager)

Usage Examples:
    # r assertions
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
from datetime import datetime
from pathlib import Path
from typing import TypeGuard, cast, overload

from pydantic import BaseModel, RootModel, TypeAdapter, ValidationError

from flext_core import r
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_tests import c, m, t, u
from flext_tests.utilities import FlextTestsUtilities

_length_validate = FlextTestsUtilities.Tests.Length.validate
_deep_match = FlextTestsUtilities.Tests.DeepMatch.match

_TEST_PAYLOAD_DICT_ADAPTER = TypeAdapter(dict[str, t.Tests.object])
_TEST_PAYLOAD_LIST_ADAPTER = TypeAdapter(list[t.Tests.object])
_GUARD_PAYLOAD_DICT_ADAPTER = TypeAdapter(dict[str, t.Tests.object])
_GUARD_PAYLOAD_LIST_ADAPTER = TypeAdapter(list[t.Tests.object])


def _is_non_string_sequence(
    value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object | None,
) -> TypeGuard[Sequence[t.Tests.object]]:
    return isinstance(value, Sequence) and (not isinstance(value, str | bytes))


def _to_test_payload(
    value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object | None,
) -> t.Tests.object:
    if value is None or isinstance(value, (str, int, float, bool, bytes, BaseModel)):
        return value
    if isinstance(value, Mapping):
        try:
            mapping_value = _TEST_PAYLOAD_DICT_ADAPTER.validate_python(value)
            return {key: _to_test_payload(item) for key, item in mapping_value.items()}
        except ValidationError:
            return cast("t.Tests.object", dict(value))
    if _is_non_string_sequence(value):
        try:
            sequence_value = _TEST_PAYLOAD_LIST_ADAPTER.validate_python(value)
            return [_to_test_payload(seq_item) for seq_item in sequence_value]
        except ValidationError:
            return cast("t.Tests.object", list(value))
    return cast("t.Tests.object", value)


def _to_normalized(value: t.Tests.object) -> t.NormalizedValue:
    """Convert _Testobject to pure NormalizedValue."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime):
        return value
    if isinstance(value, BaseModel):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized(item) for item in value]
    return str(value)


def _to_extract_value(value: t.Tests.object) -> t.NormalizedValue | BaseModel:
    """Convert _Testobject to NormalizedValue | BaseModel for extract() calls."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, Path):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_normalized(item) for item in value]
    return str(value)


def _as_guard_input(
    value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object | None,
) -> t.Tests.object:
    if isinstance(value, BaseModel | str | int | float | bool | Path):
        return value
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            mapping_value = _GUARD_PAYLOAD_DICT_ADAPTER.validate_python(value)
            return {key: _as_guard_input(item) for key, item in mapping_value.items()}
        except ValidationError:
            return cast("t.Tests.object", {str(k): v for k, v in value.items()})
    if _is_non_string_sequence(value):
        try:
            sequence_value = _GUARD_PAYLOAD_LIST_ADAPTER.validate_python(value)
            return [_as_guard_input(seq_item) for seq_item in sequence_value]
        except ValidationError:
            return cast("t.Tests.object", list(value))
    return cast("t.Tests.object", value)


def _to_chk_value(
    value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object | None,
) -> t.NormalizedValue:
    """Convert a test value to NormalizedValue for use with u.chk()."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value
    if isinstance(value, BaseModel):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime):
        return value
    if isinstance(value, Mapping):
        typed_mapping = cast("Mapping[str, t.Tests.object]", value)
        return {str(k): _to_chk_value(v) for k, v in typed_mapping.items()}
    if isinstance(value, (list, tuple)):
        typed_seq = cast("Sequence[t.Tests.object]", value)
        return [_to_chk_value(item) for item in typed_seq]
    return str(value)


def _check_has_lacks(
    value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object | None,
    has: t.Tests.Matcher.MatcherKwargValue | None,
    lacks: t.Tests.Matcher.MatcherKwargValue | None,
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
                        )
                    )
            else:
                check_val = _as_guard_input(item)
                target_raw = _as_guard_input(value)
                if isinstance(target_raw, RootModel):
                    target_raw = _as_guard_input(target_raw.model_dump())
                if not isinstance(
                    target_raw, (Mapping, str, list, tuple, set, frozenset)
                ):
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value, item=item
                        )
                    )
                if isinstance(target_raw, (set, frozenset, tuple)):
                    if check_val not in target_raw:
                        raise AssertionError(
                            msg
                            or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                                container=value, item=item
                            )
                        )
                    continue
                if isinstance(target_raw, str):
                    if str(check_val) not in target_raw:
                        raise AssertionError(
                            msg
                            or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                                container=value, item=item
                            )
                        )
                elif check_val not in target_raw:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_CONTAINS_FAILED.format(
                            container=value, item=item
                        )
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
                        )
                    )
            else:
                check_val = _as_guard_input(item)
                target_raw_2 = _as_guard_input(value)
                if isinstance(target_raw_2, RootModel):
                    target_raw_2 = _as_guard_input(target_raw_2.model_dump())
                if not isinstance(target_raw_2, (Mapping, str, list)):
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value, item=item
                        )
                    )
                if isinstance(target_raw_2, str):
                    if str(check_val) in target_raw_2:
                        raise AssertionError(
                            msg
                            or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                                container=value, item=item
                            )
                        )
                elif check_val in target_raw_2:
                    raise AssertionError(
                        msg
                        or c.Tests.Matcher.ERR_LACKS_FAILED.format(
                            container=value, item=item
                        )
                    )


class FlextTestsMatchers:
    """Test matchers with powerful generalist methods.

    Short alias: tm

    Core Methods (5 main methods):
        tm.ok(result, **kw)     - Assert r success, optional validation
        tm.fail(result, **kw)   - Assert r failure, optional validation
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

    @staticmethod
    def assert_result_success[TResult](
        result: r[TResult], msg: str | None = None
    ) -> TResult:
        """Assert result is success and return unwrapped value.

        Args:
            result: r to check
            msg: Optional custom error message

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure

        """
        if not result.is_success:
            error_msg = msg or f"Expected success but got failure: {result.error}"
            raise AssertionError(error_msg)
        return result.value

    @staticmethod
    def check[TResult](result: r[TResult]) -> m.Tests.Chain[TResult]:
        """Start chained assertions on result (railway pattern)."""
        return m.Tests.Chain[TResult](result=result)

    @staticmethod
    def fail[TResult](
        result: r[TResult], **kwargs: t.Tests.Matcher.MatcherKwargValue
    ) -> str:
        r"""Enhanced assertion for r failure with optional error validation.

        Examples:
            # Basic failure assertions
            tm.fail(result)                   # Assert failure
            tm.fail(result, has="not found")  # Failure with error containing
            tm.fail(result, code="VALIDATION")  # Failure with specific code
            tm.fail(result, match=r"Error: \\\\d+")  # Error matches regex

            # Multiple error checks
            tm.fail(result, has=["invalid", "required"], lacks="internal")

            # Error metadata checks
            tm.fail(result, code="VALIDATION", data={"field": "email"})

        Args:
            result: r to check
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
            ValueError: If parameter validation fails (via Pydantic model)

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated via m.Tests.FailParams model.

        """
        try:
            params = m.Tests.FailParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc
        if result.is_success:
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_FAIL_EXPECTED.format(value=result.value)
            )
        err = result.error or ""
        if params.has or params.lacks or params.starts or params.ends or params.match:
            _check_has_lacks(err, params.has, params.lacks, params.msg, as_str=True)
            if params.starts is not None and (not u.chk(err, starts=params.starts)):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_STARTSWITH.format(
                        text=err, prefix=params.starts
                    )
                )
            if params.ends is not None and (not u.chk(err, ends=params.ends)):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_ENDSWITH.format(
                        text=err, suffix=params.ends
                    )
                )
            if params.match is not None and (not u.chk(err, match=params.match)):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_NOT_MATCHES.format(
                        text=err, pattern=params.match
                    )
                )
        if params.code is not None:
            actual_code = result.error_code
            if actual_code != params.code:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_ERROR_CODE_MISMATCH.format(
                        expected=params.code, actual=actual_code
                    )
                )
        if params.code_has is not None:
            actual_code = result.error_code or ""
            code_has_value = params.code_has
            if isinstance(code_has_value, str):
                items_list: list[str] = [code_has_value]
            else:
                items_list = [str(x) for x in code_has_value]
            for item in items_list:
                if item not in actual_code:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_CODE_NOT_CONTAINS.format(
                            expected=item, actual=actual_code
                        )
                    )
        if params.data is not None:
            actual_raw = result.error_data
            actual_data: MutableMapping[str, t.Tests.object] = {}
            if actual_raw is not None:
                actual_data = {
                    str(k): _to_test_payload(v) for k, v in actual_raw.root.items()
                }
            for key, expected_value in params.data.items():
                if key not in actual_data:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_DATA_KEY_MISSING.format(key=key)
                    )
                if actual_data[key] != expected_value:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_ERROR_DATA_VALUE_MISMATCH.format(
                            key=key, expected=expected_value, actual=actual_data[key]
                        )
                    )
        return err

    @staticmethod
    @overload
    def ok[TResult](
        result: r[TResult],
    ) -> TResult: ...

    @staticmethod
    @overload
    def ok[TResult](
        result: r[TResult], **kwargs: t.Tests.Matcher.MatcherKwargValue
    ) -> TResult | t.Tests.object: ...

    @staticmethod
    def ok[TResult](
        result: r[TResult], **kwargs: t.Tests.Matcher.MatcherKwargValue
    ) -> TResult | t.Tests.object:
        """Enhanced assertion for r success with optional value validation.

        Uses Pydantic 2 models for parameter validation and computation.
        All parameters are validated via m.Tests.OkParams model.

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
            result: r to validate
            **kwargs: Parameters validated via m.Tests.OkParams model
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
            params = m.Tests.OkParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc
        if not result.is_success:
            raise AssertionError(
                params.msg or c.Tests.Matcher.ERR_OK_FAILED.format(error=result.error)
            )
        result_value: t.Tests.Matcher.MatcherKwargValue = cast(
            "t.Tests.Matcher.MatcherKwargValue", result.value
        )
        extracted_payload: t.Tests.object | None = None
        if params.path is not None:
            if isinstance(params.path, str):
                path_str: str = params.path
            else:
                path_str = ".".join(params.path)
            if not (isinstance(result_value, (BaseModel, Mapping))):
                raise AssertionError(
                    params.msg
                    or f"Path extraction requires dict or model, got {type(result_value).__name__}"
                )
            extract_data: BaseModel | Mapping[str, t.NormalizedValue | BaseModel]
            if isinstance(result_value, BaseModel):
                extract_data = result_value
            else:
                mapping_value = cast("Mapping[str, t.Tests.object]", result_value)
                try:
                    validated = _GUARD_PAYLOAD_DICT_ADAPTER.validate_python(
                        mapping_value
                    )
                    extract_data = {
                        str(k): _to_extract_value(v) for k, v in validated.items()
                    }
                except ValidationError:
                    extract_data = {}
            extracted = u.extract(extract_data, path_str)
            if extracted.is_failure:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_SCOPE_PATH_NOT_FOUND.format(
                        path=path_str, error=extracted.error
                    )
                )
            extracted_payload = _to_test_payload(
                cast("t.Tests.Matcher.MatcherKwargValue", extracted.value)
            )
            result_value = extracted_payload
        has_validation = (
            params.eq is not None
            or params.ne is not None
            or params.none is not None
            or (params.empty is not None)
            or (params.gt is not None)
            or (params.gte is not None)
            or (params.lt is not None)
            or (params.lte is not None)
            or (params.starts is not None)
            or (params.ends is not None)
            or (params.match is not None)
        )
        if has_validation:
            is_type = params.is_ if not isinstance(params.is_, tuple) else None
            if not u.chk(
                _to_chk_value(result_value),
                eq=_to_chk_value(params.eq) if params.eq is not None else None,
                ne=_to_chk_value(params.ne) if params.ne is not None else None,
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
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not isinstance(result_value, params.is_)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=params.is_, actual=type(result_value).__name__
                )
            )
        _check_has_lacks(result_value, params.has, params.lacks, params.msg)
        result_payload = _to_test_payload(result_value)
        if params.len is not None and (
            not _length_validate(result_payload, params.len)
        ):
            actual_len = len(result_value) if isinstance(result_value, Sized) else 0
            if isinstance(params.len, int):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_LEN_EXACT_FAILED.format(
                        expected=params.len, actual=actual_len
                    )
                )
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_LEN_RANGE_FAILED.format(
                    min=params.len[0], max=params.len[1], actual=actual_len
                )
            )
        if params.deep is not None:
            if not isinstance(result_value, (BaseModel, Mapping)):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(result_value).__name__}"
                )
            deep_input: BaseModel | Mapping[str, t.Tests.object]
            if isinstance(result_value, BaseModel):
                deep_input = result_value
            else:
                try:
                    deep_input = _TEST_PAYLOAD_DICT_ADAPTER.validate_python(
                        result_value
                    )
                except ValidationError:
                    deep_input = {}
            match_result = _deep_match(deep_input, params.deep)
            if not match_result.matched:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_DEEP_PATH_FAILED.format(
                        path=match_result.path, reason=match_result.reason
                    )
                )
        if params.path is None:
            result_payload = _to_test_payload(
                cast("t.Tests.Matcher.MatcherKwargValue", result.value)
            )
        else:
            result_payload = extracted_payload or _to_test_payload(
                cast("t.Tests.Matcher.MatcherKwargValue", result.value)
            )
        if params.where is not None and (not params.where(result_payload)):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_PREDICATE_FAILED.format(value=result_payload)
            )
        if result_value is None:
            raise AssertionError(
                params.msg
                or "Value is None but validation passed - this should not happen"
            )
        return result_payload

    @staticmethod
    @contextmanager
    def scope(**kwargs: t.Tests.object) -> Iterator[m.Tests.TestScope]:
        """Enhanced isolated test execution scope.

        Uses Pydantic 2 model (ScopeParams) for parameter validation and computation.
        All parameters are validated automatically via u.from_kwargs.

        Provides isolated configuration, container, and context for tests.
        Supports temporary environment variables, working directory changes,
        and automatic cleanup functions.

        Args:
            **kwargs: Parameters validated via m.ScopeParams model
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
        try:
            params = m.Tests.ScopeParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Parameter validation failed: {exc}") from exc
        original_env: dict[str, str | None] = {}
        original_cwd: Path | None = None
        try:
            if params.env is not None:
                for key, value in params.env.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value
            if params.cwd is not None:
                original_cwd = Path.cwd()
                cwd_path = (
                    Path(params.cwd) if u.is_type(params.cwd, "str") else params.cwd
                )
                os.chdir(cwd_path)
            cfg: dict[str, t.Tests.object] = {}
            if params.config:
                cfg = {str(key): value for key, value in params.config.items()}
            container_dict: dict[str, t.Tests.object] = {
                k: v
                for k, v in (params.container or {}).items()
                if t.Guards.is_general_value(v)
            }
            context_map: dict[str, t.Tests.object] = {}
            if params.context:
                context_map = {str(key): value for key, value in params.context.items()}
            yield m.Tests.TestScope.model_validate({
                "config": cfg,
                "container": container_dict,
                "context": context_map,
            })
        finally:
            if params.env is not None:
                for key, original_value in original_env.items():
                    if original_value is None:
                        _ = os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
            if original_cwd is not None:
                os.chdir(original_cwd)
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
                        warnings.warn(
                            c.Tests.Matcher.ERR_SCOPE_CLEANUP_FAILED.format(
                                error=str(e)
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )

    @staticmethod
    def that(
        value: t.Tests.Matcher.MatcherKwargValue | t.Tests.object,
        **kwargs: t.Tests.Matcher.MatcherKwargValue,
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
            tm.that(email, match=r"^[\\w.]+@[\\w.]+$")

            # Sequence assertions
            tm.that(items, len=5, first="a", last="z", unique=True)
            tm.that(items, all_=str, sorted=True)
            tm.that(items, has=["required1", "required2"])

            # Mapping assertions
            tm.that(data, keys=["id", "name"], kv={"status": "active"})
            tm.that(config, attrs=["debug", "timeout"], attr_eq={"debug": True})

            # r in tm.that() (auto-detected)
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
            ok: For r: assert success
            error: For r: error contains
            deep: Deep structural matching specification
            where: Custom predicate function
            contains, excludes, length, length_gt, etc.: Legacy parameters (deprecated)

        Raises:
            AssertionError: If value doesn't satisfy constraints
            ValueError: If parameter validation fails (via Pydantic model)

        """
        raw_eq = kwargs.get("eq") if "eq" in kwargs else None
        raw_ne = kwargs.get("ne") if "ne" in kwargs else None
        raw_has = kwargs.get("has") if "has" in kwargs else None
        raw_contains = kwargs.get("contains") if "contains" in kwargs else None
        try:
            params = m.Tests.ThatParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError):
            non_serializable_keys = {"eq", "ne", "has", "contains", "lacks", "excludes"}
            filtered_kwargs = {
                key: val
                for key, val in kwargs.items()
                if key not in non_serializable_keys
            }
            try:
                params = m.Tests.ThatParams.model_validate(filtered_kwargs)
            except (TypeError, ValueError, AttributeError) as filtered_exc:
                raise ValueError(
                    f"Parameter validation failed: {filtered_exc}"
                ) from filtered_exc
        subject = value
        if isinstance(subject, BaseModel) and FlextUtilitiesGuards.is_result_like(
            subject
        ):
            result_obj = subject
            actual_value: t.Tests.object | str = ""
            if params.ok is not None:
                if params.ok and (not result_obj.is_success):
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_OK_FAILED.format(error=result_obj.error)
                    )
                if not params.ok and result_obj.is_success:
                    value_str: str = str(result_obj.value)
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_FAIL_EXPECTED.format(value=value_str)
                    )
                if result_obj.is_success:
                    actual_value = getattr(result_obj, "value", "")
            elif params.has is not None:
                err = result_obj.error or ""
                _check_has_lacks(err, params.has, None, params.msg, as_str=True)
                actual_value = err
            elif params.ok is None:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_OK_FAILED.format(error=result_obj.error)
                )
            else:
                actual_value = result_obj.error or ""
            subject = _to_test_payload(actual_value)
        subject_payload = _to_test_payload(subject)
        has_validation = (
            raw_eq is not None
            or raw_ne is not None
            or params.eq is not None
            or (params.ne is not None)
            or (params.gt is not None)
            or (params.gte is not None)
            or (params.lt is not None)
            or (params.lte is not None)
            or (params.none is not None)
            or (params.empty is not None)
            or (params.starts is not None)
            or (params.ends is not None)
            or (params.match is not None)
        )
        if has_validation:
            eq_value = raw_eq if "eq" in kwargs else params.eq
            ne_value = raw_ne if "ne" in kwargs else params.ne
            if not u.chk(
                _to_chk_value(subject_payload),
                eq=_to_chk_value(eq_value) if eq_value is not None else None,
                ne=_to_chk_value(ne_value) if ne_value is not None else None,
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
                    or f"Assertion failed: {subject_payload!r} did not satisfy constraints"
                )
                raise AssertionError(error_msg)
        if (
            params.is_ is not None
            and (not isinstance(params.is_, tuple))
            and (not isinstance(subject_payload, params.is_))
        ):
            raise AssertionError(
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=params.is_, actual=type(subject_payload).__name__)}"
            )
        if (
            params.not_ is not None
            and (not isinstance(params.not_, tuple))
            and isinstance(subject_payload, params.not_)
        ):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_TYPE_FAILED.format(
                    expected=f"not {params.not_}", actual=type(subject_payload).__name__
                )
            )
        if (
            params.is_ is not None
            and isinstance(params.is_, tuple)
            and not isinstance(subject_payload, params.is_)
        ):
            raise AssertionError(
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=params.is_, actual=type(subject_payload).__name__)}"
            )
        if (
            params.not_ is not None
            and isinstance(params.not_, tuple)
            and isinstance(subject_payload, params.not_)
        ):
            error_msg = (
                params.msg
                or f"Assertion failed: {c.Tests.Matcher.ERR_TYPE_FAILED.format(expected=f'not {params.not_}', actual=type(subject_payload).__name__)}"
            )
            raise AssertionError(error_msg)
        effective_has = (
            raw_has
            if raw_has is not None
            else (raw_contains if raw_contains is not None else params.has)
        )
        _check_has_lacks(subject_payload, effective_has, params.lacks, params.msg)
        value_payload = subject_payload
        if params.len is not None and (not _length_validate(value_payload, params.len)):
            actual_len = (
                len(subject_payload) if isinstance(subject_payload, Sized) else 0
            )
            if isinstance(params.len, int):
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_LEN_EXACT_FAILED.format(
                        expected=params.len, actual=actual_len
                    )
                )
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_LEN_RANGE_FAILED.format(
                    min=params.len[0], max=params.len[1], actual=actual_len
                )
            )
        if isinstance(subject_payload, (list, tuple)):
            seq_value: list[t.Tests.object] = []
            try:
                seq_value = _TEST_PAYLOAD_LIST_ADAPTER.validate_python(subject_payload)
            except ValidationError:
                pass
            if params.first is not None:
                if not seq_value:
                    raise AssertionError(
                        params.msg or "Sequence is empty, cannot check first"
                    )
                if seq_value[0] != params.first:
                    raise AssertionError(
                        params.msg
                        or f"First item: expected {params.first!r}, got {seq_value[0]!r}"
                    )
            if params.last is not None:
                if not seq_value:
                    raise AssertionError(
                        params.msg or "Sequence is empty, cannot check last"
                    )
                if seq_value[-1] != params.last:
                    raise AssertionError(
                        params.msg
                        or f"Last item: expected {params.last!r}, got {seq_value[-1]!r}"
                    )
            if params.all_ is not None:
                if isinstance(params.all_, type):

                    def _all_match(
                        check_type: type, seq: Sequence[t.Tests.object]
                    ) -> bool:
                        return all(isinstance(x, check_type) for x in seq)

                    if not _all_match(params.all_, seq_value):
                        failed_idx = next(
                            (
                                i
                                for i, item in enumerate(list(seq_value))
                                if not isinstance(item, params.all_)
                            ),
                            None,
                        )
                        raise AssertionError(
                            params.msg
                            or c.Tests.Matcher.ERR_ALL_ITEMS_FAILED.format(
                                index=failed_idx
                            )
                        )
                elif callable(params.all_) and (
                    not all(params.all_(_to_test_payload(item)) for item in seq_value)
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
                        or c.Tests.Matcher.ERR_ALL_ITEMS_FAILED.format(index=failed_idx)
                    )
            if params.any_ is not None:
                if isinstance(params.any_, type):
                    any_type = params.any_
                    if not any(isinstance(item, any_type) for item in seq_value):
                        raise AssertionError(
                            params.msg or c.Tests.Matcher.ERR_ANY_ITEMS_FAILED
                        )
                elif callable(params.any_) and (
                    not any(params.any_(_to_test_payload(item)) for item in seq_value)
                ):
                    raise AssertionError(
                        params.msg or c.Tests.Matcher.ERR_ANY_ITEMS_FAILED
                    )
            sorted_param = params.sorted
            if sorted_param is not None:
                value_list = list(seq_value)
                if sorted_param is True:
                    sorted_list = sorted(
                        value_list, key=lambda x: (type(x).__name__, str(x))
                    )
                    if value_list != sorted_list:
                        raise AssertionError(params.msg or "Sequence is not sorted")
                elif callable(sorted_param):
                    user_key_fn = sorted_param

                    def comparable_key(x: t.Tests.object) -> tuple[str, str]:
                        """Wrap user key to return comparable tuple."""
                        result = user_key_fn(_to_test_payload(x))
                        type_name = type(result).__name__
                        return (str(type_name), str(result))

                    sorted_list = sorted(value_list, key=comparable_key)
                    if value_list != sorted_list:
                        raise AssertionError(
                            params.msg or "Sequence is not sorted by key function"
                        )
            if params.unique is not None and params.unique:
                value_len = len(seq_value)
                value_set_len = len(set(seq_value))
                if value_len != value_set_len:
                    raise AssertionError(
                        params.msg or "Sequence contains duplicate items"
                    )
        if isinstance(subject_payload, Mapping):
            mapping_value: dict[str, t.Tests.object] = {}
            try:
                mapping_value = _TEST_PAYLOAD_DICT_ADAPTER.validate_python(
                    subject_payload
                )
            except ValidationError:
                pass
            if params.keys is not None:
                key_set: set[str] = set(params.keys)
                missing = key_set - set(mapping_value.keys())
                if missing:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_KEYS_MISSING.format(keys=list(missing))
                    )
            if params.lacks_keys is not None:
                lacks_key_set: set[str] = set(params.lacks_keys)
                present = lacks_key_set & set(mapping_value.keys())
                if present:
                    raise AssertionError(
                        params.msg
                        or c.Tests.Matcher.ERR_KEYS_EXTRA.format(keys=list(present))
                    )
            if params.values is not None:
                value_list = list(mapping_value.values())
                for expected_val in params.values:
                    if expected_val not in value_list:
                        raise AssertionError(
                            params.msg
                            or f"Expected value {expected_val!r} not found in mapping"
                        )
            if params.kv is not None:
                if isinstance(params.kv, tuple) and len(params.kv) == 2:
                    key, expected_val = params.kv
                    if key not in mapping_value:
                        raise AssertionError(
                            params.msg or f"Key {key!r} not found in mapping"
                        )
                    if mapping_value[key] != expected_val:
                        raise AssertionError(
                            params.msg
                            or f"Key {key!r}: expected {expected_val!r}, got {mapping_value[key]!r}"
                        )
                elif hasattr(params.kv, "keys") and hasattr(params.kv, "items"):
                    mapping_kv: Mapping[str, object] = params.kv
                    for key, expected_obj in mapping_kv.items():
                        if key not in mapping_value:
                            raise AssertionError(
                                params.msg or f"Key {key!r} not found in mapping"
                            )
                        if mapping_value[key] != expected_obj:
                            raise AssertionError(
                                params.msg
                                or f"Key {key!r}: expected {expected_obj!r}, got {mapping_value[key]!r}"
                            )
        if params.attrs is not None:
            attrs_target = value
            if isinstance(params.attrs, str):
                attr_list: list[str] = [params.attrs]
            else:
                attr_list = list(params.attrs)
            for attr in attr_list:
                if not hasattr(attrs_target, attr):
                    raise AssertionError(
                        params.msg or f"Object missing attribute: {attr}"
                    )
        if params.methods is not None:
            methods_target = value
            if isinstance(params.methods, str):
                method_list: list[str] = [params.methods]
            else:
                method_list = list(params.methods)
            for method in method_list:
                if not hasattr(methods_target, method):
                    raise AssertionError(
                        params.msg or f"Object missing method: {method}"
                    )
                if not callable(getattr(methods_target, method)):
                    raise AssertionError(
                        params.msg or f"Object attribute {method} is not callable"
                    )
        if params.attr_eq is not None:
            attr_eq_target = value
            if isinstance(params.attr_eq, tuple) and len(params.attr_eq) == 2:
                attr, expected_val = params.attr_eq
                if not hasattr(attr_eq_target, attr):
                    raise AssertionError(
                        params.msg or f"Object missing attribute: {attr}"
                    )
                actual_val = getattr(attr_eq_target, attr)
                if actual_val != expected_val:
                    raise AssertionError(
                        params.msg
                        or f"Attribute {attr}: expected {expected_val!r}, got {actual_val!r}"
                    )
            else:
                for attr, expected_val in params.attr_eq.items():
                    if not hasattr(attr_eq_target, attr):
                        raise AssertionError(
                            params.msg or f"Object missing attribute: {attr}"
                        )
                    actual_val = getattr(attr_eq_target, attr)
                    if actual_val != expected_val:
                        raise AssertionError(
                            params.msg
                            or f"Attribute {attr}: expected {expected_val!r}, got {actual_val!r}"
                        )
        if params.deep is not None:
            if not isinstance(subject_payload, (BaseModel, dict)):
                raise AssertionError(
                    params.msg
                    or f"Deep matching requires dict or model, got {type(subject_payload).__name__}"
                )
            deep_value: BaseModel | Mapping[str, t.Tests.object]
            if isinstance(subject_payload, BaseModel):
                deep_value = subject_payload
            else:
                try:
                    deep_value = _TEST_PAYLOAD_DICT_ADAPTER.validate_python(
                        subject_payload
                    )
                except ValidationError:
                    deep_value = {}
            match_result = _deep_match(deep_value, params.deep)
            if not match_result.matched:
                raise AssertionError(
                    params.msg
                    or c.Tests.Matcher.ERR_DEEP_PATH_FAILED.format(
                        path=match_result.path, reason=match_result.reason
                    )
                )
        if params.where is not None and (not params.where(subject_payload)):
            raise AssertionError(
                params.msg
                or c.Tests.Matcher.ERR_PREDICATE_FAILED.format(value=subject_payload)
            )


tm = FlextTestsMatchers
__all__ = ["FlextTestsMatchers", "tm"]
