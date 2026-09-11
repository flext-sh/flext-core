"""Result matcher utilities for flext-tests."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import cast, overload

from flext_core import p as core_p
from flext_core import u
from flext_tests import c, m, p, t
from flext_tests._utilities._matchers._assertions import (
    FlextTestsMatchersAssertionsMixin,
)
from flext_tests._utilities._matchers._containment import (
    FlextTestsMatchersContainmentMixin,
)
from flext_tests._utilities._matchers._that import (
    FlextTestsMatchersThatMixin as FlextTestsMatchersRulesMixin,
)
from flext_tests._utilities._matchers._typeguards import (
    FlextTestsMatchersTypeGuardsMixin,
)
from flext_tests._utilities.payload import FlextTestsPayloadUtilities
from flext_tests._utilities.result import FlextTestsResultUtilitiesMixin


class FlextTestsMatchersResultMixin:
    """Assertions over r[T] result values."""

    class Tests:
        """Test utility namespace."""

        class Matchers:
            """Matcher assertion helpers."""

            @staticmethod
            def check[TResult](result: p.Result[TResult]) -> m.Tests.Chain[TResult]:
                """Start chained assertions on result."""
                return m.Tests.Chain(result=result)

            @staticmethod
            def fail[TResult](
                result: core_p.ResultView[TResult], **kwargs: t.Tests.MatcherKwargValue
            ) -> str:
                """Assert that a result failed and validate its error payload."""
                try:
                    params = m.Tests.FailParams.model_validate(kwargs)
                except c.EXC_BASIC_TYPE as exc:
                    msg = f"Parameter validation failed: {exc}"
                    raise ValueError(msg) from exc
                err: str = FlextTestsResultUtilitiesMixin.assert_failure(result)
                FlextTestsMatchersResultMixin.Tests.Matchers.fail_text(err, params)
                FlextTestsMatchersResultMixin.Tests.Matchers.fail_code(result, params)
                FlextTestsMatchersResultMixin.Tests.Matchers.fail_data(result, params)
                return err

            @staticmethod
            def fail_text(err: str, params: m.Tests.FailParams) -> None:
                """Validate textual error constraints."""
                if not any(
                    getattr(params, name)
                    for name in ("has", "lacks", "starts", "ends", "match")
                ):
                    return
                FlextTestsMatchersContainmentMixin.check_has_lacks(
                    err, params.has, params.lacks, params.msg, as_str=True
                )
                if params.starts is not None and (
                    not u.chk(err, m.GuardCheckSpec(starts=params.starts))
                ):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_NOT_STARTSWITH.format(
                            text=err, prefix=params.starts
                        )
                    )
                if params.ends is not None and (
                    not u.chk(err, m.GuardCheckSpec(ends=params.ends))
                ):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_NOT_ENDSWITH.format(text=err, suffix=params.ends)
                    )
                if params.match is not None and params.match.search(err) is None:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_NOT_MATCHES.format(
                            text=err, pattern=params.match.pattern
                        )
                    )

            @staticmethod
            def fail_code[TResult](
                result: core_p.ResultView[TResult], params: m.Tests.FailParams
            ) -> None:
                """Validate error code constraints."""
                if params.code is not None and result.error_code != params.code:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_ERROR_CODE_MISMATCH.format(
                            expected=params.code, actual=result.error_code
                        )
                    )
                if params.code_has is None:
                    return
                actual_code = result.error_code or ""
                code_items: t.StrSequence = (
                    [params.code_has]
                    if isinstance(params.code_has, str)
                    else list(params.code_has)
                )
                for item in code_items:
                    if item not in actual_code:
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_ERROR_CODE_NOT_CONTAINS.format(
                                expected=item, actual=actual_code
                            )
                        )

            @staticmethod
            def fail_data[TResult](
                result: core_p.ResultView[TResult], params: m.Tests.FailParams
            ) -> None:
                """Validate structured error data constraints."""
                if params.data is None:
                    return
                actual_raw = result.error_data
                actual_data: MutableMapping[str, t.Tests.TestobjectSerializable] = {}
                if actual_raw is not None:
                    actual_data = {
                        key: FlextTestsPayloadUtilities.to_payload(value)
                        for key, value in actual_raw.items()
                    }
                for key, expected_value in params.data.items():
                    if key not in actual_data:
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_ERROR_DATA_KEY_MISSING.format(key=key)
                        )
                    if actual_data[key] != expected_value:
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_ERROR_DATA_VALUE_MISMATCH.format(
                                key=key,
                                expected=expected_value,
                                actual=actual_data[key],
                            )
                        )

            @staticmethod
            def ok_extract_path(
                result_value: t.Tests.TestResultValue, params: m.Tests.OkParams
            ) -> tuple[t.Tests.TestResultValue, t.Tests.TestobjectSerializable | None]:
                """Apply optional path extraction to a successful result value."""
                if params.path is None:
                    return result_value, None
                path_str = (
                    params.path
                    if isinstance(params.path, str)
                    else ".".join(params.path)
                )
                # mro-j47u: matcher type failures remain assertion failures.
                match result_value:
                    case m.BaseModel() | Mapping():
                        pass
                    case _:
                        failure = (
                            "Path extraction requires dict or model, got "
                            f"{type(result_value).__name__}"
                        )
                        raise AssertionError(params.msg or failure)
                try:
                    extract_data = FlextTestsPayloadUtilities.to_config_map(
                        result_value
                    )
                except c.ValidationError as exc:
                    failure = f"Path extraction payload is invalid: {exc}"
                    raise AssertionError(params.msg or failure) from exc
                extracted = u.extract(extract_data, path_str)
                if extracted.failure:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_SCOPE_PATH_NOT_FOUND.format(
                            path=path_str, error=extracted.error
                        )
                    )
                extracted_payload = FlextTestsPayloadUtilities.to_payload(
                    extracted.value
                )
                return extracted_payload, extracted_payload

            @staticmethod
            def ok_has_scalar_validation(params: m.Tests.OkParams) -> bool:
                """Return whether scalar guard validation is requested."""
                return any(
                    getattr(params, name) is not None
                    for name in (
                        "eq",
                        "ne",
                        "none",
                        "empty",
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                        "starts",
                        "ends",
                        "match",
                    )
                )

            @staticmethod
            def ok_preserves_result_identity(params: m.Tests.OkParams) -> bool:
                """True when no structural extraction kwargs are set (return TResult)."""
                return all(
                    getattr(params, name) is None
                    for name in (
                        "path",
                        "len",
                        "deep",
                        "paths",
                        "items",
                        "attrs_match",
                        "where",
                    )
                )

            @staticmethod
            def ok_validate_scalar[TResult: t.Tests.TestResultValue](
                result_value: TResult | t.Tests.TestobjectSerializable,
                params: m.Tests.OkParams,
            ) -> TResult | t.Tests.TestobjectSerializable:
                """Validate scalar predicates for a successful result."""
                if not FlextTestsMatchersResultMixin.Tests.Matchers.ok_has_scalar_validation(
                    params
                ):
                    return result_value
                is_type = params.is_ if not isinstance(params.is_, tuple) else None
                result_payload = FlextTestsPayloadUtilities.to_payload(result_value)
                eq_payload, ne_payload = (
                    FlextTestsMatchersTypeGuardsMixin.prepare_eq_ne_payloads(
                        result_payload,
                        params.eq,
                        params.ne,
                        msg=params.msg,
                        default_msg=f"Value {result_value!r} did not satisfy constraints",
                    )
                )
                chk_value: t.GuardInput | None = (
                    None
                    if result_payload is None
                    else FlextTestsPayloadUtilities.to_normalized_value(result_payload)
                )
                guard = m.GuardCheckSpec.model_validate({
                    "eq": FlextTestsPayloadUtilities.to_normalized_value(eq_payload)
                    if eq_payload is not None
                    else None,
                    "ne": FlextTestsPayloadUtilities.to_normalized_value(ne_payload)
                    if ne_payload is not None
                    else None,
                    "is_": is_type,
                    "none": params.none,
                    "empty": params.empty,
                    "gt": params.gt,
                    "gte": params.gte,
                    "lt": params.lt,
                    "lte": params.lte,
                    "starts": params.starts,
                    "ends": params.ends,
                })
                if not u.chk(chk_value, guard):
                    raise AssertionError(
                        params.msg
                        or f"Value {result_value!r} did not satisfy constraints"
                    )
                if (
                    params.match is not None
                    and isinstance(result_payload, str)
                    and params.match.search(result_payload) is None
                ):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_NOT_MATCHES.format(
                            text=result_payload, pattern=params.match.pattern
                        )
                    )
                return result_value

            @staticmethod
            def ok_validate_type[TResult: t.Tests.TestResultValue](
                result_value: TResult | t.Tests.TestobjectSerializable,
                params: m.Tests.OkParams,
            ) -> TResult | t.Tests.TestobjectSerializable:
                """Validate tuple-based runtime type constraints."""
                if not (
                    params.is_ is not None
                    and isinstance(params.is_, tuple)
                    and not any(
                        FlextTestsMatchersTypeGuardsMixin.matches_runtime_type(
                            result_value, expected_type
                        )
                        for expected_type in params.is_
                    )
                ):
                    return result_value
                raise AssertionError(
                    params.msg
                    or c.Tests.ERR_TYPE_FAILED.format(
                        expected=params.is_, actual=type(result_value).__name__
                    )
                )

            @staticmethod
            @overload
            def ok[TResult](result: core_p.ResultView[TResult]) -> TResult: ...

            @staticmethod
            @overload
            def ok[TResult: t.Tests.TestResultValue](
                result: core_p.ResultView[TResult], **kwargs: t.Tests.MatcherKwargValue
            ) -> TResult | t.Tests.TestobjectSerializable: ...

            @staticmethod
            def ok[TResult](
                result: core_p.ResultView[TResult], **kwargs: t.Tests.MatcherKwargValue
            ) -> TResult | t.Tests.TestobjectSerializable:
                # mro-j47u: matchers observe the protocol and preserve source identity.
                if not kwargs:
                    return FlextTestsResultUtilitiesMixin.assert_success(result)
                structured_result = cast(
                    "core_p.ResultView[t.Tests.TestResultValue]", result
                )
                try:
                    params = m.Tests.OkParams.model_validate(kwargs)
                except c.EXC_BASIC_TYPE as exc:
                    msg = f"Parameter validation failed: {exc}"
                    raise ValueError(msg) from exc
                result_value: t.Tests.TestResultValue = (
                    FlextTestsResultUtilitiesMixin.assert_success(
                        structured_result, error_msg=params.msg
                    )
                )
                result_value, extracted_payload = (
                    FlextTestsMatchersResultMixin.Tests.Matchers.ok_extract_path(
                        result_value, params
                    )
                )
                result_value = (
                    FlextTestsMatchersResultMixin.Tests.Matchers.ok_validate_scalar(
                        result_value, params
                    )
                )
                result_value = (
                    FlextTestsMatchersResultMixin.Tests.Matchers.ok_validate_type(
                        result_value, params
                    )
                )
                FlextTestsMatchersContainmentMixin.check_has_lacks(
                    result_value, params.has, params.lacks, params.msg
                )
                result_payload = (
                    FlextTestsMatchersResultMixin.Tests.Matchers.ok_payload(
                        result, result_value, extracted_payload, params
                    )
                )
                FlextTestsMatchersResultMixin.Tests.Matchers.ok_validate_structured(
                    result, result_value, result_payload, params
                )
                if result_value is None:
                    raise AssertionError(
                        params.msg
                        or "Value is None but validation passed - this should not happen"
                    )
                # Preserve source identity for the no-kwargs overload (TResult).
                # Structural matchers still observe via result_payload above.
                if FlextTestsMatchersResultMixin.Tests.Matchers.ok_preserves_result_identity(
                    params
                ):
                    return cast("TResult", result_value)
                return result_payload

            @staticmethod
            def ok_payload[TResult: t.Tests.TestResultValue](
                result: core_p.ResultView[TResult],
                result_value: t.Tests.TestResultValue,
                extracted_payload: t.Tests.TestobjectSerializable | None,
                params: m.Tests.OkParams,
            ) -> t.Tests.TestobjectSerializable:
                if params.path is None:
                    return FlextTestsPayloadUtilities.to_payload(result.value)
                if extracted_payload is not None:
                    return extracted_payload
                return FlextTestsPayloadUtilities.to_payload(result_value)

            @staticmethod
            def ok_validate_structured[TResult: t.Tests.TestResultValue](
                result: core_p.ResultView[TResult],
                result_value: t.Tests.TestResultValue,
                result_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.OkParams,
            ) -> None:
                if params.len is not None:
                    FlextTestsMatchersAssertionsMixin.assert_len_match(
                        payload=result_payload,
                        sized=result_value,
                        length_spec=params.len,
                        msg=params.msg,
                    )
                if params.deep is not None:
                    FlextTestsMatchersResultMixin.Tests.Matchers.ok_validate_deep(
                        result_value, params
                    )
                if params.paths is not None:
                    FlextTestsMatchersRulesMixin.apply_path_rules(
                        result_payload, params.paths, inherited_msg=params.msg
                    )
                if params.items is not None:
                    FlextTestsMatchersRulesMixin.apply_item_rules(
                        result_payload, params.items, inherited_msg=params.msg
                    )
                if params.attrs_match is not None:
                    FlextTestsMatchersRulesMixin.apply_attribute_rules(
                        result.value, params.attrs_match, inherited_msg=params.msg
                    )
                if params.where is not None and (not params.where(result_payload)):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_PREDICATE_FAILED.format(value=result_payload)
                    )

            @staticmethod
            def ok_validate_deep[TResult: t.Tests.TestResultValue](
                result_value: TResult | t.Tests.TestobjectSerializable,
                params: m.Tests.OkParams,
            ) -> TResult | t.Tests.TestobjectSerializable:
                deep_spec = params.deep
                if deep_spec is None:
                    return result_value
                match result_value:
                    case m.BaseModel() | Mapping():
                        pass
                    case _:
                        failure = (
                            "Deep matching requires dict or model, got "
                            f"{type(result_value).__name__}"
                        )
                        raise AssertionError(params.msg or failure)
                deep_input: (
                    m.BaseModel | t.MappingKV[str, t.Tests.TestobjectSerializable]
                )
                if isinstance(result_value, m.BaseModel):
                    deep_input = result_value
                else:
                    try:
                        deep_input = t.Tests.TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER.validate_python(
                            result_value
                        )
                    except c.ValidationError as exc:
                        failure = f"Deep matching payload is invalid: {exc}"
                        raise AssertionError(params.msg or failure) from exc
                match_result = FlextTestsPayloadUtilities.deep_match(
                    deep_input, deep_spec
                )
                if not match_result.matched:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_DEEP_PATH_FAILED.format(
                            path=match_result.path, reason=match_result.reason
                        )
                    )
                return result_value


__all__: list[str] = ["FlextTestsMatchersResultMixin"]
