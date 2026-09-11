"""Universal 'that' matcher utilities for flext-tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from flext_core import u
from flext_tests import c, m, p, r, t
from flext_tests._utilities._matchers._assertions import (
    FlextTestsMatchersAssertionsMixin,
)
from flext_tests._utilities._matchers._containment import (
    FlextTestsMatchersContainmentMixin,
)
from flext_tests._utilities._matchers._typeguards import (
    FlextTestsMatchersTypeGuardsMixin,
)
from flext_tests._utilities.payload import FlextTestsPayloadUtilities


class FlextTestsMatchersThatMixin:
    """Fluent matcher assertions."""

    class Tests:
        """Test utility namespace."""

        class Matchers:
            """Matcher assertion helpers."""

            # mro-j47u: cls dispatch preserves overrides on the composed matcher MRO.
            @classmethod
            def _that_params(
                cls, kwargs: dict[str, t.Tests.MatcherKwargValue]
            ) -> tuple[
                m.Tests.ThatParams,
                t.Tests.MatcherKwargValue | None,
                t.Tests.MatcherKwargValue | None,
                t.Tests.MatcherKwargValue | None,
                t.Tests.MatcherKwargValue | None,
            ]:
                """Validate matcher kwargs and retain raw non-serializable values."""
                raw_eq = kwargs.get("eq") if "eq" in kwargs else None
                raw_ne = kwargs.get("ne") if "ne" in kwargs else None
                raw_has = kwargs.get("has") if "has" in kwargs else None
                raw_contains = kwargs.get("contains") if "contains" in kwargs else None
                try:
                    params = m.Tests.ThatParams.model_validate(kwargs)
                except c.EXC_BASIC_TYPE:
                    params = cls._filtered_params(kwargs)
                return params, raw_eq, raw_ne, raw_has, raw_contains

            @staticmethod
            def _filtered_params(
                kwargs: dict[str, t.Tests.MatcherKwargValue],
            ) -> m.Tests.ThatParams:
                """Validate kwargs after removing values Pydantic cannot serialize."""
                non_serializable_keys = frozenset({
                    "eq",
                    "ne",
                    "has",
                    "contains",
                    "lacks",
                    "excludes",
                })
                filtered_kwargs = {
                    key: val
                    for key, val in kwargs.items()
                    if key not in non_serializable_keys
                }
                try:
                    return m.Tests.ThatParams.model_validate(filtered_kwargs)
                except c.EXC_BASIC_TYPE as filtered_exc:
                    message = f"Parameter validation failed: {filtered_exc}"
                    raise ValueError(message) from filtered_exc

            @classmethod
            def _validate_declared_types(
                cls, value: p.AttributeProbe, params: m.Tests.ThatParams
            ) -> None:
                """Validate ``is_`` and ``not_`` against the original value."""
                value_type_name = type(value).__name__
                if params.is_ is not None:
                    cls._validate_is_type(value, params, value_type_name)
                if params.not_ is not None:
                    not_types = (
                        params.not_
                        if isinstance(params.not_, tuple)
                        else (params.not_,)
                    )
                    if any(
                        FlextTestsMatchersTypeGuardsMixin.matches_runtime_type(
                            value, forbidden_type
                        )
                        for forbidden_type in not_types
                    ):
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_TYPE_FAILED.format(
                                expected=f"not {params.not_}", actual=value_type_name
                            )
                        )

            @staticmethod
            def _validate_is_type(
                value: p.AttributeProbe,
                params: m.Tests.ThatParams,
                value_type_name: str,
            ) -> None:
                """Validate ``is_`` including FLEXT wrapper/model shortcuts."""
                is_types = (
                    params.is_ if isinstance(params.is_, tuple) else (params.is_,)
                )
                expected_types = tuple(
                    item for item in is_types if isinstance(item, type)
                )
                root_value = getattr(value, "root", None)
                is_mapping_wrapper = (
                    dict in expected_types
                    and isinstance(root_value, dict)
                    and value.__class__.__name__ == "Dict"
                )
                is_model_mapping = (
                    dict in expected_types
                    and isinstance(value, m.BaseModel)
                    and not isinstance(value, m.RootModel)
                )
                is_sequence_wrapper = (
                    list in expected_types
                    and isinstance(root_value, t.SEQUENCE_PAIR_TYPES)
                    and value_type_name == "ObjectList"
                )
                matches_declared_type = any(
                    FlextTestsMatchersTypeGuardsMixin.matches_runtime_type(
                        value, expected_type
                    )
                    for expected_type in expected_types
                )
                if (
                    matches_declared_type
                    or is_mapping_wrapper
                    or is_model_mapping
                    or is_sequence_wrapper
                ):
                    return
                type_error = c.Tests.ERR_TYPE_FAILED.format(
                    expected=params.is_, actual=value_type_name
                )
                raise AssertionError(params.msg or f"Assertion failed: {type_error}")

            @staticmethod
            def _is_type_only(
                params: m.Tests.ThatParams,
                raw_eq: t.Tests.MatcherKwargValue | None,
                raw_ne: t.Tests.MatcherKwargValue | None,
            ) -> bool:
                """Return whether only type checks were requested."""
                if params.is_ is None and params.not_ is None:
                    return False
                if raw_eq is not None or raw_ne is not None:
                    return False
                return all(
                    getattr(params, name) is None
                    for name in (
                        "ok",
                        "has",
                        "lacks",
                        "eq",
                        "ne",
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                        "none",
                        "empty",
                        "starts",
                        "ends",
                        "match",
                        "len",
                    )
                )

            @classmethod
            def _result_subject(
                cls, subject: p.AttributeProbe, params: m.Tests.ThatParams
            ) -> p.AttributeProbe:
                """Return the value to validate after result-aware unwrapping."""
                if not isinstance(subject, r):
                    return subject
                result_obj = subject
                if params.ok is not None:
                    return cls._ok_value(result_obj, params)
                if params.has is not None:
                    err = result_obj.error or ""
                    FlextTestsMatchersContainmentMixin.check_has_lacks(
                        err, params.has, None, params.msg, as_str=True
                    )
                    return err
                if result_obj.success:
                    return getattr(result_obj, "value", "")
                raise AssertionError(
                    params.msg or c.Tests.ERR_OK_FAILED.format(error=result_obj.error)
                )

            @staticmethod
            def _ok_value(
                result_obj: p.Result[p.AttributeProbe], params: m.Tests.ThatParams
            ) -> p.AttributeProbe:
                """Validate result ok/fail expectation and return success value."""
                if params.ok and not result_obj.success:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_OK_FAILED.format(error=result_obj.error)
                    )
                if not params.ok and result_obj.success:
                    value_str = str(result_obj.value)
                    raise AssertionError(
                        params.msg or c.Tests.ERR_FAIL_EXPECTED.format(value=value_str)
                    )
                return getattr(result_obj, "value", "") if result_obj.success else ""

            @staticmethod
            def _has_scalar_validation(params: m.Tests.ThatParams) -> bool:
                """Return whether scalar guard validation is requested."""
                return any(
                    getattr(params, name) is not None
                    for name in (
                        "eq",
                        "ne",
                        "gt",
                        "gte",
                        "lt",
                        "lte",
                        "none",
                        "empty",
                        "starts",
                        "ends",
                        "match",
                    )
                )

            @classmethod
            def _validate_scalar(
                cls,
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
                raw_eq: t.Tests.MatcherKwargValue | None,
                raw_ne: t.Tests.MatcherKwargValue | None,
                kwargs: Mapping[str, t.Tests.MatcherKwargValue],
            ) -> None:
                """Validate scalar predicates."""
                if not cls._has_scalar_validation(params):
                    return
                chk_payload = (
                    None
                    if params.none is True
                    and isinstance(subject_payload, str)
                    and not subject_payload
                    else subject_payload
                )
                eq_value = raw_eq if "eq" in kwargs else params.eq
                ne_value = raw_ne if "ne" in kwargs else params.ne
                eq_payload, ne_payload = (
                    FlextTestsMatchersTypeGuardsMixin.prepare_eq_ne_payloads(
                        subject_payload,
                        eq_value,
                        ne_value,
                        msg=params.msg,
                        default_msg=(
                            f"Assertion failed: {subject_payload!r} "
                            "did not satisfy constraints"
                        ),
                    )
                )
                guard = m.GuardCheckSpec.model_validate({
                    "eq": FlextTestsPayloadUtilities.to_normalized_value(eq_payload)
                    if eq_payload is not None
                    else None,
                    "ne": FlextTestsPayloadUtilities.to_normalized_value(ne_payload)
                    if ne_payload is not None
                    else None,
                    "gt": params.gt,
                    "gte": params.gte,
                    "lt": params.lt,
                    "lte": params.lte,
                    "none": params.none,
                    "empty": params.empty,
                    "starts": params.starts,
                    "ends": params.ends,
                })
                chk_plain: t.GuardInput | None = (
                    None
                    if chk_payload is None
                    else FlextTestsPayloadUtilities.to_normalized_value(chk_payload)
                )
                if not u.chk(chk_plain, guard):
                    raise AssertionError(
                        params.msg
                        or (
                            f"Assertion failed: {subject_payload!r} "
                            "did not satisfy constraints"
                        )
                    )
                if (
                    params.match is not None
                    and isinstance(subject_payload, str)
                    and params.match.search(subject_payload) is None
                ):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_NOT_MATCHES.format(
                            text=subject_payload, pattern=params.match.pattern
                        )
                    )

            @staticmethod
            def _validate_common(
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
                *,
                effective_has: p.AttributeProbe | None,
            ) -> None:
                """Validate containment and length predicates."""
                FlextTestsMatchersContainmentMixin.check_has_lacks(
                    subject_payload, effective_has, params.lacks, params.msg
                )
                if params.len is not None:
                    FlextTestsMatchersAssertionsMixin.assert_len_match(
                        payload=subject_payload,
                        sized=subject_payload,
                        length_spec=params.len,
                        msg=params.msg,
                    )

            @staticmethod
            def _sequence_value(
                subject_payload: t.Tests.TestobjectSerializable,
            ) -> t.SequenceOf[t.Tests.TestobjectSerializable]:
                """Validate and normalize a sequence payload."""
                if not isinstance(subject_payload, t.SEQUENCE_PAIR_TYPES):
                    return ()
                try:
                    sequence_adapter = t.Tests.TESTOBJECT_SERIALIZABLE_SEQUENCE_ADAPTER
                    validated: t.SequenceOf[t.Tests.TestobjectSerializable] = (
                        sequence_adapter.validate_python(subject_payload)
                    )
                except c.ValidationError:
                    return ()
                else:
                    return validated

            @staticmethod
            def _validate_sequence_edges(
                seq_value: t.SequenceOf[t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate first/last sequence predicates."""
                if params.first is not None:
                    if not seq_value:
                        raise AssertionError(
                            params.msg or "Sequence is empty, cannot check first"
                        )
                    if seq_value[0] != params.first:
                        raise AssertionError(
                            params.msg
                            or (
                                f"First item: expected {params.first!r}, "
                                f"got {seq_value[0]!r}"
                            )
                        )
                if params.last is not None:
                    if not seq_value:
                        raise AssertionError(
                            params.msg or "Sequence is empty, cannot check last"
                        )
                    if seq_value[-1] != params.last:
                        raise AssertionError(
                            params.msg
                            or (
                                f"Last item: expected {params.last!r}, "
                                f"got {seq_value[-1]!r}"
                            )
                        )

            @classmethod
            def _validate_sequence(
                cls,
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate sequence-specific predicates."""
                seq_value = cls._sequence_value(subject_payload)
                if not seq_value and not isinstance(
                    subject_payload, t.SEQUENCE_PAIR_TYPES
                ):
                    return
                cls._validate_sequence_edges(seq_value, params)
                cls._validate_sequence_quantifiers(seq_value, params)
                cls._validate_sequence_order(seq_value, params)

            @classmethod
            def _validate_sequence_quantifiers(
                cls,
                seq_value: t.SequenceOf[t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate all_/any_ sequence predicates."""
                if params.all_ is not None:
                    cls._validate_all(seq_value, params)
                if params.any_ is not None:
                    cls._validate_any(seq_value, params)

            @staticmethod
            def _validate_all(
                seq_value: t.SequenceOf[t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate that all sequence items match a predicate/type."""
                if isinstance(params.all_, type):
                    all_type = params.all_
                    if all(isinstance(item, all_type) for item in seq_value):
                        return
                    failed_idx = next(
                        (
                            index
                            for index, item in enumerate(seq_value)
                            if not isinstance(item, all_type)
                        ),
                        None,
                    )
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_ALL_ITEMS_FAILED.format(index=failed_idx)
                    )
                if callable(params.all_) and not all(
                    params.all_(FlextTestsPayloadUtilities.to_payload(item))
                    for item in seq_value
                ):
                    failed_idx = next(
                        (
                            index
                            for index, item in enumerate(list(seq_value))
                            if not params.all_(
                                FlextTestsPayloadUtilities.to_payload(item)
                            )
                        ),
                        None,
                    )
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_ALL_ITEMS_FAILED.format(index=failed_idx)
                    )

            @staticmethod
            def _validate_any(
                seq_value: t.SequenceOf[t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate that any sequence item matches a predicate/type."""
                if isinstance(params.any_, type):
                    any_type = params.any_
                    if not any(isinstance(item, any_type) for item in seq_value):
                        raise AssertionError(params.msg or c.Tests.ERR_ANY_ITEMS_FAILED)
                    return
                if callable(params.any_) and not any(
                    params.any_(FlextTestsPayloadUtilities.to_payload(item))
                    for item in seq_value
                ):
                    raise AssertionError(params.msg or c.Tests.ERR_ANY_ITEMS_FAILED)

            @classmethod
            def _validate_sequence_order(
                cls,
                seq_value: t.SequenceOf[t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate sorted/unique sequence predicates."""
                sorted_param = params.sorted
                if sorted_param is not None:
                    value_list = list(seq_value)
                    if sorted_param is True:
                        sorted_list = sorted(value_list, key=cls._default_sort_key)
                        if value_list != sorted_list:
                            raise AssertionError(params.msg or "Sequence is not sorted")
                    elif callable(sorted_param):

                        def user_sort_key(
                            item: t.Tests.TestobjectSerializable,
                        ) -> t.StrPair:
                            return cls._comparable_key(sorted_param, item)

                        sorted_list = sorted(value_list, key=user_sort_key)
                        if value_list != sorted_list:
                            raise AssertionError(
                                params.msg or "Sequence is not sorted by key function"
                            )
                if (
                    params.unique is not None
                    and params.unique
                    and len(seq_value) != len(set(seq_value))
                ):
                    raise AssertionError(
                        params.msg or "Sequence contains duplicate items"
                    )

            @staticmethod
            def _default_sort_key(item: object) -> t.StrPair:
                """Return a deterministic key for heterogeneous matcher values."""
                return type(item).__name__, str(item)

            @staticmethod
            def _comparable_key(
                user_key_fn: Callable[[t.Tests.Testobject], t.Tests.Testobject],
                item: t.Tests.TestobjectSerializable,
            ) -> t.StrPair:
                """Wrap user key to return comparable tuple."""
                result = user_key_fn(FlextTestsPayloadUtilities.to_payload(item))
                return (str(type(result).__name__), str(result))

            @staticmethod
            def _mapping_value(
                subject_payload: t.Tests.TestobjectSerializable,
            ) -> t.MappingKV[str, t.Tests.TestobjectSerializable]:
                """Validate and normalize a mapping payload."""
                if not isinstance(subject_payload, Mapping):
                    return dict[str, t.Tests.TestobjectSerializable]()
                try:
                    validated: t.MappingKV[str, t.Tests.TestobjectSerializable] = (
                        t.Tests.TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER.validate_python(
                            subject_payload
                        )
                    )
                except c.ValidationError:
                    return dict[str, t.Tests.TestobjectSerializable]()
                else:
                    return validated

            @classmethod
            def _validate_mapping(
                cls,
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate mapping-specific predicates."""
                mapping_value = cls._mapping_value(subject_payload)
                if not mapping_value and not isinstance(subject_payload, Mapping):
                    return
                if params.keys is not None:
                    missing = set(params.keys) - set(mapping_value.keys())
                    if missing:
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_KEYS_MISSING.format(keys=list(missing))
                        )
                if params.lacks_keys is not None:
                    present = set(params.lacks_keys) & set(mapping_value.keys())
                    if present:
                        raise AssertionError(
                            params.msg
                            or c.Tests.ERR_KEYS_EXTRA.format(keys=list(present))
                        )
                if params.values is not None:
                    value_list = list(mapping_value.values())
                    for expected_val in params.values:
                        if expected_val not in value_list:
                            raise AssertionError(
                                params.msg
                                or (
                                    f"Expected value {expected_val!r} "
                                    "not found in mapping"
                                )
                            )
                if params.kv is not None:
                    cls._validate_kv(mapping_value, params)

            @staticmethod
            def _validate_kv(
                mapping_value: t.MappingKV[str, t.Tests.TestobjectSerializable],
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate key-value mapping predicates."""
                kv_items: t.SequenceOf[tuple[p.AttributeProbe, p.AttributeProbe]] = ()
                match params.kv:
                    case (key, expected_val):
                        kv_items = ((key, expected_val),)
                    case Mapping() as mapping_kv:
                        kv_items = tuple(mapping_kv.items())
                    case _:
                        kv_items = ()
                for key, expected_val in kv_items:
                    match key:
                        case str():
                            pass
                        case _:
                            raise AssertionError(
                                params.msg
                                or f"Mapping key must be str, got {type(key).__name__}"
                            )
                    if key not in mapping_value:
                        raise AssertionError(
                            params.msg or f"Key {key!r} not found in mapping"
                        )
                    actual_obj = mapping_value[key]
                    if actual_obj != expected_val:
                        raise AssertionError(
                            params.msg
                            or (
                                f"Key {key!r}: expected {expected_val!r}, "
                                f"got {actual_obj!r}"
                            )
                        )

            @classmethod
            def _validate_attrs(
                cls, subject: p.AttributeProbe, params: m.Tests.ThatParams
            ) -> None:
                """Validate attrs/methods/attr_eq predicates."""
                if params.attrs is not None:
                    attr_list: t.StrSequence = (
                        [params.attrs]
                        if isinstance(params.attrs, str)
                        else list(params.attrs)
                    )
                    for attr in attr_list:
                        if not hasattr(subject, attr):
                            raise AssertionError(
                                params.msg or f"Object missing attribute: {attr}"
                            )
                if params.methods is not None:
                    method_list: t.StrSequence = (
                        [params.methods]
                        if isinstance(params.methods, str)
                        else list(params.methods)
                    )
                    for method in method_list:
                        if not hasattr(subject, method):
                            raise AssertionError(
                                params.msg or f"Object missing method: {method}"
                            )
                        match getattr(subject, method):
                            case method_value if callable(method_value):
                                pass
                            case _:
                                raise AssertionError(
                                    params.msg
                                    or f"Object attribute {method} is not callable"
                                )
                if params.attr_eq is not None:
                    cls._validate_attr_eq(subject, params)

            @staticmethod
            def _validate_attr_eq(
                subject: p.AttributeProbe, params: m.Tests.ThatParams
            ) -> None:
                """Validate attribute equality predicates."""
                attr_items: t.SequenceOf[tuple[p.AttributeProbe, p.AttributeProbe]] = ()
                match params.attr_eq:
                    case (attr, expected_val):
                        attr_items = ((attr, expected_val),)
                    case Mapping() as attr_mapping:
                        attr_items = tuple(attr_mapping.items())
                    case _:
                        attr_items = ()
                for attr, expected_val in attr_items:
                    match attr:
                        case str():
                            pass
                        case _:
                            raise AssertionError(
                                params.msg
                                or (
                                    "Attribute name must be str, "
                                    f"got {type(attr).__name__}"
                                )
                            )
                    if not hasattr(subject, attr):
                        raise AssertionError(
                            params.msg or f"Object missing attribute: {attr}"
                        )
                    actual_val = getattr(subject, attr)
                    if actual_val != expected_val:
                        raise AssertionError(
                            params.msg
                            or (
                                f"Attribute {attr}: expected {expected_val!r}, "
                                f"got {actual_val!r}"
                            )
                        )

            # NOTE (multi-agent, mro-wkii.17 / agent: codex): combine runtime
            # assertion with generic narrowing so all analyzers retain T.
            @staticmethod
            def not_none[T](value: T | None, *, msg: str | None = None) -> T:
                """Assert and return a non-None value without changing identity."""
                if value is None:
                    raise AssertionError(msg or "Expected a non-None value")
                return value

            @classmethod
            def that(cls, value: object, **kwargs: t.Tests.MatcherKwargValue) -> None:
                """Assert a value against universal matcher constraints."""
                params, raw_eq, raw_ne, raw_has, raw_contains = cls._that_params(kwargs)
                if "eq" in kwargs and kwargs["eq"] is None and params.none is None:
                    params = params.model_copy(update={"none": True})
                if "ne" in kwargs and kwargs["ne"] is None and params.none is None:
                    params = params.model_copy(update={"none": False})
                cls._validate_declared_types(value, params)
                if cls._is_type_only(params, raw_eq, raw_ne):
                    return
                subject = cls._result_subject(value, params)
                subject_payload = FlextTestsPayloadUtilities.to_payload(subject)
                cls._validate_scalar(subject_payload, params, raw_eq, raw_ne, kwargs)
                effective_has = (
                    raw_has
                    if raw_has is not None
                    else raw_contains
                    if raw_contains is not None
                    else params.has
                )
                cls._validate_common(
                    subject_payload, params, effective_has=effective_has
                )
                cls._validate_sequence(subject_payload, params)
                cls._validate_mapping(subject_payload, params)
                cls._validate_attrs(subject, params)
                cls._validate_deep(subject_payload, params)
                cls._validate_rule_sets(subject, subject_payload, params)

            @staticmethod
            def _validate_deep(
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate deep structural constraints."""
                if params.deep is None:
                    return
                match subject_payload:
                    case m.BaseModel():
                        deep_value = subject_payload
                    case dict():
                        mapping_adapter = (
                            t.Tests.TESTOBJECT_SERIALIZABLE_MAPPING_ADAPTER
                        )
                        deep_value = mapping_adapter.validate_python(subject_payload)
                    case _:
                        raise AssertionError(
                            params.msg
                            or (
                                "Deep matching requires dict or model, got "
                                f"{type(subject_payload).__name__}"
                            )
                        )
                match_result = FlextTestsPayloadUtilities.deep_match(
                    deep_value, params.deep
                )
                if not match_result.matched:
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_DEEP_PATH_FAILED.format(
                            path=match_result.path, reason=match_result.reason
                        )
                    )

            @staticmethod
            def _validate_rule_sets(
                subject: p.AttributeProbe,
                subject_payload: t.Tests.TestobjectSerializable,
                params: m.Tests.ThatParams,
            ) -> None:
                """Validate path, item, attribute, and predicate rule sets."""
                if params.paths is not None:
                    FlextTestsMatchersThatMixin.apply_path_rules(
                        subject_payload, params.paths, inherited_msg=params.msg
                    )
                if params.items is not None:
                    FlextTestsMatchersThatMixin.apply_item_rules(
                        subject_payload, params.items, inherited_msg=params.msg
                    )
                if params.attrs_match is not None:
                    FlextTestsMatchersThatMixin.apply_attribute_rules(
                        subject, params.attrs_match, inherited_msg=params.msg
                    )
                if params.where is not None and not params.where(subject_payload):
                    raise AssertionError(
                        params.msg
                        or c.Tests.ERR_PREDICATE_FAILED.format(value=subject_payload)
                    )

    @staticmethod
    def _rule_kwargs(rule: object) -> dict[str, t.Tests.MatcherKwargValue]:
        parsed = m.Tests.MatchRule.parse(rule)
        return {
            key: getattr(parsed, key)
            for key in type(parsed).model_fields
            if key in parsed.model_fields_set or getattr(parsed, key) is not None
        }

    @classmethod
    def _apply_rule(
        cls,
        subject: t.Tests.TestobjectSerializable | m.BaseModel | p.AttributeProbe,
        rule: m.Tests.MatchRule,
        *,
        inherited_msg: str | None = None,
    ) -> None:
        kwargs = cls._rule_kwargs(rule)
        if inherited_msg is not None and "msg" not in kwargs:
            kwargs["msg"] = inherited_msg
        if not hasattr(cls.Tests.Matchers, "that"):
            message = "Matcher rule runner missing"
            raise AssertionError(message)
        cls.Tests.Matchers.that(subject, **kwargs)

    @staticmethod
    def _extract_path_value(
        subject: t.Tests.TestobjectSerializable
        | m.BaseModel
        | t.MappingKV[str, t.Tests.TestobjectSerializable],
        path: str,
    ) -> t.Tests.TestobjectSerializable:
        match subject:
            case m.BaseModel() | Mapping():
                pass
            case _:
                message = (
                    "Path assertions require dict or model, got "
                    f"{type(subject).__name__}"
                )
                raise AssertionError(message)
        extracted = u.extract(FlextTestsPayloadUtilities.to_config_map(subject), path)
        if extracted.failure:
            raise AssertionError(
                c.Tests.ERR_SCOPE_PATH_NOT_FOUND.format(
                    path=path, error=extracted.error
                )
            )
        return FlextTestsPayloadUtilities.to_payload(extracted.value)

    @classmethod
    def apply_path_rules(
        cls,
        subject: t.Tests.TestobjectSerializable
        | m.BaseModel
        | t.MappingKV[str, t.Tests.TestobjectSerializable],
        rules: Mapping[str, m.Tests.MatchRule],
        *,
        inherited_msg: str | None = None,
    ) -> None:
        for path, rule in rules.items():
            try:
                cls._apply_rule(
                    cls._extract_path_value(subject, path),
                    rule,
                    inherited_msg=inherited_msg,
                )
            except AssertionError as exc:
                raise AssertionError(
                    inherited_msg or f"Path rule '{path}' failed: {exc}"
                ) from exc

    @classmethod
    def apply_item_rules(
        cls,
        subject: t.Tests.TestobjectSerializable
        | t.SequenceOf[t.Tests.TestobjectSerializable],
        rules: Sequence[m.Tests.MatchRule] | Mapping[str | int, m.Tests.MatchRule],
        *,
        inherited_msg: str | None = None,
    ) -> None:
        match subject:
            case Sequence() if not isinstance(subject, t.STR_BINARY_TYPES):
                sequence_value = list(subject)
            case _:
                raise AssertionError(
                    inherited_msg
                    or (
                        "Item assertions require a sequence, got "
                        f"{type(subject).__name__}"
                    )
                )
        match rules:
            case Sequence():
                for index, rule in enumerate(rules):
                    cls._apply_rule(
                        sequence_value[index], rule, inherited_msg=inherited_msg
                    )
                return
            case Mapping():
                pass
            case _:
                raise AssertionError(
                    inherited_msg
                    or "Item assertions must be a sequence or selector mapping"
                )
        for selector, rule in rules.items():
            if selector in {"*", "all"}:
                for item in sequence_value:
                    cls._apply_rule(item, rule, inherited_msg=inherited_msg)
                continue
            target_index = (
                0
                if selector == "first"
                else -1
                if selector == "last"
                else int(selector)
            )
            cls._apply_rule(
                sequence_value[target_index], rule, inherited_msg=inherited_msg
            )

    @staticmethod
    def _resolve_attribute_path(
        subject: p.AttributeProbe, attr_path: str
    ) -> p.AttributeProbe:
        current: p.AttributeProbe = subject
        for segment in attr_path.split("."):
            if isinstance(current, Mapping) and segment in current:
                current = current[segment]
                continue
            if not hasattr(current, segment):
                message = f"Object missing attribute path: {attr_path}"
                raise AssertionError(message)
            current = getattr(current, segment)
        return current

    @classmethod
    def apply_attribute_rules(
        cls,
        subject: p.AttributeProbe,
        rules: Mapping[str, m.Tests.MatchRule],
        *,
        inherited_msg: str | None = None,
    ) -> None:
        for attr_path, rule in rules.items():
            try:
                cls._apply_rule(
                    FlextTestsPayloadUtilities.to_payload(
                        cls._resolve_attribute_path(subject, attr_path)
                    ),
                    rule,
                    inherited_msg=inherited_msg,
                )
            except AssertionError as exc:
                raise AssertionError(
                    inherited_msg or f"Attribute rule '{attr_path}' failed: {exc}"
                ) from exc


__all__: list[str] = ["FlextTestsMatchersThatMixin"]
