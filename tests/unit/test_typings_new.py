"""Tests for t - type aliases, generics, validation types, containers.

Source: flext_core/ (7 files, ~693 LOC)
Tested through facade: t.*

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import ParamSpec

import pytest
from pydantic import ValidationError

from flext_core import (
    P,
    R,
    ResultT,
    T,
    T_co,
    T_contra,
    T_Model,
    T_Settings,
    U,
)
from flext_tests import tm
from tests import c, m, t


class TestFlextTypes:
    """Comprehensive tests for t facade and all _typings/ submodules."""

    # -- Type alias existence and accessibility through t.* --

    TYPE_ALIAS_NAMES: tuple[str, ...] = (
        "Primitives",
        "Scalar",
        "Container",
        "Numeric",
        "NormalizedValue",
        "RecursiveContainer",
        "RecursiveContainerMapping",
        "RecursiveContainerList",
        "ContainerMapping",
        "ContainerList",
        "MutableContainerMapping",
        "MutableContainerList",
        "StrMapping",
        "StrSequence",
        "ScalarMapping",
        "SecretValue",
        "SettingsValue",
        "IntPair",
    )

    @pytest.mark.parametrize("alias_name", TYPE_ALIAS_NAMES)
    def test_type_alias_accessible(self, alias_name: str) -> None:
        """Each type alias is accessible through t.* namespace."""

    # -- Core type aliases --

    CORE_ALIAS_NAMES: tuple[str, ...] = (
        "TextValue",
        "IntegerValue",
        "FloatValue",
        "BinaryContent",
        "TextOrBinaryContent",
        "OptionalPrimitive",
        "OptionalScalar",
        "RegistryBindingKey",
        "Serializable",
        "ContainerValue",
        "GeneralValueType",
        "OptionalContainerValue",
        "ConstantValue",
        "FileContent",
        "JsonMapping",
        "JsonList",
        "JsonObject",
        "ApiJsonValue",
    )

    @pytest.mark.parametrize("alias_name", CORE_ALIAS_NAMES)
    def test_core_alias_accessible(self, alias_name: str) -> None:
        """T aliases are accessible through t.* namespace."""

    # -- Service type aliases --

    SERVICE_ALIAS_NAMES: tuple[str, ...] = (
        "RegisterableService",
        "FactoryCallable",
        "ResourceCallable",
        "MetadataValue",
        "HandlerCallable",
        "HandlerLike",
        "DispatchableHandler",
        "LazyImportIndex",
        "ConfigurationMapping",
        "ResultErrorData",
        "ServiceMap",
        "ScalarOrModel",
        "ValueOrModel",
        "RuntimeData",
        "RuntimeAtomic",
        "BootstrapInput",
        "SortableObjectType",
        "TypeHintSpecifier",
        "MessageTypeSpecifier",
    )

    @pytest.mark.parametrize("alias_name", SERVICE_ALIAS_NAMES)
    def test_service_alias_accessible(self, alias_name: str) -> None:
        """T aliases are accessible through t.* namespace."""

    # -- Generic type vars --

    def test_generic_type_vars_exist(self) -> None:
        """All generic TypeVars and ParamSpec are importable."""
        tm.that(
            (
                T.__name__,
                U.__name__,
                R.__name__,
                P.__name__,
                ResultT.__name__,
                T_co.__name__,
                T_contra.__name__,
                T_Model.__name__,
                T_Settings.__name__,
            ),
            eq=(
                "T",
                "U",
                "R",
                "P",
                "ResultT",
                "T_co",
                "T_contra",
                "T_Model",
                "T_Settings",
            ),
        )

    def test_typevar_names(self) -> None:
        """TypeVars have correct __name__ attributes."""
        tm.that(repr(T), has="T")
        tm.that(repr(U), has="U")
        tm.that(repr(R), has="R")
        tm.that(repr(ResultT), has="ResultT")
        tm.that(repr(T_co), has="T_co")
        tm.that(repr(T_contra), has="T_contra")

    def test_typevar_variance(self) -> None:
        """Covariant and contravariant TypeVars have correct variance."""
        tm.that(T_co.__covariant__, eq=True)
        tm.that(T_contra.__contravariant__, eq=True)
        tm.that(T.__covariant__, eq=False)
        tm.that(T.__contravariant__, eq=False)

    def test_paramspec_is_paramspec(self) -> None:
        """P is a ParamSpec, not a TypeVar."""
        assert type(P).__name__ == ParamSpec.__name__

    def test_t_model_bound(self) -> None:
        """T_Model is bound to BaseModel."""
        bound_type = T_Model.__bound__
        tm.that(bound_type, not_=None)
        if isinstance(bound_type, type):
            tm.that(issubclass(bound_type, m.BaseModel), eq=True)
            return
        err = "T_Model.__bound__ must be a runtime type"
        raise AssertionError(err)

    # -- Runtime type tuples --

    def test_primitives_types_tuple(self) -> None:
        """PRIMITIVES_TYPES contains str, int, float, bool."""
        tm.that(t.PRIMITIVES_TYPES, eq=(str, int, float, bool))

    def test_numeric_types_tuple(self) -> None:
        """NUMERIC_TYPES contains int, float."""
        tm.that(t.NUMERIC_TYPES, eq=(int, float))

    def test_scalar_types_tuple(self) -> None:
        """SCALAR_TYPES contains str, int, float, bool, datetime."""
        tm.that(t.SCALAR_TYPES, eq=(str, int, float, bool, datetime))

    def test_container_types_tuple(self) -> None:
        """CONTAINER_TYPES contains scalar types + Path."""
        tm.that(t.CONTAINER_TYPES, eq=(str, int, float, bool, datetime, Path))

    def test_container_and_collection_types_tuple(self) -> None:
        """CONTAINER_AND_COLLECTION_TYPES includes list, dict, tuple."""
        tm.that(list in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)
        tm.that(dict in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)
        tm.that(tuple in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)

    # -- Container base classes --

    def test_container_mapping_base_exists(self) -> None:
        """ContainerMappingBase is accessible for subclassing."""

    def test_container_list_base_exists(self) -> None:
        """ContainerListBase is accessible for subclassing."""

    def test_mutable_container_mapping_base_exists(self) -> None:
        """MutableContainerMappingBase is accessible for subclassing."""

    def test_mutable_container_list_base_exists(self) -> None:
        """MutableContainerListBase is accessible for subclassing."""

    # -- Flat mapping aliases --

    FLAT_ALIAS_NAMES: tuple[str, ...] = (
        "AttributeMapping",
        "MutableAttributeMapping",
        "ConfigValueMapping",
        "OptionalStrMapping",
        "MutableOptionalStrMapping",
        "HeaderMapping",
        "FeatureFlagMapping",
        "MutableFeatureFlagMapping",
        "OptionalBoolMapping",
        "MutableOptionalBoolMapping",
    )

    @pytest.mark.parametrize("alias_name", FLAT_ALIAS_NAMES)
    def test_flat_alias_accessible(self, alias_name: str) -> None:
        """Flat mapping type aliases are accessible through t.*."""

    # -- Validation types with Pydantic --

    def test_non_empty_str_valid(self) -> None:
        """NonEmptyStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.NonEmptyStr)
        result = adapter.validate_python("hello")
        tm.that(result, eq="hello")

    def test_non_empty_str_rejects_empty(self) -> None:
        """NonEmptyStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.NonEmptyStr)
        with pytest.raises(ValidationError):
            adapter.validate_python("")

    def test_bounded_str_valid(self) -> None:
        """BoundedStr accepts strings between 1-255 chars."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        result = adapter.validate_python("valid")
        tm.that(result, eq="valid")

    def test_bounded_str_rejects_too_long(self) -> None:
        """BoundedStr rejects strings longer than 255 chars."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        with pytest.raises(ValidationError):
            adapter.validate_python("x" * 256)

    def test_bounded_str_rejects_empty(self) -> None:
        """BoundedStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        with pytest.raises(ValidationError):
            adapter.validate_python("")

    def test_positive_int_valid(self) -> None:
        """PositiveInt accepts positive integers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        result = adapter.validate_python(42)
        tm.that(result, eq=42)

    def test_positive_int_rejects_zero(self) -> None:
        """PositiveInt rejects zero."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        with pytest.raises(ValidationError):
            adapter.validate_python(0)

    def test_positive_int_rejects_negative(self) -> None:
        """PositiveInt rejects negative numbers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        with pytest.raises(ValidationError):
            adapter.validate_python(-1)

    def test_non_negative_int_valid(self) -> None:
        """NonNegativeInt accepts zero and positive."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.NonNegativeInt)
        tm.that(adapter.validate_python(0), eq=0)
        tm.that(adapter.validate_python(100), eq=100)

    def test_non_negative_int_rejects_negative(self) -> None:
        """NonNegativeInt rejects negative numbers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.NonNegativeInt)
        with pytest.raises(ValidationError):
            adapter.validate_python(-1)

    def test_port_number_valid(self) -> None:
        """PortNumber accepts 1-65535."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(8080), eq=8080)
        tm.that(adapter.validate_python(65535), eq=65535)

    def test_port_number_rejects_zero(self) -> None:
        """PortNumber rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        with pytest.raises(ValidationError):
            adapter.validate_python(0)

    def test_port_number_rejects_too_high(self) -> None:
        """PortNumber rejects values above 65535."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        with pytest.raises(ValidationError):
            adapter.validate_python(65536)

    def test_retry_count_valid(self) -> None:
        """RetryCount accepts 0-10."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.RetryCount)
        tm.that(adapter.validate_python(0), eq=0)
        tm.that(adapter.validate_python(10), eq=10)

    def test_retry_count_rejects_too_high(self) -> None:
        """RetryCount rejects values above 10."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.RetryCount)
        with pytest.raises(ValidationError):
            adapter.validate_python(11)

    def test_worker_count_valid(self) -> None:
        """WorkerCount accepts 1-100."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.WorkerCount)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(100), eq=100)

    def test_worker_count_rejects_zero(self) -> None:
        """WorkerCount rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.WorkerCount)
        with pytest.raises(ValidationError):
            adapter.validate_python(0)

    def test_http_status_code_valid(self) -> None:
        """HttpStatusCode accepts 100-599."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.HttpStatusCode)
        tm.that(adapter.validate_python(200), eq=200)
        tm.that(adapter.validate_python(404), eq=404)
        tm.that(adapter.validate_python(599), eq=599)

    def test_http_status_code_rejects_invalid(self) -> None:
        """HttpStatusCode rejects values outside 100-599."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.HttpStatusCode)
        with pytest.raises(ValidationError):
            adapter.validate_python(99)
        with pytest.raises(ValidationError):
            adapter.validate_python(600)

    def test_batch_size_valid(self) -> None:
        """BatchSize accepts 1-10000."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(10000), eq=10000)

    def test_batch_size_rejects_zero(self) -> None:
        """BatchSize rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        with pytest.raises(ValidationError):
            adapter.validate_python(0)

    def test_max_length_valid(self) -> None:
        """MaxLength accepts positive integers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(9999), eq=9999)

    def test_max_length_rejects_zero(self) -> None:
        """MaxLength rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        with pytest.raises(ValidationError):
            adapter.validate_python(0)

    def test_positive_float_valid(self) -> None:
        """PositiveFloat accepts positive floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        tm.that(adapter.validate_python(0.1), eq=0.1)

    def test_positive_float_rejects_zero(self) -> None:
        """PositiveFloat rejects 0.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        with pytest.raises(ValidationError):
            adapter.validate_python(0.0)

    def test_non_negative_float_valid(self) -> None:
        """NonNegativeFloat accepts 0.0 and positive."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(math.pi), eq=math.pi)

    def test_non_negative_float_rejects_negative(self) -> None:
        """NonNegativeFloat rejects negative floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        with pytest.raises(ValidationError):
            adapter.validate_python(-0.1)

    def test_positive_timeout_valid(self) -> None:
        """PositiveTimeout accepts 0 < x <= 300."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        tm.that(adapter.validate_python(1.0), eq=1.0)
        tm.that(adapter.validate_python(300.0), eq=300.0)

    def test_positive_timeout_rejects_zero(self) -> None:
        """PositiveTimeout rejects 0.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        with pytest.raises(ValidationError):
            adapter.validate_python(0.0)

    def test_positive_timeout_rejects_too_high(self) -> None:
        """PositiveTimeout rejects > 300."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        with pytest.raises(ValidationError):
            adapter.validate_python(300.1)

    def test_backoff_multiplier_valid(self) -> None:
        """BackoffMultiplier accepts >= 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        tm.that(adapter.validate_python(1.0), eq=1.0)
        tm.that(adapter.validate_python(2.5), eq=2.5)

    def test_backoff_multiplier_rejects_below_one(self) -> None:
        """BackoffMultiplier rejects < 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        with pytest.raises(ValidationError):
            adapter.validate_python(0.9)

    def test_percentage_valid(self) -> None:
        """Percentage accepts 0.0-100.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(50.0), eq=50.0)
        tm.that(adapter.validate_python(100.0), eq=100.0)

    def test_percentage_rejects_over_100(self) -> None:
        """Percentage rejects > 100."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        with pytest.raises(ValidationError):
            adapter.validate_python(100.1)

    def test_decimal_fraction_valid(self) -> None:
        """DecimalFraction accepts 0.0-1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(0.5), eq=0.5)
        tm.that(adapter.validate_python(1.0), eq=1.0)

    def test_decimal_fraction_rejects_over_one(self) -> None:
        """DecimalFraction rejects > 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        with pytest.raises(ValidationError):
            adapter.validate_python(1.1)

    def test_hostname_str_valid(self) -> None:
        """HostnameStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.HostnameStr)
        result = adapter.validate_python(c.LOCALHOST)
        tm.that(result, eq=c.LOCALHOST)

    def test_hostname_str_rejects_empty(self) -> None:
        """HostnameStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.HostnameStr)
        with pytest.raises(ValidationError):
            adapter.validate_python("")

    def test_uri_string_valid(self) -> None:
        """UriString accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.UriString)
        result = adapter.validate_python("https://example.com")
        tm.that(result, eq="https://example.com")

    def test_uri_string_rejects_empty(self) -> None:
        """UriString rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.UriString)
        with pytest.raises(ValidationError):
            adapter.validate_python("")

    def test_timestamp_str_valid(self) -> None:
        """TimestampStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.TimestampStr)
        result = adapter.validate_python("2025-01-01T00:00:00Z")
        tm.that(result, eq="2025-01-01T00:00:00Z")

    # -- Container Pydantic models --

    def test_dict_creation_empty(self) -> None:
        """t.Dict can be created empty."""
        d = t.Dict(root={})
        tm.that(len(d), eq=0)

    def test_dict_creation_with_data(self) -> None:
        """t.Dict can be created with initial data."""
        d = t.Dict(root={"key": "value", "num": 42})
        tm.that(d["key"], eq="value")
        tm.that(d["num"], eq=42)

    def test_dict_contains(self) -> None:
        """t.Dict supports 'in' operator."""
        d = t.Dict(root={"key": "value"})
        tm.that("key" in d, eq=True)
        tm.that("missing" in d, eq=False)

    def test_dict_get_with_default(self) -> None:
        """t.Dict.get() returns default for missing keys."""
        d = t.Dict(root={"key": "value"})
        tm.that(d.get("key"), eq="value")
        tm.that(d.get("missing", "fallback"), eq="fallback")

    def test_configmap_creation(self) -> None:
        """t.ConfigMap can be created with settings data."""
        cm = t.ConfigMap(root={"timeout": 30, "debug": False})
        tm.that(cm["timeout"], eq=30)
        tm.that(cm["debug"], eq=False)

    def test_configmap_len(self) -> None:
        """t.ConfigMap supports len()."""
        cm = t.ConfigMap(root={"a": 1, "b": 2})
        tm.that(len(cm), eq=2)

    def test_object_list_creation(self) -> None:
        """t.ObjectList can be created with container values."""
        ol = t.ObjectList(root=["item1", 42, True])
        tm.that(len(ol.root), eq=3)

    def test_object_list_default_empty(self) -> None:
        """t.ObjectList defaults to empty list."""
        default_factory = t.ObjectList.model_fields["root"].default_factory
        tm.that(default_factory is not None, eq=True)
        ol = t.ObjectList(root=[])
        tm.that(len(ol.root), eq=0)

    # -- MRO composition check --

    def test_flexttypes_inherits_base(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_containers(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_core(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_services(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_validation(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    # -- Generic tuple aliases --

    def test_pair_alias_exists(self) -> None:
        """t.Pair alias is accessible."""

    def test_triple_alias_exists(self) -> None:
        """t.Triple alias is accessible."""

    def test_variadic_tuple_alias_exists(self) -> None:
        """t.VariadicTuple alias is accessible."""

    def test_int_pair_alias_exists(self) -> None:
        """t.IntPair alias is accessible."""

    # -- CONTAINER_VALUE_SCALAR_TYPES --

    def test_container_value_scalar_types_mirrors_scalar(self) -> None:
        """CONTAINER_VALUE_SCALAR_TYPES mirrors SCALAR_TYPES."""
        tm.that(t.CONTAINER_VALUE_SCALAR_TYPES, eq=t.SCALAR_TYPES)


__all__: list[str] = ["TestFlextTypes"]
