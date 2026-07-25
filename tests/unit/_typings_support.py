"""Shared typing facade test constants."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.typings import t

TYPE_ALIAS_NAMES: t.VariadicTuple[str] = (
    "Primitives",
    "Scalar",
    "Numeric",
    "JsonValue",
    "JsonMapping",
    "JsonList",
    "StrMapping",
    "StrSequence",
    "ScalarMapping",
    "SecretValue",
    "SettingsValue",
    "IntPair",
)
CORE_ALIAS_NAMES: t.VariadicTuple[str] = (
    "StrictStr",
    "StrictInt",
    "StrictFloat",
    "StrictBytes",
    "TextOrBinaryContent",
    "RegistryBindingKey",
    "JsonValue",
    "FileContent",
    "JsonMapping",
    "JsonList",
)
SERVICE_ALIAS_NAMES: t.VariadicTuple[str] = (
    "ConfigurationMapping",
    "DispatchableHandler",
    "FactoryCallable",
    "HandlerCallable",
    "JsonPayload",
    "JsonValue",
    "MessageTypeSpecifier",
    "RegisterableService",
    "ResourceCallable",
    "JsonMapping",
    "ScalarOrModel",
    "ServiceMap",
    "SortableObjectType",
    "TypeHintSpecifier",
)
PUBLIC_ALIAS_NAMES: t.VariadicTuple[str] = (
    TYPE_ALIAS_NAMES + CORE_ALIAS_NAMES + SERVICE_ALIAS_NAMES
)
FLAT_ALIAS_NAMES: t.VariadicTuple[str] = (
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
