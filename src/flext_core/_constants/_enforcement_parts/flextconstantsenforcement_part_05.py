"""Rule text enforcement constants for FlextConstantsEnforcement (core rules)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from flext_core._constants._enforcement_data.loader import SMELL_RULES_TEXT

from .flextconstantsenforcement_part_09 import NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT

from collections.abc import Mapping

from flext_core._typings.base import FlextTypingBase as t


_BASE_ENFORCEMENT_RULES_TEXT: dict[str, t.StrPair] = {
    "no_any": ("Any is FORBIDDEN (detected recursively)", "Use a t.* type contract."),
    "no_bare_collection": (
        "bare {kind}[...] annotation FORBIDDEN",
        "Use {replacement}.",
    ),
    "no_mutable_default": (
        "mutable default {kind}() is FORBIDDEN",
        "Use m.Field(default_factory={kind}).",
    ),
    "no_raw_collections_field_default": (
        "Field(default_factory={kind}) conflicts with a read-only field contract",
        "Use the immutable equivalent (tuple, MappingProxyType, frozenset) or declare an explicit MutableSequence/MutableMapping/MutableSet contract when in-place mutation is part of the model API.",
    ),
    "no_str_none_empty": (
        'str | None with default="" is wrong',
        'Use str with default="" (None has no business meaning here).',
    ),
    "no_inline_union": (
        "complex inline union with {arms} arms",
        "Centralize as a t.* type alias in typings.py.",
    ),
    "missing_description": (
        "m.Field() missing description",
        'Provide description="...".',
    ),
    "no_v1_config": (
        "class Config is Pydantic v1",
        "Use model_config: ClassVar[ConfigDict] = ConfigDict(...).",
    ),
    "extra_missing": (
        'model_config missing extra="forbid"',
        "Inherit a configured FLEXT base (ArbitraryTypesModel, etc.).",
    ),
    "extra_wrong": (
        'model_config extra="{extra}" not allowed',
        "Use FlexibleModel or FlexibleInternalModel.",
    ),
    "value_not_frozen": (
        "value objects must be frozen=True",
        "Inherit from ImmutableValueModel or FrozenValueModel.",
    ),
    "const_mutable": (
        "mutable constant value FORBIDDEN",
        "Use frozenset, tuple, or MappingProxyType.",
    ),
    "const_lowercase": (
        "constant names must be UPPER_CASE",
        "Rename to UPPER_SNAKE_CASE.",
    ),
    "alias_any": ("Any in type alias FORBIDDEN", "Use t.* contracts."),
    "typeadapter_name": (
        'TypeAdapter "{name}" needs UPPER_CASE naming',
        'Rename to "ADAPTER_{upper_name}".',
    ),
    "utility_not_static": (
        "utility must be @staticmethod or @classmethod",
        "Utilities must be stateless.",
    ),
    "class_prefix": (
        'class name missing project prefix "{expected}"',
        'Rename to start with "{expected}".',
    ),
    "cross_strenum": ("StrEnum in wrong layer", "Move to constants (c.*)."),
    "cross_protocol": ("Protocol in wrong layer", "Move to protocols (p.*)."),
    "nested_mro": (
        'must be nested inside a "{expected}*" container class',
        'Wrap in a container whose name starts with "{expected}".',
    ),
    "proto_inner_kind": (
        "inner class must be Protocol / namespace / ABC",
        "Declare a Protocol subclass, namespace holder, or nominal contract.",
    ),
    "proto_not_runtime": (
        "Protocol must be @runtime_checkable",
        "Decorate the Protocol with @runtime_checkable.",
    ),
    "no_accessor_methods": (
        'accessor method "{name}" FORBIDDEN (AGENTS.md §3.1)',
        'Rename "{name}" to a domain verb ({suggestion}) or expose as a field/@u.computed_field.',
    ),
    "settings_inheritance": (
        '"{name}" must inherit FlextSettings (AGENTS.md §2.6)',
        "Add FlextSettings to the MRO; remove BaseModel/BaseSettings bases.",
    ),
    "cast_outside_core": (
        "cast() call in {file} is outside flext-core (AGENTS.md §3.2)",
        "Replace cast() with FlextResult narrowing or explicit isinstance().",
    ),
    "classvar_constant_outside_constants": (
        "Constant '{name}' declared in {module} (outside _constants)",
        "Move the constant to {suggested_target} and re-export via c.*.",
    ),
    "model_rebuild_call": (
        "model_rebuild() invocation in {file} (AGENTS.md §3.4)",
        "Resolve forward refs via proper imports / __future__ annotations.",
    ),
    "pass_through_wrapper": (
        'pass-through wrapper "{name}" in {file} (AGENTS.md §3.5)',
        "Inline the wrapper at every call site and delete the function.",
    ),
    "private_attr_probe": (
        '{probe}(obj, "{name}") probes private attribute in {file}',
        "Refactor the consumer to use the public surface (AGENTS.md §3.6).",
    ),
    **NAMESPACE_IMPORT_ENFORCEMENT_RULES_TEXT,
}


class FlextConstantsEnforcementRuleText:
    """Legacy problem/fix text indexed by enforcement tag."""

    ENFORCEMENT_RULES_TEXT: Final[Mapping[str, t.StrPair]] = MappingProxyType({
        **_BASE_ENFORCEMENT_RULES_TEXT,
        **SMELL_RULES_TEXT,
    })
    """Legacy: problem/fix text indexed by tag. Use m.EnforcementCatalog for new code."""


__all__: list[str] = ["FlextConstantsEnforcementRuleText"]
