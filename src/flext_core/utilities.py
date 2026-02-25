"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via staticmethod aliases from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.Mapper.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import overload

from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.checker import FlextUtilitiesChecker
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.conversion import FlextUtilitiesConversion
from flext_core._utilities.deprecation import FlextUtilitiesDeprecation
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enum import FlextUtilitiesEnum
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.pattern import FlextUtilitiesPattern
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.result_helpers import (
    ResultHelpers as FlextUtilitiesResultHelpers,
)
from flext_core._utilities.text import FlextUtilitiesText
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilities:
    """Unified facade for all FLEXT utility functionality.

    Runtime alias u exposes a flat namespace (staticmethod aliases from subclasses).
    Use direct methods only: u.get, u.parse, u.map, u.from_kwargs, u.batch, u.extract,
    u.warn_once, etc. No subdivided namespaces (no u.Mapper.* at call sites).
    Subprojects use their project u. Aliases/namespaces: MRO registration protocol only.

    Usage:
        from flext_core import u
        result = u.parse(value, int)
        value = u.get(data, "key")
        mapped = u.map(items, fn)
    """

    # === FACADE CLASSES - Real inheritance ===

    class Args(FlextUtilitiesArgs):
        """Args utility class - real inheritance."""

    class Cache(FlextUtilitiesCache):
        """Cache utility class - real inheritance."""

    class Checker(FlextUtilitiesChecker):
        """Checker utility class - real inheritance."""

    class Collection(FlextUtilitiesCollection):
        """Collection utility class - real inheritance."""

    class Configuration(FlextUtilitiesConfiguration):
        """Configuration utility class - real inheritance."""

    class Context(FlextUtilitiesContext):
        """Context utility class - real inheritance."""

    class Conversion(FlextUtilitiesConversion):
        """Conversion utility class - real inheritance."""

    class Deprecation(FlextUtilitiesDeprecation):
        """Deprecation utility class - real inheritance."""

    class Domain(FlextUtilitiesDomain):
        """Domain utility class - real inheritance."""

    class Enum(FlextUtilitiesEnum):
        """Enum utility class - real inheritance."""

    class Generators(FlextUtilitiesGenerators):
        """Generators utility class - real inheritance."""

    class Guards(FlextUtilitiesGuards):
        """Guards utility class - real inheritance."""

    class Mapper(FlextUtilitiesMapper):
        """Mapper utility class - real inheritance."""

    @staticmethod
    def mapper() -> type[FlextUtilitiesMapper]:
        """Return the Mapper class for backward-compatible u.mapper().get(...) calls."""
        return FlextUtilitiesMapper

    class Model(FlextUtilitiesModel):
        """Model utility class - real inheritance."""

    class Pagination(FlextUtilitiesPagination):
        """Pagination utility class - real inheritance."""

    class Parser(FlextUtilitiesParser):
        """Parser utility class - real inheritance."""

    class Pattern(FlextUtilitiesPattern):
        """Pattern utility class - real inheritance."""

    class Reliability(FlextUtilitiesReliability):
        """Reliability utility class - real inheritance."""

    class Text(FlextUtilitiesText):
        """Text utility class - real inheritance."""

    # =========================================================================
    # STATIC METHOD ALIASES - All from _utilities/*.py
    # =========================================================================

    # Args
    get_enum_params = staticmethod(FlextUtilitiesArgs.get_enum_params)
    parse_kwargs = staticmethod(FlextUtilitiesArgs.parse_kwargs)
    validated = staticmethod(FlextUtilitiesArgs.validated)
    validated_with_result = staticmethod(FlextUtilitiesArgs.validated_with_result)

    # Cache
    clear_object_cache = staticmethod(FlextUtilitiesCache.clear_object_cache)
    generate_cache_key = staticmethod(FlextUtilitiesCache.generate_cache_key)
    has_cache_attributes = staticmethod(FlextUtilitiesCache.has_cache_attributes)
    normalize_component = staticmethod(FlextUtilitiesCache.normalize_component)
    sort_dict_keys = staticmethod(FlextUtilitiesCache.sort_dict_keys)
    sort_key = staticmethod(FlextUtilitiesCache.sort_key)

    # Cast (use u.cast_generic or u.Mapper.cast_generic at call sites)
    cast_generic = staticmethod(FlextUtilitiesMapper.cast_generic)

    # Collection
    batch = staticmethod(FlextUtilitiesCollection.batch)
    chunk = staticmethod(FlextUtilitiesCollection.chunk)
    coerce_dict_to_bool = staticmethod(FlextUtilitiesCollection.coerce_dict_to_bool)
    coerce_dict_to_enum = staticmethod(FlextUtilitiesCollection.coerce_dict_to_enum)
    coerce_dict_to_float = staticmethod(FlextUtilitiesCollection.coerce_dict_to_float)
    coerce_dict_to_int = staticmethod(FlextUtilitiesCollection.coerce_dict_to_int)
    coerce_dict_to_str = staticmethod(FlextUtilitiesCollection.coerce_dict_to_str)
    coerce_dict_validator = staticmethod(FlextUtilitiesCollection.coerce_dict_validator)
    coerce_list_to_bool = staticmethod(FlextUtilitiesCollection.coerce_list_to_bool)
    coerce_list_to_enum = staticmethod(FlextUtilitiesCollection.coerce_list_to_enum)
    coerce_list_to_float = staticmethod(FlextUtilitiesCollection.coerce_list_to_float)
    coerce_list_to_int = staticmethod(FlextUtilitiesCollection.coerce_list_to_int)
    coerce_list_to_str = staticmethod(FlextUtilitiesCollection.coerce_list_to_str)
    coerce_list_validator = staticmethod(FlextUtilitiesCollection.coerce_list_validator)
    count = staticmethod(FlextUtilitiesCollection.count)
    filter = staticmethod(FlextUtilitiesCollection.filter)
    find = staticmethod(FlextUtilitiesCollection.find)
    first = staticmethod(FlextUtilitiesCollection.first)
    flatten = staticmethod(FlextUtilitiesCollection.flatten)
    group = staticmethod(FlextUtilitiesCollection.group)
    group_by = staticmethod(FlextUtilitiesCollection.group_by)
    last = staticmethod(FlextUtilitiesCollection.last)
    map = staticmethod(FlextUtilitiesCollection.map)
    merge = staticmethod(FlextUtilitiesCollection.merge)
    mul = staticmethod(FlextUtilitiesCollection.mul)
    extract_mapping_items = staticmethod(FlextUtilitiesCollection.extract_mapping_items)
    extract_callable_mapping = staticmethod(
        FlextUtilitiesCollection.extract_callable_mapping,
    )
    parse_mapping = staticmethod(FlextUtilitiesCollection.parse_mapping)
    parse_sequence = staticmethod(FlextUtilitiesCollection.parse_sequence)
    partition = staticmethod(FlextUtilitiesCollection.partition)
    process = staticmethod(FlextUtilitiesCollection.process)
    unique = staticmethod(FlextUtilitiesCollection.unique)

    # Checker
    compute_accepted_message_types = staticmethod(
        FlextUtilitiesChecker.compute_accepted_message_types,
    )

    # Configuration
    build_options_from_kwargs = staticmethod(
        FlextUtilitiesConfiguration.build_options_from_kwargs,
    )
    bulk_register = staticmethod(FlextUtilitiesConfiguration.bulk_register)
    create_settings_config = staticmethod(
        FlextUtilitiesConfiguration.create_settings_config,
    )
    get_parameter = staticmethod(FlextUtilitiesConfiguration.get_parameter)
    get_singleton = staticmethod(FlextUtilitiesConfiguration.get_singleton)
    register_factory = staticmethod(FlextUtilitiesConfiguration.register_factory)
    register_singleton = staticmethod(FlextUtilitiesConfiguration.register_singleton)
    resolve_env_file = staticmethod(FlextUtilitiesConfiguration.resolve_env_file)
    set_parameter = staticmethod(FlextUtilitiesConfiguration.set_parameter)
    set_singleton = staticmethod(FlextUtilitiesConfiguration.set_singleton)
    validate_config_class = staticmethod(
        FlextUtilitiesConfiguration.validate_config_class,
    )

    # Context
    clone_container = staticmethod(FlextUtilitiesContext.clone_container)
    clone_runtime = staticmethod(FlextUtilitiesContext.clone_runtime)
    create_datetime_proxy = staticmethod(FlextUtilitiesContext.create_datetime_proxy)
    create_dict_proxy = staticmethod(FlextUtilitiesContext.create_dict_proxy)
    create_str_proxy = staticmethod(FlextUtilitiesContext.create_str_proxy)

    # Conversion - use direct static method alias
    # Tests should use u.Conversion.conversion() directly for proper overload resolution
    conversion = staticmethod(FlextUtilitiesConversion.conversion)
    join = staticmethod(FlextUtilitiesConversion.join)
    normalize = staticmethod(FlextUtilitiesConversion.normalize)
    to_str = staticmethod(FlextUtilitiesConversion.to_str)
    to_str_list = staticmethod(FlextUtilitiesConversion.to_str_list)
    to_general_value_type = staticmethod(FlextUtilitiesConversion.to_general_value_type)
    to_flexible_value = staticmethod(FlextUtilitiesConversion.to_flexible_value)

    # Deprecation
    deprecated = staticmethod(FlextUtilitiesDeprecation.deprecated)
    deprecated_class = staticmethod(FlextUtilitiesDeprecation.deprecated_class)
    deprecated_parameter = staticmethod(FlextUtilitiesDeprecation.deprecated_parameter)
    warn_once = staticmethod(FlextUtilitiesDeprecation.warn_once)

    # Domain
    compare_entities_by_id = staticmethod(FlextUtilitiesDomain.compare_entities_by_id)
    compare_value_objects_by_value = staticmethod(
        FlextUtilitiesDomain.compare_value_objects_by_value,
    )
    hash_entity_by_id = staticmethod(FlextUtilitiesDomain.hash_entity_by_id)
    hash_value_object_by_value = staticmethod(
        FlextUtilitiesDomain.hash_value_object_by_value,
    )
    validate_entity_has_id = staticmethod(FlextUtilitiesDomain.validate_entity_has_id)
    validate_value_object_immutable = staticmethod(
        FlextUtilitiesDomain.validate_value_object_immutable,
    )

    # Enum
    coerce_validator = staticmethod(FlextUtilitiesEnum.coerce_validator)
    create_discriminated_union = staticmethod(
        FlextUtilitiesEnum.create_discriminated_union,
    )
    create_enum = staticmethod(FlextUtilitiesEnum.create_enum)
    get_enum_values = staticmethod(FlextUtilitiesEnum.get_enum_values)
    is_enum_member = staticmethod(FlextUtilitiesEnum.is_enum_member)
    is_member = staticmethod(FlextUtilitiesEnum.is_member)
    is_subset = staticmethod(FlextUtilitiesEnum.is_subset)
    members = staticmethod(FlextUtilitiesEnum.members)
    names = staticmethod(FlextUtilitiesEnum.names)
    parse_enum = staticmethod(FlextUtilitiesEnum.parse_enum)
    values = staticmethod(FlextUtilitiesEnum.values)

    # Generators
    create_dynamic_type_subclass = staticmethod(
        FlextUtilitiesGenerators.create_dynamic_type_subclass,
    )
    ensure_dict = staticmethod(FlextUtilitiesGenerators.ensure_dict)
    ensure_trace_context = staticmethod(FlextUtilitiesGenerators.ensure_trace_context)
    generate = staticmethod(FlextUtilitiesGenerators.generate)
    generate_datetime_utc = staticmethod(FlextUtilitiesGenerators.generate_datetime_utc)
    generate_iso_timestamp = staticmethod(
        FlextUtilitiesGenerators.generate_iso_timestamp,
    )
    generate_operation_id = staticmethod(FlextUtilitiesGenerators.generate_operation_id)
    generate_short_id = staticmethod(FlextUtilitiesGenerators.Random.generate_short_id)

    # Guards
    chk = staticmethod(FlextUtilitiesGuards.chk)
    empty = staticmethod(FlextUtilitiesGuards.empty)
    extract_mapping_or_none = staticmethod(FlextUtilitiesGuards.extract_mapping_or_none)
    guard = staticmethod(FlextUtilitiesGuards.guard)
    has = staticmethod(FlextUtilitiesGuards.has)
    in_ = staticmethod(FlextUtilitiesGuards.in_)
    is_configuration_dict = staticmethod(FlextUtilitiesGuards.is_configuration_dict)
    is_configuration_mapping = staticmethod(
        FlextUtilitiesGuards.is_configuration_mapping,
    )
    is_context = staticmethod(FlextUtilitiesGuards.is_context)
    is_dict_non_empty = staticmethod(FlextUtilitiesGuards.is_dict_non_empty)
    is_general_value_type = staticmethod(FlextUtilitiesGuards.is_general_value_type)
    is_handler_type = staticmethod(FlextUtilitiesGuards.is_handler_type)
    is_handler_callable = staticmethod(FlextUtilitiesGuards.is_handler_callable)
    is_list = staticmethod(FlextUtilitiesGuards.is_list)
    is_list_non_empty = staticmethod(FlextUtilitiesGuards.is_list_non_empty)
    is_pydantic_model = staticmethod(FlextUtilitiesGuards.is_pydantic_model)
    is_string_non_empty = staticmethod(FlextUtilitiesGuards.is_string_non_empty)
    is_type = staticmethod(FlextUtilitiesGuards.is_type)
    is_config_value = staticmethod(FlextUtilitiesGuards.is_config_value)
    is_mapping = staticmethod(FlextUtilitiesGuards.is_mapping)
    none_ = staticmethod(FlextUtilitiesGuards.none_)
    normalize_to_metadata_value = staticmethod(
        FlextUtilitiesGuards.normalize_to_metadata_value,
    )
    require_initialized = staticmethod(FlextUtilitiesGuards.require_initialized)

    # Mapper
    agg = staticmethod(FlextUtilitiesMapper.agg)
    build = staticmethod(FlextUtilitiesMapper.build)
    build_flags_dict = staticmethod(FlextUtilitiesMapper.build_flags_dict)
    collect_active_keys = staticmethod(FlextUtilitiesMapper.collect_active_keys)
    construct = staticmethod(FlextUtilitiesMapper.construct)
    convert_dict_to_json = staticmethod(FlextUtilitiesMapper.convert_dict_to_json)
    convert_list_to_json = staticmethod(FlextUtilitiesMapper.convert_list_to_json)
    convert_to_json_value = staticmethod(FlextUtilitiesMapper.convert_to_json_value)
    deep_eq = staticmethod(FlextUtilitiesMapper.deep_eq)
    ensure = staticmethod(FlextUtilitiesMapper.ensure)
    ensure_str = staticmethod(FlextUtilitiesMapper.ensure_str)
    ensure_str_or_none = staticmethod(FlextUtilitiesMapper.ensure_str_or_none)
    narrow_to_general_value_type = staticmethod(
        FlextUtilitiesMapper.narrow_to_general_value_type,
    )
    extract = staticmethod(FlextUtilitiesMapper.extract)
    fields = staticmethod(FlextUtilitiesMapper.fields)
    fields_multi = staticmethod(FlextUtilitiesMapper.fields_multi)
    filter_dict = staticmethod(FlextUtilitiesMapper.filter_dict)

    # get - with overloads for proper type inference
    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
    ) -> t.ConfigMapValue | None: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: str,
    ) -> str: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: bool,
    ) -> bool: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: int,
    ) -> int: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: float,
    ) -> float: ...

    @staticmethod
    @overload
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.ConfigMapValue | None,
    ) -> t.ConfigMapValue | None: ...

    @staticmethod
    def get(
        data: p.AccessibleData,
        key: str,
        *,
        default: t.ConfigMapValue | None = None,
    ) -> t.ConfigMapValue | None:
        """Unified get function for dict/object access with default."""
        return FlextUtilitiesMapper.get(data, key, default=default)

    invert_dict = staticmethod(FlextUtilitiesMapper.invert_dict)
    is_json_primitive = staticmethod(FlextUtilitiesMapper.is_json_primitive)
    key_by = staticmethod(FlextUtilitiesMapper.key_by)
    map_dict_keys = staticmethod(FlextUtilitiesMapper.map_dict_keys)
    normalize_context_values = staticmethod(
        FlextUtilitiesMapper.normalize_context_values,
    )
    omit = staticmethod(FlextUtilitiesMapper.omit)
    pick = staticmethod(FlextUtilitiesMapper.pick)
    pluck = staticmethod(FlextUtilitiesMapper.pluck)
    process_context_data = staticmethod(FlextUtilitiesMapper.process_context_data)
    prop = staticmethod(FlextUtilitiesMapper.prop)
    # NOTE: take has complex overloads - use u.Mapper.take() for type inference
    take = staticmethod(FlextUtilitiesMapper.take)
    transform = staticmethod(FlextUtilitiesMapper.transform)
    transform_values = staticmethod(FlextUtilitiesMapper.transform_values)

    # Model
    dump = staticmethod(FlextUtilitiesModel.dump)
    from_dict = staticmethod(FlextUtilitiesModel.from_dict)
    from_kwargs = staticmethod(FlextUtilitiesModel.from_kwargs)
    load = staticmethod(FlextUtilitiesModel.load)
    merge_defaults = staticmethod(FlextUtilitiesModel.merge_defaults)
    normalize_to_metadata = staticmethod(FlextUtilitiesModel.normalize_to_metadata)
    to_dict = staticmethod(FlextUtilitiesModel.to_dict)
    update = staticmethod(FlextUtilitiesModel.update)

    # Pagination
    build_pagination_response = staticmethod(
        FlextUtilitiesPagination.build_pagination_response,
    )
    extract_page_params = staticmethod(FlextUtilitiesPagination.extract_page_params)
    extract_pagination_config = staticmethod(
        FlextUtilitiesPagination.extract_pagination_config,
    )
    prepare_pagination_data = staticmethod(
        FlextUtilitiesPagination.prepare_pagination_data,
    )
    validate_pagination_params = staticmethod(
        FlextUtilitiesPagination.validate_pagination_params,
    )

    # Pattern
    match = staticmethod(FlextUtilitiesPattern.match)

    # Reliability
    calculate_delay = staticmethod(FlextUtilitiesReliability.calculate_delay)
    chain = staticmethod(FlextUtilitiesReliability.chain)
    compose = staticmethod(FlextUtilitiesReliability.compose)
    flow = staticmethod(FlextUtilitiesReliability.flow)
    flow_result = staticmethod(FlextUtilitiesReliability.flow_result)
    flow_through = staticmethod(FlextUtilitiesReliability.flow_through)
    fold_result = staticmethod(FlextUtilitiesReliability.fold_result)
    pipe = staticmethod(FlextUtilitiesReliability.pipe)
    retry = staticmethod(FlextUtilitiesReliability.retry)
    tap_result = staticmethod(FlextUtilitiesReliability.tap_result)
    then = staticmethod(FlextUtilitiesReliability.then)
    with_retry = staticmethod(FlextUtilitiesReliability.with_retry)
    with_timeout = staticmethod(FlextUtilitiesReliability.with_timeout)

    # Runtime
    is_dict_like = staticmethod(FlextRuntime.is_dict_like)
    is_list_like = staticmethod(FlextRuntime.is_list_like)
    normalize_to_general_value = staticmethod(FlextRuntime.normalize_to_general_value)
    runtime_normalize_to_metadata_value = staticmethod(
        FlextRuntime.normalize_to_metadata_value,
    )
    runtime_generate_datetime_utc = staticmethod(FlextRuntime.generate_datetime_utc)
    generate_id = staticmethod(FlextRuntime.generate_id)
    generate_prefixed_id = staticmethod(FlextRuntime.generate_prefixed_id)

    # Text
    clean_text = staticmethod(FlextUtilitiesText.clean_text)
    format_app_id = staticmethod(FlextUtilitiesText.format_app_id)
    safe_string = staticmethod(FlextUtilitiesText.safe_string)
    truncate_text = staticmethod(FlextUtilitiesText.truncate_text)

    # Parser - Direct access aliases (u.parse, u.convert, u.conv_*, u.norm_*)
    parse = staticmethod(FlextUtilitiesParser.parse)
    convert = staticmethod(FlextUtilitiesParser.convert)
    conv_str = staticmethod(FlextUtilitiesParser.conv_str)
    conv_int = staticmethod(FlextUtilitiesParser.conv_int)
    conv_str_list = staticmethod(FlextUtilitiesParser.conv_str_list)
    conv_str_list_truthy = staticmethod(FlextUtilitiesParser.conv_str_list_truthy)
    conv_str_list_safe = staticmethod(FlextUtilitiesParser.conv_str_list_safe)
    norm_str = staticmethod(FlextUtilitiesParser.norm_str)
    norm_list = staticmethod(FlextUtilitiesParser.norm_list)
    norm_join = staticmethod(FlextUtilitiesParser.norm_join)
    norm_in = staticmethod(FlextUtilitiesParser.norm_in)

    # Validation - Core

    # Validation/ResultHelpers
    any_ = staticmethod(FlextUtilitiesResultHelpers.any_)
    err = staticmethod(FlextUtilitiesResultHelpers.err)
    fail = staticmethod(FlextUtilitiesResultHelpers.fail)
    not_ = staticmethod(FlextUtilitiesResultHelpers.not_)
    ok = staticmethod(FlextUtilitiesResultHelpers.ok)
    or_ = staticmethod(FlextUtilitiesResultHelpers.or_)
    result_val = staticmethod(FlextUtilitiesResultHelpers.val)
    starts = staticmethod(FlextUtilitiesResultHelpers.starts)
    try_ = staticmethod(FlextUtilitiesResultHelpers.try_)
    val = staticmethod(FlextUtilitiesResultHelpers.val)
    vals = staticmethod(FlextUtilitiesResultHelpers.vals)
    vals_sequence = staticmethod(FlextUtilitiesResultHelpers.vals_sequence)


u = FlextUtilities

__all__ = [
    "FlextUtilities",
    "u",
]
