"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from pathlib import Path
from types import EllipsisType
from typing import Annotated, TypeAliasType, get_args, get_origin

from pydantic import BaseModel, validate_call, with_config
from pydantic_core import from_json, to_json, to_jsonable_python
from typing_extensions import TypeForm

import flext_core
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities import (
    FlextUtilitiesBase,
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
)
from flext_core.flextlazy_native_export import FlextTypingNativeExport


class FlextUtilitiesPydantic(
    FlextModelsPydantic,
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
):
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    Extends the models pydantic part (same re-exports, one lineage): the
    facade collision law sees a single declaring root per pydantic name.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    # Re-exports inherited from FlextModelsPydantic (Field, SkipValidation,
    # AfterValidator, PlainValidator, WrapValidator, PlainSerializer,
    # computed_field, model_serializer, field_validator, model_validator):
    # single declaring root per pydantic name keeps the facade collision law
    # green; the u.field_validator/u.model_validator pass-throughs retired at
    # census zero (fleet, merges and ai-hub).

    validate_call = FlextTypingNativeExport(validate_call)
    with_config = FlextTypingNativeExport(with_config)

    from_json = FlextTypingNativeExport(from_json)
    to_json = FlextTypingNativeExport(to_json)
    to_jsonable_python = FlextTypingNativeExport(to_jsonable_python)

    @staticmethod
    def normalize_to_container(
        value: flext_core.p.GuardInput,
    ) -> flext_core.t.JsonPayload:
        """Preserve model identity and use native JSON for supported data shapes.

        Returns:
            The original ordinary model or a validated runtime data value.

        Raises:
            TypeError: If the value has no supported container representation.

        """
        root_model = isinstance(
            value, (flext_core.m.ConfigMap, flext_core.m.Dict, flext_core.m.ObjectList)
        )
        if value is None:
            return None
        if isinstance(value, BaseModel) and not root_model:
            return value
        if isinstance(value, Path):
            return str(value)
        source = value
        if (
            root_model
            or isinstance(value, (*flext_core.c.SCALAR_TYPES, Mapping, AbstractSet))
            or (
                isinstance(value, Sequence)
                and not isinstance(value, flext_core.c.STR_BYTES_TYPES)
            )
        ):
            return FlextUtilitiesPydantic.normalize_to_json_value(source)
        message = flext_core.c.ERR_RUNTIME_CONTAINER_INVALID_TYPE.format(
            type_name=type(value).__name__,
        )
        raise TypeError(message)

    @staticmethod
    def _json_data[OutputT: flext_core.t.JsonValue](
        value: flext_core.p.GuardInput | flext_core.p.ModelInput,
        adapter: flext_core.t.ValueAdapter[OutputT],
    ) -> OutputT:
        if isinstance(value, flext_core.p.HasModelDump):
            return adapter.validate_python(value.model_dump(mode="json"), strict=True)
        payload = (
            flext_core.m.ConfigMap.model_validate(value).root
            if isinstance(value, Mapping)
            else value
        )
        return adapter.validate_python(to_jsonable_python(payload), strict=True)

    @staticmethod
    def normalize_to_json_value(
        value: flext_core.p.GuardInput | flext_core.p.ModelInput,
    ) -> flext_core.t.JsonValue:
        """Serialize through the native SDK and validate the JSON value contract.

        Returns:
            A validated JSON scalar, object or array.

        """
        return FlextUtilitiesPydantic._json_data(
            value, FlextUtilitiesTypingAdapterJson.json_value_adapter()
        )

    @staticmethod
    def normalize_to_json_mapping(
        value: flext_core.p.GuardInput | flext_core.p.ModelInput,
    ) -> flext_core.t.JsonMapping:
        """Serialize through the same SDK pipeline and require a JSON object.

        Returns:
            The validated JSON mapping.

        """
        return FlextUtilitiesPydantic._json_data(
            value, FlextUtilitiesTypingAdapterJson.json_dict_adapter()
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: flext_core.p.ConfigModelInput | None,
    ) -> flext_core.t.JsonMapping | None:
        """Preserve absent model input and validate every supplied object shape.

        Returns:
            The JSON mapping, or None for absent input.

        """
        if value is None:
            return None
        payload = (
            value.model_dump(mode="json")
            if isinstance(value, flext_core.p.HasModelDump)
            else value
        )
        parsed = flext_core.m.ConfigMap.model_validate(payload)
        return FlextUtilitiesPydantic.normalize_to_json_mapping(parsed)

    @staticmethod
    def normalize_metadata_input_mapping(
        value: flext_core.p.MetadataInput,
    ) -> flext_core.t.JsonMapping | None:
        """Validate string-keyed metadata before serializing its values.

        Returns:
            The normalized mapping, or None for absent metadata.

        """
        if value is None:
            return None
        return FlextUtilitiesPydantic.normalize_to_json_mapping(value)

    @staticmethod
    def validate_metadata_model_input[ModelT: BaseModel](
        value: flext_core.p.MetadataInput,
        metadata_model: type[ModelT],
    ) -> ModelT:
        """Materialize the requested SDK model from validated metadata attributes.

        Returns:
            The original matching model or a validated metadata model.

        """
        if isinstance(value, metadata_model):
            return value
        attributes = FlextUtilitiesPydantic.normalize_metadata_input_mapping(value)
        return metadata_model.model_validate({
            flext_core.c.FIELD_ATTRIBUTES: {}
            if attributes is None
            else dict(attributes)
        })

    @staticmethod
    def _empty_collection_seed[ContractT](contract: TypeForm[ContractT]) -> bytes:
        """Select a native JSON seed without discarding the original contract.

        Returns:
            The empty object or array input for the collection shape.

        Raises:
            TypeError: The contract has no supported collection shape.

        """
        current: flext_core.t.TypeFormSpecifier = contract
        while True:
            origin = get_origin(current)
            if origin is Annotated:
                current = get_args(current)[0]
            elif isinstance(current, TypeAliasType):
                current = current.__value__
            elif isinstance(origin, TypeAliasType):
                current = origin.__value__
            else:
                break
        resolved = origin or current
        if isinstance(resolved, type) and not issubclass(
            resolved, (*flext_core.c.STR_BINARY_TYPES, BaseModel)
        ):
            if issubclass(resolved, Mapping):
                return flext_core.c.EMPTY_COLLECTION_JSON_SEEDS["object"]
            if issubclass(resolved, (Sequence, AbstractSet)):
                return flext_core.c.EMPTY_COLLECTION_JSON_SEEDS["array"]
        msg = f"empty default requires a collection contract; got {contract!r}"
        raise TypeError(msg)

    @staticmethod
    def empty[ContractT](
        contract: TypeForm[ContractT] | EllipsisType = ...,
        *,
        contract_factory: Callable[[], TypeForm[ContractT]] | None = None,
    ) -> Callable[[], ContractT]:
        """Build a lazy factory that validates a fresh empty input in full.

        Only the outer collection shape selects the JSON seed. The cached
        adapter retains the original contract, including aliases, element types
        and constraints; no JSON schema is requested for arbitrary element types.
        ``contract_factory`` resolves a type after its enclosing declaration
        exists. Exactly one contract source is required. Resolution and adapter
        construction happen on the first value, never while declaring a field.

        Args:
            contract: A collection contract already available at declaration.
            contract_factory: Resolve a contract once its declarations exist.

        Returns:
            A zero-argument factory with independent validated collection values.

        Raises:
            TypeError: Exactly one contract source must be supplied.

        """
        resolve_contract: Callable[[], TypeForm[ContractT]]
        if contract_factory is None:
            if isinstance(contract, EllipsisType):
                msg = "empty requires a collection contract or contract_factory"
                raise TypeError(msg)
            fixed_contract = contract

            def resolve_fixed_contract() -> TypeForm[ContractT]:
                return fixed_contract

            resolve_contract = resolve_fixed_contract
        elif isinstance(contract, EllipsisType):
            resolve_contract = contract_factory
        else:
            msg = "empty accepts either contract or contract_factory, not both"
            raise TypeError(msg)
        adapter: flext_core.t.ValueAdapter[ContractT] | None = None
        seed = b""

        def factory() -> ContractT:
            nonlocal adapter, seed
            if adapter is None:
                resolved_contract = resolve_contract()
                seed = FlextUtilitiesPydantic._empty_collection_seed(resolved_contract)
                native_adapter = FlextUtilitiesBase.type_adapter(
                    resolved_contract,
                    config=flext_core.m.ArbitraryTypesModel.model_config,
                )
                adapter = native_adapter
            return adapter.validate_json(seed)

        return factory


__all__: tuple[str, ...] = ("FlextUtilitiesPydantic",)
