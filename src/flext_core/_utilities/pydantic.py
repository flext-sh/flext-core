"""Pydantic v2 runtime utilities and validators exported via FlextUtilities.

Field helpers, validators, type adapters, and JSON helpers.

Architecture: Abstraction boundary - utilities layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from pathlib import Path
from types import EllipsisType
from typing import Annotated, TypeAliasType, get_args, get_origin

from pydantic import (
    AfterValidator,
    BaseModel,
    PlainSerializer,
    PlainValidator,
    SkipValidation,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_serializer,
    model_serializer,
    validate_call,
    with_config,
)
from pydantic_core import from_json, to_json, to_jsonable_python
from typing_extensions import TypeForm

from flext_core import _models, c, p, t
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._utilities import (
    FlextUtilitiesBase,
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
)
from flext_core.flextlazy_native_export import FlextTypingNativeExport


class FlextUtilitiesPydantic(
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
):
    """Runtime utilities: field helpers, validators, type adapters, JSON handlers.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use u.* / up.* instead.
    """

    Field = FlextTypingNativeExport(FlextModelsPydantic.Field)
    SkipValidation = SkipValidation

    AfterValidator = AfterValidator
    PlainValidator = PlainValidator
    WrapValidator = WrapValidator
    PlainSerializer = PlainSerializer
    WrapSerializer = WrapSerializer

    # Deprecated-signature window: consumers outside the upload line still read
    # these from u; the pass-through warns at call time and dies at census zero.
    field_validator = FlextTypingNativeExport(
        warnings.deprecated(
            "compat: u.field_validator -> m.field_validator; "
            "fix: decorate with m.field_validator"
        )(FlextModelsPydantic.field_validator)
    )
    model_validator = FlextTypingNativeExport(
        warnings.deprecated(
            "compat: u.model_validator -> m.model_validator; "
            "fix: decorate with m.model_validator"
        )(FlextModelsPydantic.model_validator)
    )

    computed_field = FlextTypingNativeExport(computed_field)
    field_serializer = FlextTypingNativeExport(field_serializer)
    model_serializer = FlextTypingNativeExport(model_serializer)
    validate_call = FlextTypingNativeExport(validate_call)
    with_config = FlextTypingNativeExport(with_config)

    from_json = FlextTypingNativeExport(from_json)
    to_json = FlextTypingNativeExport(to_json)
    to_jsonable_python = FlextTypingNativeExport(to_jsonable_python)

    @staticmethod
    def normalize_to_container(value: p.GuardInput) -> t.JsonPayload:
        """Preserve model identity and use native JSON for supported data shapes.

        Returns:
            The original ordinary model or a validated runtime data value.

        Raises:
            TypeError: If the value has no supported container representation.

        """
        models = _models.FlextModelsContainers
        root_model = isinstance(
            value, (models.ConfigMap, models.Dict, models.ObjectList)
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
            or isinstance(value, (*c.SCALAR_TYPES, Mapping, AbstractSet))
            or (
                isinstance(value, Sequence) and not isinstance(value, c.STR_BYTES_TYPES)
            )
        ):
            return FlextUtilitiesPydantic.normalize_to_json_value(source)
        message = c.ERR_RUNTIME_CONTAINER_INVALID_TYPE.format(
            type_name=type(value).__name__,
        )
        raise TypeError(message)

    @staticmethod
    def _json_data[OutputT: t.JsonValue](
        value: p.GuardInput | p.ModelInput,
        adapter: t.ValueAdapter[OutputT],
    ) -> OutputT:
        if isinstance(value, p.HasModelDump):
            return adapter.validate_python(value.model_dump(mode="json"), strict=True)
        payload = (
            _models.FlextModelsContainers.ConfigMap.model_validate(value).root
            if isinstance(value, Mapping)
            else value
        )
        return adapter.validate_python(to_jsonable_python(payload), strict=True)

    @staticmethod
    def normalize_to_json_value(value: p.GuardInput | p.ModelInput) -> t.JsonValue:
        """Serialize through the native SDK and validate the JSON value contract.

        Returns:
            A validated JSON scalar, object or array.

        """
        return FlextUtilitiesPydantic._json_data(
            value, FlextUtilitiesTypingAdapterJson.json_value_adapter()
        )

    @staticmethod
    def normalize_to_json_mapping(value: p.GuardInput | p.ModelInput) -> t.JsonMapping:
        """Serialize through the same SDK pipeline and require a JSON object.

        Returns:
            The validated JSON mapping.

        """
        return FlextUtilitiesPydantic._json_data(
            value, FlextUtilitiesTypingAdapterJson.json_dict_adapter()
        )

    @staticmethod
    def normalize_model_input_mapping(
        value: p.ConfigModelInput | None,
    ) -> t.JsonMapping | None:
        """Preserve absent model input and validate every supplied object shape.

        Returns:
            The JSON mapping, or None for absent input.

        """
        if value is None:
            return None
        payload = (
            value.model_dump(mode="json")
            if isinstance(value, p.HasModelDump)
            else value
        )
        parsed = _models.FlextModelsContainers.ConfigMap.model_validate(payload)
        return FlextUtilitiesPydantic.normalize_to_json_mapping(parsed)

    @staticmethod
    def normalize_metadata_input_mapping(
        value: p.MetadataInput,
    ) -> t.JsonMapping | None:
        """Validate string-keyed metadata before serializing its values.

        Returns:
            The normalized mapping, or None for absent metadata.

        """
        if value is None:
            return None
        return FlextUtilitiesPydantic.normalize_to_json_mapping(value)

    @staticmethod
    def validate_metadata_attributes(value: p.MetadataInput) -> t.JsonMapping:
        """Validate attribute keys without stringifying unsupported values.

        Returns:
            Validated public attributes, excluding absent values.

        Raises:
            ValueError: If an attribute key is reserved for internal metadata.

        """
        normalized = FlextUtilitiesPydantic.normalize_metadata_input_mapping(value)
        if normalized is None:
            return {}
        for key in normalized:
            if key.startswith("_"):
                message = c.ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED.format(key=key)
                raise ValueError(message)
        return FlextUtilitiesTypingAdapterJson.json_mapping_adapter().validate_python({
            key: item for key, item in normalized.items() if item is not None
        })

    @staticmethod
    def validate_metadata_model_input[ModelT: BaseModel](
        value: p.MetadataInput,
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
            c.FIELD_ATTRIBUTES: {} if attributes is None else dict(attributes)
        })

    @staticmethod
    def _empty_collection_seed[ContractT](contract: TypeForm[ContractT]) -> bytes:
        """Select a native JSON seed without discarding the original contract.

        Returns:
            The empty object or array input for the collection shape.

        Raises:
            TypeError: The contract has no supported collection shape.

        """
        current: t.TypeFormSpecifier = contract
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
            resolved, (*c.STR_BINARY_TYPES, BaseModel)
        ):
            if issubclass(resolved, Mapping):
                return c.EMPTY_COLLECTION_JSON_SEEDS["object"]
            if issubclass(resolved, (Sequence, AbstractSet)):
                return c.EMPTY_COLLECTION_JSON_SEEDS["array"]
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
        adapter: t.ValueAdapter[ContractT] | None = None
        seed = b""

        def factory() -> ContractT:
            nonlocal adapter, seed
            if adapter is None:
                resolved_contract = resolve_contract()
                seed = FlextUtilitiesPydantic._empty_collection_seed(resolved_contract)
                native_adapter = FlextUtilitiesBase.type_adapter(
                    resolved_contract,
                    config=_models.FlextModelsBase.ArbitraryTypesModel.model_config,
                )
                adapter = native_adapter
            return adapter.validate_json(seed)

        return factory


__all__: tuple[str, ...] = ("FlextUtilitiesPydantic",)
