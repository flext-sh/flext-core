"""Pydantic v2 base model types exported via FlextModels.

This module provides public aliases for pydantic v2 base model classes
that are used across the flext ecosystem. All projects consuming these
must extend from flext_core* instead of directly from pydantic.

Architecture: Abstraction boundary - models layer
Boundary: flext-core is sole owner of pydantic v2 integration. All other
projects receive pydantic model bases ONLY through public facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal, override

import pydantic
import pydantic_settings
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BeforeValidator,
    Discriminator,
    FailFast,
    GetPydanticSchema,
    PlainSerializer,
    PlainValidator,
    SerializeAsAny,
    SkipValidation,
    StringConstraints,
    ValidateAs,
    WrapSerializer,
    WrapValidator,
    json_schema as pydantic_json_schema,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticOmit, core_schema

from flext_core.flextlazy_native_export import FlextTypingNativeExport


class FlextModelsPydantic:
    """Public base model classes from pydantic v2.

    **NEVER import pydantic directly outside flext-core/src/.**
    Extend from these bases via m.* instead: p.BaseModel, m.RootModel

    Available model bases (accessible as m.MODEL_NAME):
        BaseModel: Pydantic v2 base for all data models with validation
        RootModel: Container model for single validated values/collections
    """

    Field = FlextTypingNativeExport(pydantic.Field)
    FieldInfo = FieldInfo

    PrivateAttr = FlextTypingNativeExport(pydantic.PrivateAttr)

    class JsonSchemaGenerator(pydantic_json_schema.GenerateJsonSchema):
        """JSON Schema generator that omits a shared omitted definition per field.

        pydantic drops a field whose schema raises ``PydanticOmit``
        (``SkipJsonSchema``), but ``definitions_schema`` catches only
        ``PydanticInvalidForJsonSchema``. A PEP 695 alias over
        ``SkipJsonSchema`` (``t.Port``, ``t.SkipJsonSchema``) used by two fields
        of one model becomes a shared definition, and its ``PydanticOmit``
        escaped ``model_json_schema`` (pydantic 2.14.3).

        Each definition is generated once, on its first reference or in
        declaration order, whichever comes first; one that omits itself makes
        every reference to it omit inside the referencing field, where pydantic
        already drops the field. Schemas without an omitted definition are
        unchanged.
        """

        _pending_definitions: dict[str, core_schema.CoreSchema]
        _started_refs: set[str]
        _omitted_refs: set[str]

        @override
        def generate(
            self,
            schema: core_schema.CoreSchema,
            mode: pydantic_json_schema.JsonSchemaMode = "validation",
        ) -> pydantic_json_schema.JsonSchemaValue:
            """Start one generation with no definition pending or omitted.

            Returns:
                The JSON Schema of ``schema``.

            """
            self._pending_definitions = {}
            self._started_refs = set()
            self._omitted_refs = set()
            return super().generate(schema, mode)

        @override
        def definitions_schema(
            self, schema: core_schema.DefinitionsSchema
        ) -> pydantic_json_schema.JsonSchemaValue:
            """Generate every definition once, then the wrapped schema.

            Returns:
                The JSON Schema of the wrapped schema.

            """
            for definition in schema["definitions"]:
                self._pending_definitions[self._definition_ref(definition)] = definition
            for ref in tuple(self._pending_definitions):
                self._generate_definition(ref)
            return self.generate_inner(schema["schema"])

        @override
        def definition_ref_schema(
            self, schema: core_schema.DefinitionReferenceSchema
        ) -> pydantic_json_schema.JsonSchemaValue:
            """Reference a definition, omitting the field when it omitted itself.

            Returns:
                The reference to the generated definition.

            Raises:
                PydanticOmit: The referenced definition omits itself.

            """
            ref = schema["schema_ref"]
            if ref in self._pending_definitions:
                self._generate_definition(ref)
            if ref in self._omitted_refs:
                raise PydanticOmit
            return super().definition_ref_schema(schema)

        def _generate_definition(self, ref: str) -> None:
            """Generate one pending definition once, recording its omission."""
            if ref in self._started_refs:
                return
            self._started_refs.add(ref)
            definition = self._pending_definitions[ref]
            try:
                self.generate_inner(definition)
            except PydanticOmit:
                self._omitted_refs.add(ref)
            except pydantic.PydanticInvalidForJsonSchema as exc:
                core_ref = pydantic_json_schema.CoreRef(ref)
                defs_ref = self.get_defs_ref((core_ref, self.mode))
                self._core_defs_invalid_for_json_schema[defs_ref] = exc

        @staticmethod
        def _definition_ref(definition: core_schema.CoreSchema) -> str:
            """The core reference every entry of a definitions schema carries.

            Returns:
                The definition's core reference.

            Raises:
                TypeError: The definitions entry carries no reference.

            """
            ref = definition.get("ref")
            if not isinstance(ref, str):
                msg = f"definitions entry without a reference: {definition['type']}"
                raise TypeError(msg)
            return ref

    class BaseModel(pydantic.BaseModel):
        """Canonical BaseModel exported through the FLEXT models facade."""

        @override
        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool | None = None,
            ref_template: str = pydantic_json_schema.DEFAULT_REF_TEMPLATE,
            schema_generator: type[pydantic_json_schema.GenerateJsonSchema]
            | None = None,
            mode: pydantic_json_schema.JsonSchemaMode = "validation",
            *,
            union_format: Literal["any_of", "primitive_type_array"] = "any_of",
        ) -> pydantic_json_schema.JsonSchemaValue:
            """JSON Schema through the FLEXT generator (shared omitted definitions).

            ``schema_generator`` absent selects ``JsonSchemaGenerator``, resolved
            here because a class body cannot name its enclosing facade.

            Returns:
                The JSON Schema of the model.

            """
            return super().model_json_schema(
                by_alias=by_alias,
                ref_template=ref_template,
                schema_generator=FlextModelsPydantic.JsonSchemaGenerator
                if schema_generator is None
                else schema_generator,
                mode=mode,
                union_format=union_format,
            )

    class BaseSettings(pydantic_settings.BaseSettings):
        """Canonical BaseSettings exported through the FLEXT models facade."""

    class EnvSettingsSource(pydantic_settings.EnvSettingsSource):
        """Canonical environment source exported through the FLEXT models facade."""

    class RootModel[RootValueT](pydantic.RootModel[RootValueT]):
        """Canonical RootModel exported through the FLEXT models facade."""

    class ConfigDict(pydantic.ConfigDict, total=False):
        """Canonical model configuration exported through the models facade."""

    class SettingsConfigDict(pydantic_settings.SettingsConfigDict, total=False):
        """Canonical settings configuration exported through the models facade."""

    ForceDecode = FlextTypingNativeExport(pydantic_settings.ForceDecode)

    field_validator = FlextTypingNativeExport(pydantic.field_validator)
    model_validator = FlextTypingNativeExport(pydantic.model_validator)
    field_serializer = FlextTypingNativeExport(pydantic.field_serializer)
    model_serializer = FlextTypingNativeExport(pydantic.model_serializer)
    computed_field = FlextTypingNativeExport(pydantic.computed_field)

    # Annotation constraints and tagged-union discrimination
    Discriminator = Discriminator
    StringConstraints = StringConstraints
    # Field alias declarations (runtime markers placed in ``Field`` arguments).
    AliasChoices = AliasChoices
    AliasPath = AliasPath

    # Annotation validators
    AfterValidator = AfterValidator
    BeforeValidator = BeforeValidator
    FailFast = FailFast
    type InstanceOf[T] = pydantic.InstanceOf[T]
    PlainValidator = PlainValidator
    ValidateAs = ValidateAs
    WrapValidator = WrapValidator
    SkipValidation = SkipValidation

    # Serializers
    PlainSerializer = PlainSerializer
    SerializeAsAny = SerializeAsAny
    WrapSerializer = WrapSerializer

    # Types of the pydantic final classes and protocols: annotation-only names.
    # ``u.type_adapter`` constructs the adapter this type describes.
    type FieldSerializationInfo = pydantic.FieldSerializationInfo

    type TypeAdapter[T] = pydantic.TypeAdapter[T]

    # Annotation marker that wraps a callable schema hook.
    GetPydanticSchema = GetPydanticSchema

    # Validation exception (re-exported so consumers avoid `import pydantic`)
    ValidationError = pydantic.ValidationError


__all__: list[str] = ["FlextModelsPydantic"]
