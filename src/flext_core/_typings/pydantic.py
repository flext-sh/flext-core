"""Pydantic v2 constrained and specialized types exported via FlextTypes.

Including: StrictStr, EmailStr, UUID*, HttpUrl, PositiveInt, etc.

Architecture: Abstraction boundary - typings layer

Type aliases use PEP 695 `type` statements for concrete and generic aliases.
Runtime callables remain plain attributes.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import GenericAlias, NoneType, UnionType
from typing import Annotated, TypeAliasType, TypeVar, Union, get_args, get_origin

import pydantic
import pydantic_core
from pydantic import JsonValue, json_schema as pydantic_json_schema
from pydantic_core import core_schema


class FlextTypesPydantic:
    """Constrained and specialized types: strict types, URL types, numeric types, etc.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use t.* instead.
    """

    # String constraints
    type ConfigDict = pydantic.ConfigDict
    type ComputedFieldInfo = pydantic.fields.ComputedFieldInfo
    type FieldInfo = pydantic.fields.FieldInfo
    type StrictStr = pydantic.StrictStr
    type EmailStr = pydantic.EmailStr
    type NameEmail = pydantic.NameEmail

    # Validation exception for annotation positions; the runtime class stays
    # on the m/c facades for except/isinstance (TypeAliasType is not catchable).
    type ValidationError = pydantic.ValidationError

    # Numeric constraints. The upstream ``con*`` factories are discouraged for
    # removal in Pydantic 3 (pydantic/types.py:167-174@v2.14.1); constraints are
    # ``Annotated[..., m.Field(...)]`` / ``StringConstraints`` aliases instead.
    type PositiveInt = pydantic.PositiveInt
    type NonNegativeInt = pydantic.NonNegativeInt
    type NegativeInt = pydantic.NegativeInt
    type NonPositiveInt = pydantic.NonPositiveInt
    type PositiveFloat = pydantic.PositiveFloat
    type NonNegativeFloat = pydantic.NonNegativeFloat
    type NegativeFloat = pydantic.NegativeFloat
    type NonPositiveFloat = pydantic.NonPositiveFloat
    type FiniteFloat = pydantic.FiniteFloat
    type ByteSize = pydantic.ByteSize

    # Strict types
    type StrictInt = pydantic.StrictInt
    type StrictFloat = pydantic.StrictFloat
    type StrictBool = pydantic.StrictBool
    type StrictBytes = pydantic.StrictBytes

    # Date and time types
    type AwareDatetime = pydantic.AwareDatetime
    type NaiveDatetime = pydantic.NaiveDatetime
    type FutureDate = pydantic.FutureDate
    type FutureDatetime = pydantic.FutureDatetime
    type PastDate = pydantic.PastDate
    type PastDatetime = pydantic.PastDatetime

    # URL and network types
    type AnyUrl = pydantic.AnyUrl
    type AnyHttpUrl = pydantic.AnyHttpUrl
    type AnyWebsocketUrl = pydantic.AnyWebsocketUrl
    type HttpUrl = pydantic.HttpUrl
    type WebsocketUrl = pydantic.WebsocketUrl
    type FileUrl = pydantic.FileUrl
    type FtpUrl = pydantic.FtpUrl
    type PostgresDsn = pydantic.PostgresDsn
    type MySQLDsn = pydantic.MySQLDsn
    type MariaDBDsn = pydantic.MariaDBDsn
    type CockroachDsn = pydantic.CockroachDsn
    type ClickHouseDsn = pydantic.ClickHouseDsn
    type MongoDsn = pydantic.MongoDsn
    type KafkaDsn = pydantic.KafkaDsn
    type RedisDsn = pydantic.RedisDsn
    type SnowflakeDsn = pydantic.SnowflakeDsn
    type NatsDsn = pydantic.NatsDsn

    # File system types
    type FilePath = pydantic.FilePath
    type DirectoryPath = pydantic.DirectoryPath
    type NewPath = pydantic.NewPath
    type SocketPath = pydantic.SocketPath

    # UUID types
    type UUID1 = pydantic.UUID1
    type UUID3 = pydantic.UUID3
    type UUID4 = pydantic.UUID4
    type UUID5 = pydantic.UUID5
    type UUID6 = pydantic.UUID6
    type UUID7 = pydantic.UUID7
    type UUID8 = pydantic.UUID8

    # Binary and encoding types
    type Base64Str = pydantic.Base64Str
    type Base64Bytes = pydantic.Base64Bytes
    type Base64UrlStr = pydantic.Base64UrlStr
    type Base64UrlBytes = pydantic.Base64UrlBytes
    type EncodedStr = pydantic.EncodedStr
    type EncodedBytes = pydantic.EncodedBytes

    # JSON and special types: generic type forms recognized by pydantic.
    type Json[T] = pydantic.Json[T]
    # JsonValue is also module-level so beartype can resolve forward references
    # emitted from aliases that flow through this class namespace. The class
    # member republishes the module-level pydantic alias: a divergent
    # redefinition here shadowed pydantic.JsonValue through every composed
    # facade with mutually incompatible unrollings (abstract Mapping/Sequence
    # here vs pydantic's concrete dict/list/tuple), so pydantic remains the
    # single source of truth for the JSON type surface.
    type JsonValue = pydantic.JsonValue
    type BaseModelType = pydantic.BaseModel
    # Payload unions admit existing models without constructing BaseModel from data.
    type BaseModelInstance = pydantic.InstanceOf[pydantic.BaseModel]
    # PEP 695 aliases, never bare class-scope assignments: a bare
    # ``BaseModel = pydantic.BaseModel`` is a variable to mypy, so it is not
    # valid as a type and breaks isinstance narrowing (mypy [unreachable]).
    type BaseModel = pydantic.BaseModel
    type TypeAdapter[T] = pydantic.TypeAdapter[T]
    type ImportString[T] = pydantic.ImportString[T]
    type InstanceOf[T] = pydantic.InstanceOf[T]
    type InstanceOfSequence[T] = pydantic.InstanceOf[Sequence[T]]
    type Secret[T] = pydantic.Secret[T]
    type SecretBytes = pydantic.SecretBytes

    class SecretStr(pydantic.SecretStr):
        """Secret string type, constructible and usable in annotations."""

    # IP types
    type IPvAnyAddress = pydantic.IPvAnyAddress
    type IPvAnyInterface = pydantic.IPvAnyInterface
    type IPvAnyNetwork = pydantic.IPvAnyNetwork

    # Constraint helper types (runtime markers / classes)
    StringConstraints = pydantic.StringConstraints
    UrlConstraints = pydantic.UrlConstraints

    # Validation error payload types
    type ErrorDetails = pydantic_core.ErrorDetails
    type ErrorType = core_schema.ErrorType
    type ErrorTypeInfo = pydantic_core.ErrorTypeInfo
    type InitErrorDetails = pydantic_core.InitErrorDetails

    # Annotation validators and serializers (runtime markers placed in
    # ``Annotated`` metadata); t owns every annotation marker type.
    AfterValidator = pydantic.AfterValidator
    BeforeValidator = pydantic.BeforeValidator
    PlainValidator = pydantic.PlainValidator
    WrapValidator = pydantic.WrapValidator
    PlainSerializer = pydantic.PlainSerializer
    WrapSerializer = pydantic.WrapSerializer
    GetPydanticSchema = pydantic.GetPydanticSchema

    # Annotation and alias helper types (runtime markers / classes)
    AliasGenerator = pydantic.AliasGenerator
    AliasChoices = pydantic.AliasChoices
    AliasPath = pydantic.AliasPath
    Discriminator = pydantic.Discriminator
    Tag = pydantic.Tag
    ValidateAs = pydantic.ValidateAs
    WithJsonSchema = pydantic.WithJsonSchema
    SkipJsonSchema = pydantic_json_schema.SkipJsonSchema
    SerializeAsAny = pydantic.SerializeAsAny
    AllowInfNan = pydantic.AllowInfNan
    Strict = pydantic.Strict
    FailFast = pydantic.FailFast

    # A port target as pydantic hands it over: a class, a PEP 695 alias, a
    # type parameter, a generic or union form, or None.
    type PortTarget = (
        type | TypeVar | TypeAliasType | GenericAlias | UnionType | NoneType | None
    )

    @staticmethod
    def port_instance_schema(
        target: PortTarget,
        bound: Mapping[TypeVar, PortTarget] | None = None,
    ) -> core_schema.CoreSchema:
        """Resolve a port target to the ``isinstance`` schema pydantic runs.

        Pydantic's ``InstanceOf`` needs one class and does not see through
        PEP 695 aliases (``t.LoggingWriteCompletion``, ``p.*`` aliases). This
        resolves aliases (substituting ``bound`` type parameters of an
        enclosing generic alias), generic origins, ``Annotated`` and unions to
        the classes ``isinstance`` accepts, so no model needs
        ``arbitrary_types_allowed``.

        Returns:
            The ``is_instance``/``none``/``union`` core schema for ``target``.

        """
        scope = bound or {}
        origin = get_origin(target)
        args = get_args(target)
        resolve = FlextTypesPydantic.port_instance_schema
        schema: core_schema.CoreSchema
        if isinstance(target, TypeVar):
            schema = (
                resolve(scope[target]) if target in scope else core_schema.any_schema()
            )
        elif isinstance(target, TypeAliasType):
            schema = resolve(target.__value__, scope)
        elif isinstance(origin, TypeAliasType):
            parameters = {
                parameter: scope.get(arg, arg)
                for parameter, arg in zip(origin.__type_params__, args, strict=False)
                if isinstance(parameter, TypeVar)
            }
            schema = resolve(origin.__value__, {**scope, **parameters})
        elif origin in {Union, UnionType}:
            schema = core_schema.union_schema([resolve(arg, scope) for arg in args])
        elif origin is Annotated:
            schema = resolve(args[0], scope)
        elif target is None or target is NoneType:
            schema = core_schema.none_schema()
        else:
            schema = core_schema.is_instance_schema(origin or target)
        return schema

    # Dependency port: a field typed by a class, normally a
    # ``@runtime_checkable`` Protocol from ``p``, or an alias over such classes.
    # ``port_instance_schema`` validates the value with ``isinstance`` on
    # construction and on assignment; the port never enters the JSON Schema.
    # Declare the field with ``m.Field(exclude=True, description=...)``:
    # field-level metadata cannot live inside a type alias.
    type Port[P] = pydantic_json_schema.SkipJsonSchema[
        Annotated[
            P,
            pydantic.GetPydanticSchema(
                lambda source, _handler: FlextTypesPydantic.port_instance_schema(source)
            ),
        ]
    ]


__all__: list[str] = ["FlextTypesPydantic", "JsonValue"]
