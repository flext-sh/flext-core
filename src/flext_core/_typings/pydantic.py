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
from typing import Annotated

import pydantic
import pydantic_core
from pydantic import json_schema as pydantic_json_schema
from pydantic_core import core_schema

type JsonValue = pydantic.JsonValue


class FlextTypesPydantic:
    """Constrained and specialized types: strict types, URL types, numeric types, etc.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use t.* instead.
    """

    # String constraints
    constr = pydantic.constr
    type StrictStr = pydantic.StrictStr
    type EmailStr = pydantic.EmailStr
    type NameEmail = pydantic.NameEmail

    # Numeric constraints (callables — remain attributes)
    conint = pydantic.conint
    confloat = pydantic.confloat
    conbytes = pydantic.conbytes
    conlist = pydantic.conlist
    conset = pydantic.conset
    condate = pydantic.condate
    condecimal = pydantic.condecimal
    confrozenset = pydantic.confrozenset
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
    # PEP 695 aliases, never bare class-scope assignments: a bare
    # ``BaseModel = pydantic.BaseModel`` is a variable to mypy, so it is not
    # valid as a type and breaks isinstance narrowing (mypy [unreachable]).
    type BaseModel = pydantic.BaseModel
    type TypeAdapter[T] = pydantic.TypeAdapter[T]
    type ImportString[T] = pydantic.ImportString[T]
    type InstanceOf[T] = pydantic.InstanceOf[T]
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
    SkipValidation = pydantic.SkipValidation
    AllowInfNan = pydantic.AllowInfNan
    Strict = pydantic.Strict
    FailFast = pydantic.FailFast
    OnErrorOmit = pydantic.OnErrorOmit

    # Dependency port: a field typed by a ``@runtime_checkable`` Protocol from
    # ``p``. Pydantic validates the value with ``isinstance`` on construction and
    # on assignment, and the port never enters the JSON Schema. Declare the field
    # with ``m.Field(exclude=True, description=...)``: field-level metadata cannot
    # live inside a type alias.
    type Port[P] = Annotated[P, pydantic_json_schema.SkipJsonSchema()]
