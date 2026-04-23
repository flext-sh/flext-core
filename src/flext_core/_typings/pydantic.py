"""Pydantic v2 constrained and specialized types exported via FlextTypes.

Including: StrictStr, EmailStr, UUID*, HttpUrl, PositiveInt, etc.

Architecture: Abstraction boundary - typings layer

Type aliases use PEP 613 TypeAlias (class body compatible with mypy 1.20+) instead
of PEP 695 `type X = ...` which PEP 695 officially restricts to module scope and
mypy treats inconsistently inside class bodies (rejecting usages as types).
Runtime callables and generic classes that accept `[T]` parameterization remain
as plain attributes because `TypeAlias` only applies to non-generic aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypeAlias

import pydantic
import pydantic_core
from pydantic_core import core_schema


class FlextTypesPydantic:
    """Constrained and specialized types: strict types, URL types, numeric types, etc.

    **NEVER import pydantic directly outside flext-core/src/.**
    Use t.* instead.
    """

    # String constraints
    constr = pydantic.constr
    StrictStr: TypeAlias = pydantic.StrictStr
    EmailStr: TypeAlias = pydantic.EmailStr
    NameEmail: TypeAlias = pydantic.NameEmail

    # Numeric constraints (callables — remain attributes)
    conint = pydantic.conint
    confloat = pydantic.confloat
    conbytes = pydantic.conbytes
    conlist = pydantic.conlist
    conset = pydantic.conset
    condate = pydantic.condate
    condecimal = pydantic.condecimal
    confrozenset = pydantic.confrozenset
    PositiveInt: TypeAlias = pydantic.PositiveInt
    NonNegativeInt: TypeAlias = pydantic.NonNegativeInt
    NegativeInt: TypeAlias = pydantic.NegativeInt
    NonPositiveInt: TypeAlias = pydantic.NonPositiveInt
    PositiveFloat: TypeAlias = pydantic.PositiveFloat
    NonNegativeFloat: TypeAlias = pydantic.NonNegativeFloat
    NegativeFloat: TypeAlias = pydantic.NegativeFloat
    NonPositiveFloat: TypeAlias = pydantic.NonPositiveFloat
    FiniteFloat: TypeAlias = pydantic.FiniteFloat
    ByteSize: TypeAlias = pydantic.ByteSize

    # Strict types
    StrictInt: TypeAlias = pydantic.StrictInt
    StrictFloat: TypeAlias = pydantic.StrictFloat
    StrictBool: TypeAlias = pydantic.StrictBool
    StrictBytes: TypeAlias = pydantic.StrictBytes

    # Date and time types
    AwareDatetime: TypeAlias = pydantic.AwareDatetime
    NaiveDatetime: TypeAlias = pydantic.NaiveDatetime
    FutureDate: TypeAlias = pydantic.FutureDate
    FutureDatetime: TypeAlias = pydantic.FutureDatetime
    PastDate: TypeAlias = pydantic.PastDate
    PastDatetime: TypeAlias = pydantic.PastDatetime

    # URL and network types
    AnyUrl: TypeAlias = pydantic.AnyUrl
    AnyHttpUrl: TypeAlias = pydantic.AnyHttpUrl
    AnyWebsocketUrl: TypeAlias = pydantic.AnyWebsocketUrl
    HttpUrl: TypeAlias = pydantic.HttpUrl
    WebsocketUrl: TypeAlias = pydantic.WebsocketUrl
    FileUrl: TypeAlias = pydantic.FileUrl
    FtpUrl: TypeAlias = pydantic.FtpUrl
    PostgresDsn: TypeAlias = pydantic.PostgresDsn
    MySQLDsn: TypeAlias = pydantic.MySQLDsn
    MariaDBDsn: TypeAlias = pydantic.MariaDBDsn
    CockroachDsn: TypeAlias = pydantic.CockroachDsn
    ClickHouseDsn: TypeAlias = pydantic.ClickHouseDsn
    MongoDsn: TypeAlias = pydantic.MongoDsn
    KafkaDsn: TypeAlias = pydantic.KafkaDsn
    RedisDsn: TypeAlias = pydantic.RedisDsn
    SnowflakeDsn: TypeAlias = pydantic.SnowflakeDsn
    NatsDsn: TypeAlias = pydantic.NatsDsn

    # File system types
    FilePath: TypeAlias = pydantic.FilePath
    DirectoryPath: TypeAlias = pydantic.DirectoryPath
    NewPath: TypeAlias = pydantic.NewPath
    SocketPath: TypeAlias = pydantic.SocketPath

    # UUID types
    UUID1: TypeAlias = pydantic.UUID1
    UUID3: TypeAlias = pydantic.UUID3
    UUID4: TypeAlias = pydantic.UUID4
    UUID5: TypeAlias = pydantic.UUID5
    UUID6: TypeAlias = pydantic.UUID6
    UUID7: TypeAlias = pydantic.UUID7
    UUID8: TypeAlias = pydantic.UUID8

    # Binary and encoding types
    Base64Str: TypeAlias = pydantic.Base64Str
    Base64Bytes: TypeAlias = pydantic.Base64Bytes
    Base64UrlStr: TypeAlias = pydantic.Base64UrlStr
    Base64UrlBytes: TypeAlias = pydantic.Base64UrlBytes
    EncodedStr: TypeAlias = pydantic.EncodedStr
    EncodedBytes: TypeAlias = pydantic.EncodedBytes

    # JSON and special types
    # pydantic.Json / ImportString / InstanceOf / Secret are Generic classes/aliases
    # used with [T] parameterization (e.g. `Json[dict[str, int]]`). TypeAlias only
    # supports non-generic aliases, so we keep them as plain attributes which pydantic
    # and the mypy plugin recognize as type markers.
    Json = pydantic.Json
    JsonValue: TypeAlias = pydantic.JsonValue
    ImportString = pydantic.ImportString
    InstanceOf = pydantic.InstanceOf
    Secret = pydantic.Secret
    SecretStr = pydantic.SecretStr
    SecretBytes: TypeAlias = pydantic.SecretBytes

    # IP types
    IPvAnyAddress: TypeAlias = pydantic.IPvAnyAddress
    IPvAnyInterface: TypeAlias = pydantic.IPvAnyInterface
    IPvAnyNetwork: TypeAlias = pydantic.IPvAnyNetwork

    # Constraint helper types (runtime markers / classes)
    StringConstraints = pydantic.StringConstraints
    UrlConstraints = pydantic.UrlConstraints
    ErrorDetails = pydantic_core.ErrorDetails
    ErrorType = core_schema.ErrorType
    ErrorTypeInfo = pydantic_core.ErrorTypeInfo
    InitErrorDetails = pydantic_core.InitErrorDetails

    # Annotation and alias helper types (runtime markers / classes)
    AliasGenerator = pydantic.AliasGenerator
    AliasChoices = pydantic.AliasChoices
    AliasPath = pydantic.AliasPath
    Discriminator = pydantic.Discriminator
    Tag = pydantic.Tag
    ValidateAs = pydantic.ValidateAs
    WithJsonSchema = pydantic.WithJsonSchema
    SerializeAsAny = pydantic.SerializeAsAny
    SkipValidation = pydantic.SkipValidation
    AllowInfNan = pydantic.AllowInfNan
    Strict = pydantic.Strict
    FailFast = pydantic.FailFast
    OnErrorOmit = pydantic.OnErrorOmit
