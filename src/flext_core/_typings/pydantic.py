"""Pydantic v2 constrained and specialized types exported via FlextTypes.

Including: StrictStr, EmailStr, UUID*, HttpUrl, PositiveInt, etc.

Architecture: Abstraction boundary - typings layer

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
    StrictStr = pydantic.StrictStr
    EmailStr = pydantic.EmailStr
    NameEmail = pydantic.NameEmail

    # Numeric constraints
    conint = pydantic.conint
    confloat = pydantic.confloat
    conbytes = pydantic.conbytes
    conlist = pydantic.conlist
    conset = pydantic.conset
    condate = pydantic.condate
    condecimal = pydantic.condecimal
    confrozenset = pydantic.confrozenset
    PositiveInt = pydantic.PositiveInt
    NonNegativeInt = pydantic.NonNegativeInt
    NegativeInt = pydantic.NegativeInt
    NonPositiveInt = pydantic.NonPositiveInt
    PositiveFloat = pydantic.PositiveFloat
    NonNegativeFloat = pydantic.NonNegativeFloat
    NegativeFloat = pydantic.NegativeFloat
    NonPositiveFloat = pydantic.NonPositiveFloat
    FiniteFloat = pydantic.FiniteFloat
    ByteSize = pydantic.ByteSize

    # Strict types
    StrictInt = pydantic.StrictInt
    StrictFloat = pydantic.StrictFloat
    StrictBool = pydantic.StrictBool
    StrictBytes = pydantic.StrictBytes

    # Date and time types
    AwareDatetime = pydantic.AwareDatetime
    NaiveDatetime = pydantic.NaiveDatetime
    FutureDate = pydantic.FutureDate
    FutureDatetime = pydantic.FutureDatetime
    PastDate = pydantic.PastDate
    PastDatetime = pydantic.PastDatetime

    # URL and network types
    AnyUrl = pydantic.AnyUrl
    AnyHttpUrl = pydantic.AnyHttpUrl
    AnyWebsocketUrl = pydantic.AnyWebsocketUrl
    HttpUrl = pydantic.HttpUrl
    WebsocketUrl = pydantic.WebsocketUrl
    FileUrl = pydantic.FileUrl
    FtpUrl = pydantic.FtpUrl
    PostgresDsn = pydantic.PostgresDsn
    MySQLDsn = pydantic.MySQLDsn
    MariaDBDsn = pydantic.MariaDBDsn
    CockroachDsn = pydantic.CockroachDsn
    ClickHouseDsn = pydantic.ClickHouseDsn
    MongoDsn = pydantic.MongoDsn
    KafkaDsn = pydantic.KafkaDsn
    RedisDsn = pydantic.RedisDsn
    SnowflakeDsn = pydantic.SnowflakeDsn
    NatsDsn = pydantic.NatsDsn

    # File system types
    FilePath = pydantic.FilePath
    DirectoryPath = pydantic.DirectoryPath
    NewPath = pydantic.NewPath
    SocketPath = pydantic.SocketPath

    # UUID types
    UUID1 = pydantic.UUID1
    UUID3 = pydantic.UUID3
    UUID4 = pydantic.UUID4
    UUID5 = pydantic.UUID5
    UUID6 = pydantic.UUID6
    UUID7 = pydantic.UUID7
    UUID8 = pydantic.UUID8

    # Binary and encoding types
    Base64Str = pydantic.Base64Str
    Base64Bytes = pydantic.Base64Bytes
    Base64UrlStr = pydantic.Base64UrlStr
    Base64UrlBytes = pydantic.Base64UrlBytes
    EncodedStr = pydantic.EncodedStr
    EncodedBytes = pydantic.EncodedBytes

    # JSON and special types
    Json = pydantic.Json
    JsonValue = pydantic.JsonValue
    ImportString = pydantic.ImportString
    InstanceOf = pydantic.InstanceOf
    Secret = pydantic.Secret
    SecretStr = pydantic.SecretStr
    SecretBytes = pydantic.SecretBytes

    # IP types
    IPvAnyAddress = pydantic.IPvAnyAddress
    IPvAnyInterface = pydantic.IPvAnyInterface
    IPvAnyNetwork = pydantic.IPvAnyNetwork

    # Constraint helper types
    StringConstraints = pydantic.StringConstraints
    UrlConstraints = pydantic.UrlConstraints
    ErrorDetails = pydantic_core.ErrorDetails
    ErrorType = core_schema.ErrorType
    ErrorTypeInfo = pydantic_core.ErrorTypeInfo
    InitErrorDetails = pydantic_core.InitErrorDetails

    # Annotation and alias helper types
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
