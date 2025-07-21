"""Domain exceptions module for backward compatibility.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module re-exports exceptions from core for backward compatibility.
"""

from __future__ import annotations

from flext_core.domain.core import APIError
from flext_core.domain.core import AuthenticationError
from flext_core.domain.core import AuthorizationError
from flext_core.domain.core import ConfigurationError
from flext_core.domain.core import ConnectionError  # noqa: A004
from flext_core.domain.core import DatabaseError
from flext_core.domain.core import DataError
from flext_core.domain.core import DomainError
from flext_core.domain.core import ExternalServiceError
from flext_core.domain.core import LDAPError
from flext_core.domain.core import LDIFError
from flext_core.domain.core import MeltanoError
from flext_core.domain.core import NotFoundError
from flext_core.domain.core import OICError
from flext_core.domain.core import OracleError
from flext_core.domain.core import PluginError
from flext_core.domain.core import RepositoryError
from flext_core.domain.core import SchemaError
from flext_core.domain.core import ServiceError
from flext_core.domain.core import SingerError
from flext_core.domain.core import TapError
from flext_core.domain.core import TargetError
from flext_core.domain.core import TimeoutError  # noqa: A004
from flext_core.domain.core import TransformationError
from flext_core.domain.core import ValidationError
from flext_core.domain.core import WMSError

__all__ = [
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ConnectionError",
    "DataError",
    "DatabaseError",
    "DomainError",
    "ExternalServiceError",
    "LDAPError",
    "LDIFError",
    "MeltanoError",
    "NotFoundError",
    "OICError",
    "OracleError",
    "PluginError",
    "RepositoryError",
    "SchemaError",
    "ServiceError",
    "SingerError",
    "TapError",
    "TargetError",
    "TimeoutError",
    "TransformationError",
    "ValidationError",
    "WMSError",
]
