"""Type variables and parameter specifications for the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import ParamSpec, TypeVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings

EnumT = TypeVar("EnumT", bound=StrEnum)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
TRuntime = TypeVar("TRuntime")
TV = TypeVar("TV")
TV_co = TypeVar("TV_co", covariant=True)
U = TypeVar("U")
ValidatedParams = ParamSpec("ValidatedParams")
ValidatedReturn = TypeVar("ValidatedReturn")
