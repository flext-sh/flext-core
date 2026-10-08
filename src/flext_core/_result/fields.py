"""Typed field-population boundary for the FlextResult family.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel

from flext_core.typings import JsonDict


class FlextResultFieldModel(BaseModel):
    """Typed field-population boundary for the result family.

    Deliberately carries an explicit keyword-only ``__init__``: it gives the
    family's factory constructor (``FlextResultBase.__init__``) a typed,
    signature-compatible member to override, instead of pydantic's
    ``(**data: Any) -> None`` population contract that no typed factory
    signature can satisfy. Left undecorated (no PEP 698 ``@override``) so
    implicit ``__init__`` overrides stay exempt from signature checks.
    """

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    success: bool = True
    error: str | None = None
    error_code: str | None = None
    error_data: JsonDict | None = None

    def __init__(
        self,
        *,
        success: bool = True,
        error: str | None = None,
        error_code: str | None = None,
        error_data: JsonDict | None = None,
    ) -> None:
        super().__init__(
            success=success,
            error=error,
            error_code=error_code,
            error_data=error_data,
        )


__all__: list[str] = ["FlextResultFieldModel"]
