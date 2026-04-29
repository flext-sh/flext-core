"""Collection models for categorized data.

Pydantic v2 model surface only. Aggregation/normalization helpers live in
``FlextUtilitiesCollection`` (utilities tier). Models here expose fields,
validators, and computed properties - no domain helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import FlextModelsBase as m, FlextModelsPydantic as mp, FlextTypes as t


class FlextModelsCollections:
    """Collection models namespace (Pydantic v2 only)."""

    class GuardCheckSpec(m.ArbitraryTypesModel):
        """Specification for guard conditions used in collection filters."""

        eq: Annotated[
            t.JsonValue | None,
            mp.Field(
                default=None,
                title="Equals",
                description="Require the value to equal this value.",
            ),
        ] = None
        ne: Annotated[
            t.JsonValue | None,
            mp.Field(
                default=None,
                title="Not Equals",
                description="Require the value to differ from this value.",
            ),
        ] = None
        gt: Annotated[
            float | str | None,
            mp.Field(
                default=None,
                title="Greater Than",
                description="Require value to be greater than this (sortable: numeric or string).",
            ),
        ] = None
        gte: Annotated[
            float | str | None,
            mp.Field(
                default=None,
                title="Greater Than Or Equal",
                description="Require value to be greater than or equal to this (sortable: numeric or string).",
            ),
        ] = None
        lt: Annotated[
            float | str | None,
            mp.Field(
                default=None,
                title="Less Than",
                description="Require value to be less than this (sortable: numeric or string).",
            ),
        ] = None
        lte: Annotated[
            float | str | None,
            mp.Field(
                default=None,
                title="Less Than Or Equal",
                description="Require value to be less than or equal to this (sortable: numeric or string).",
            ),
        ] = None
        is_: Annotated[
            type | None,
            mp.Field(
                default=None,
                title="Is Type",
                description="Require the value to be an instance of this type.",
            ),
        ] = None
        not_: Annotated[
            type | None,
            mp.Field(
                default=None,
                title="Not Type",
                description="Require the value to not be an instance of this type.",
            ),
        ] = None
        in_: Annotated[
            t.JsonList | None,
            mp.Field(
                default=None,
                title="In Values",
                description="Require the value to be present in this sequence.",
            ),
        ] = None
        not_in: Annotated[
            t.JsonList | None,
            mp.Field(
                default=None,
                title="Not In Values",
                description="Require the value to not be present in this sequence.",
            ),
        ] = None
        none: Annotated[
            bool | None,
            mp.Field(
                default=None,
                title="None Constraint",
                description="When True, require None. When False, require non-None.",
            ),
        ] = None
        empty: Annotated[
            bool | None,
            mp.Field(
                default=None,
                title="Empty Constraint",
                description="When True, require empty value; when False, require non-empty.",
            ),
        ] = None
        match: Annotated[
            str | None,
            mp.Field(
                default=None,
                title="Regex Match",
                description="Require string value to match this regular expression.",
            ),
        ] = None
        contains: Annotated[
            str | None,
            mp.Field(
                default=None,
                title="Contains",
                description="Require string or iterable value to contain this item.",
            ),
        ] = None
        starts: Annotated[
            str | None,
            mp.Field(
                default=None,
                title="Starts With",
                description="Require string value to start with this prefix.",
            ),
        ] = None
        ends: Annotated[
            str | None,
            mp.Field(
                default=None,
                title="Ends With",
                description="Require string value to end with this suffix.",
            ),
        ] = None


__all__: t.MutableSequenceOf[str] = ["FlextModelsCollections"]
