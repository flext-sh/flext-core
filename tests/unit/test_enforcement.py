"""Tests for Pydantic v2 runtime enforcement.

Verifies that FlextUtilitiesEnforcement check functions correctly
detect violations. Tests call the check functions directly to avoid
test-module auto-exemption.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import pytest
from pydantic import Field

from flext_core import c, m
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement


class TestCheckNoAny:
    """Verify Any detection in model fields."""

    def test_any_field_detected(self) -> None:
        """Any as field annotation is flagged."""
        import typing

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            data: typing.Any = Field(default=None, description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_any(fields)
        assert len(errors) == 1
        assert "Any" in errors[0]

    def test_typed_field_passes(self) -> None:
        """Str field is not flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            name: str = Field(default="x", description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_any(fields)
        assert len(errors) == 0


class TestCheckNoBareCollections:
    """Verify bare dict/list/set detection in model fields."""

    def test_bare_dict_detected(self) -> None:
        """dict[K,V] in field annotation is flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            data: dict[str, str] = Field(default_factory=dict, description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_bare_collections(fields)
        assert len(errors) == 1
        assert "dict" in errors[0]

    def test_bare_list_detected(self) -> None:
        """list[X] in field annotation is flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            items: list[str] = Field(default_factory=list, description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_bare_collections(fields)
        assert len(errors) == 1
        assert "list" in errors[0]

    def test_mapping_passes(self) -> None:
        """Mapping[K,V] is not flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            data: Mapping[str, str] = Field(default_factory=dict, description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_bare_collections(fields)
        assert len(errors) == 0

    def test_sequence_passes(self) -> None:
        """Sequence[X] is not flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            items: Sequence[str] = Field(default_factory=list, description="d")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_no_bare_collections(fields)
        assert len(errors) == 0


class TestCheckNoV1Patterns:
    """Verify Pydantic v1 pattern detection."""

    def test_v1_config_class_detected(self) -> None:
        """Class Config inside model is flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True

            class Config:
                extra = "forbid"

            name: str = Field(default="x", description="d")

        errors = FlextUtilitiesEnforcement.check_no_v1_patterns(_M)
        assert len(errors) == 1
        assert "Pydantic v1" in errors[0]


class TestCheckFieldDescriptions:
    """Verify missing description detection."""

    def test_missing_description_detected(self) -> None:
        """Field without description is flagged."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            name: str = "test"

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_field_descriptions(fields)
        assert len(errors) == 1
        assert "description" in errors[0]

    def test_description_present_passes(self) -> None:
        """Field with description passes."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            name: str = Field(default="test", description="A name")

        fields = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_field_descriptions(fields)
        assert len(errors) == 0


class TestCheckExtraPolicy:
    """Verify extra policy enforcement."""

    def test_flexible_internal_allows_ignore(self) -> None:
        """FlexibleInternalModel subclass allows extra=ignore."""

        class _M(m.FlexibleInternalModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            name: str = Field(default="x", description="d")

        errors = FlextUtilitiesEnforcement.check_extra_policy(_M)
        assert len(errors) == 0


class TestExemptions:
    """Verify exemption mechanisms."""

    def test_explicit_exempt_flag(self) -> None:
        """_flext_enforcement_exempt = True makes is_exempt return True."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True

        assert FlextUtilitiesEnforcement.is_exempt(_M)

    def test_infrastructure_base_exempt(self) -> None:
        """Infrastructure base names are auto-exempt."""
        assert FlextUtilitiesEnforcement.is_exempt(m.ArbitraryTypesModel)
        assert FlextUtilitiesEnforcement.is_exempt(m.FrozenValueModel)

    def test_test_module_exempt(self) -> None:
        """Classes in test modules are auto-exempt."""

        class _M(m.ArbitraryTypesModel):
            pass

        # This class is defined in tests.unit.test_enforcement
        assert FlextUtilitiesEnforcement.is_exempt(_M)


class TestEnforcementMode:
    """Verify enforcement mode constant is accessible."""

    def test_mode_is_warn(self) -> None:
        """Current mode should be warn."""
        assert c.ENFORCEMENT_MODE == "warn"

    def test_enforcement_constants_accessible(self) -> None:
        """All enforcement constants accessible via c.*."""
        assert hasattr(c, "ENFORCEMENT_MODE")
        assert hasattr(c, "ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS")
        assert hasattr(c, "ENFORCEMENT_INFRASTRUCTURE_BASES")
        assert hasattr(c, "ENFORCEMENT_RELAXED_EXTRA_BASES")
        assert hasattr(c, "ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS")
        assert hasattr(c, "ENFORCEMENT_COLLECTION_REPLACEMENTS")


class TestBaseModelCoverage:
    """Verify all base models have enforcement hooks."""

    @pytest.mark.parametrize(
        "base_cls",
        [
            m.ArbitraryTypesModel,
            m.StrictBoundaryModel,
            m.FlexibleInternalModel,
            m.ImmutableValueModel,
            m.TaggedModel,
            m.ContractModel,
        ],
    )
    def test_base_model_has_enforcement_hook(self, base_cls: type) -> None:
        """Each base model class should have __pydantic_init_subclass__."""
        assert hasattr(base_cls, "__pydantic_init_subclass__")
