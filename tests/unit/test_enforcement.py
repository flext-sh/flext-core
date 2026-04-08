"""Tests for Pydantic v2 runtime enforcement across all Flext* layers.

Verifies that FlextUtilitiesEnforcement check functions correctly
detect violations. Tests call the check functions directly to avoid
test-module auto-exemption.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from typing import ClassVar, Final, Protocol, runtime_checkable

import pytest
from pydantic import Field
from pydantic.warnings import PydanticDeprecatedSince20

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextModelsNamespace,
    FlextProtocols,
    FlextTypes,
    FlextUtilities,
    FlextUtilitiesEnforcement,
)
from tests import c, m, p, t, u


class TestCheckNoAny:
    """Verify Any detection in model fields."""

    def test_any_field_detected(self) -> None:
        """Any as field annotation is flagged."""

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
        with pytest.warns(PydanticDeprecatedSince20):

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

        own = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_field_descriptions(_M, own=own)
        assert len(errors) == 1
        assert "description" in errors[0]

    def test_description_present_passes(self) -> None:
        """Field with description passes."""

        class _M(m.ArbitraryTypesModel):
            _flext_enforcement_exempt: ClassVar[bool] = True
            name: str = Field(default="test", description="A name")

        own = FlextUtilitiesEnforcement.own_fields(_M)
        errors = FlextUtilitiesEnforcement.check_field_descriptions(_M, own=own)
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


class TestNamespacePrefixDerivation:
    """Verify namespace prefix derivation normalizes project slugs."""

    def test_to_pascal_case_accepts_hyphenated_slug(self) -> None:
        """Hyphenated project slugs must normalize to a valid PascalCase prefix."""
        assert FlextUtilitiesEnforcement._to_pascal_case("db-oracle") == "DbOracle"

    def test_flext_core_uses_flext_prefix(self) -> None:
        """flext-core is the canonical exception and keeps the Flext prefix."""
        assert (
            FlextUtilitiesEnforcement._derive_prefix_from_path(
                FlextUtilitiesEnforcement,
            )
            == "Flext"
        )


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


# ------------------------------------------------------------------ #
# Constants layer enforcement tests                                    #
# ------------------------------------------------------------------ #


class TestConstantsEnforcement:
    """Verify enforcement on FlextConstants subclasses."""

    def test_mutable_list_detected(self) -> None:
        """List value in constants is flagged as HARD violation."""

        class _C:
            ITEMS: list[str] = ["a", "b"]

        errors = FlextUtilitiesEnforcement.check_constants_no_mutable_values(_C)
        assert len(errors) == 1
        assert "mutable list" in errors[0]

    def test_mutable_dict_detected(self) -> None:
        """Dict value in constants is flagged."""

        class _C:
            DATA: dict[str, int] = {"x": 1}

        errors = FlextUtilitiesEnforcement.check_constants_no_mutable_values(_C)
        assert len(errors) == 1
        assert "mutable dict" in errors[0]

    def test_frozenset_passes(self) -> None:
        """Frozenset value is not flagged."""

        class _C:
            ITEMS: Final[frozenset[str]] = frozenset({"a"})

        errors = FlextUtilitiesEnforcement.check_constants_no_mutable_values(_C)
        assert len(errors) == 0

    def test_tuple_passes(self) -> None:
        """Tuple value is not flagged."""

        class _C:
            ITEMS: Final[tuple[str, ...]] = ("a", "b")

        errors = FlextUtilitiesEnforcement.check_constants_no_mutable_values(_C)
        assert len(errors) == 0

    def test_missing_final_annotation_detected(self) -> None:
        """Public attribute without Final/ClassVar is flagged."""

        class _C:
            VALUE: int = 42

        errors = FlextUtilitiesEnforcement.check_constants_final_hints(_C)
        assert len(errors) == 1
        assert "Final" in errors[0]

    def test_final_annotation_passes(self) -> None:
        """Attribute with Final passes."""

        class _C:
            VALUE: Final[int] = 42

        errors = FlextUtilitiesEnforcement.check_constants_final_hints(_C)
        assert len(errors) == 0

    def test_classvar_annotation_passes(self) -> None:
        """Attribute with ClassVar passes."""

        class _C:
            VALUE: ClassVar[int] = 42

        errors = FlextUtilitiesEnforcement.check_constants_final_hints(_C)
        assert len(errors) == 0

    def test_upper_case_passes(self) -> None:
        """UPPER_CASE name passes."""

        class _C:
            MY_VALUE: Final[int] = 42

        errors = FlextUtilitiesEnforcement.check_constants_upper_case(_C)
        assert len(errors) == 0

    def test_lower_case_detected(self) -> None:
        """Lowercase name is flagged."""

        class _C:
            my_value: int = 42

        errors = FlextUtilitiesEnforcement.check_constants_upper_case(_C)
        assert len(errors) == 1
        assert "UPPER_CASE" in errors[0]

    def test_inner_namespace_mutable_detected(self) -> None:
        """Mutable value inside inner namespace class is caught recursively."""

        class _C:
            class Inner:
                BAD: list[str] = ["x"]

        errors = FlextUtilitiesEnforcement.check_constants_no_mutable_values(_C)
        assert len(errors) == 1
        assert "Inner" in errors[0]

    def test_facade_has_enforced_namespace(self) -> None:
        """FlextConstants facade inherits FlextModelsNamespace."""
        assert issubclass(FlextConstants, FlextModelsNamespace)

    def test_enforcement_constants_accessible(self) -> None:
        """New enforcement constants accessible via c.*."""
        assert hasattr(c, "ENFORCEMENT_CONSTANTS_SKIP_ATTRS")
        assert hasattr(c, "ENFORCEMENT_UTILITIES_EXEMPT_METHODS")


# ------------------------------------------------------------------ #
# Protocols layer enforcement tests                                    #
# ------------------------------------------------------------------ #


class TestProtocolsEnforcement:
    """Verify enforcement on FlextProtocols subclasses."""

    def test_non_protocol_inner_class_detected(self) -> None:
        """Inner class that is not a Protocol is flagged."""

        class _P:
            class NotProtocol:
                pass

        errors = FlextUtilitiesEnforcement.check_protocols_inner_classes(_P)
        assert len(errors) == 1
        assert "Protocol subclass" in errors[0]

    def test_protocol_inner_class_passes(self) -> None:
        """Inner class that IS a Protocol passes."""

        class _P:
            @runtime_checkable
            class Good(Protocol):
                def method(self) -> None: ...

        errors = FlextUtilitiesEnforcement.check_protocols_inner_classes(_P)
        assert len(errors) == 0

    def test_non_runtime_checkable_detected(self) -> None:
        """Protocol without @runtime_checkable is flagged."""

        class _P:
            class Bare(Protocol):
                def method(self) -> None: ...

        errors = FlextUtilitiesEnforcement.check_protocols_runtime_checkable(_P)
        assert len(errors) == 1
        assert "runtime_checkable" in errors[0]

    def test_runtime_checkable_passes(self) -> None:
        """Protocol with @runtime_checkable passes."""

        class _P:
            @runtime_checkable
            class Good(Protocol):
                def method(self) -> None: ...

        errors = FlextUtilitiesEnforcement.check_protocols_runtime_checkable(_P)
        assert len(errors) == 0

    def test_facade_has_enforced_namespace(self) -> None:
        """FlextProtocols facade inherits FlextModelsNamespace."""
        assert issubclass(FlextProtocols, FlextModelsNamespace)


# ------------------------------------------------------------------ #
# Types layer enforcement tests                                        #
# ------------------------------------------------------------------ #


class TestTypesEnforcement:
    """Verify enforcement on FlextTypes subclasses."""

    def test_any_in_type_alias_detected(self) -> None:
        """Any in PEP 695 type alias is flagged."""

        class _T:
            type Anything = typing.Any | str

        errors = FlextUtilitiesEnforcement.check_types_no_any_in_aliases(_T)
        assert len(errors) == 1
        assert "Any" in errors[0]

    def test_clean_type_alias_passes(self) -> None:
        """Type alias without Any passes."""

        class _T:
            type Name = str | int

        errors = FlextUtilitiesEnforcement.check_types_no_any_in_aliases(_T)
        assert len(errors) == 0

    def test_facade_has_enforced_namespace(self) -> None:
        """FlextTypes facade inherits FlextModelsNamespace."""
        assert issubclass(FlextTypes, FlextModelsNamespace)


# ------------------------------------------------------------------ #
# Utilities layer enforcement tests                                    #
# ------------------------------------------------------------------ #


class TestUtilitiesEnforcement:
    """Verify enforcement on FlextUtilities subclasses."""

    def test_instance_method_detected(self) -> None:
        """Regular instance method is flagged."""

        class _U:
            def my_method(self, x: int) -> str:
                return str(x)

        errors = FlextUtilitiesEnforcement.check_utilities_method_types(_U)
        assert len(errors) == 1
        assert "staticmethod" in errors[0]

    def test_static_method_passes(self) -> None:
        """@staticmethod passes."""

        class _U:
            @staticmethod
            def my_method(x: int) -> str:
                return str(x)

        errors = FlextUtilitiesEnforcement.check_utilities_method_types(_U)
        assert len(errors) == 0

    def test_classmethod_passes(self) -> None:
        """@classmethod passes."""

        class _U:
            @classmethod
            def my_method(cls, x: int) -> str:
                return str(x)

        errors = FlextUtilitiesEnforcement.check_utilities_method_types(_U)
        assert len(errors) == 0

    def test_facade_has_enforced_namespace(self) -> None:
        """FlextUtilities facade inherits FlextModelsNamespace."""
        assert issubclass(FlextUtilities, FlextModelsNamespace)


# ------------------------------------------------------------------ #
# Cross-layer integration tests                                        #
# ------------------------------------------------------------------ #


class TestAllLayerIntegration:
    """Verify all facades fire enforcement on subclasses."""

    def test_all_facades_inherit_enforced_namespace(self) -> None:
        """All 5 facade classes inherit FlextModelsNamespace."""
        for facade in (
            FlextConstants,
            FlextProtocols,
            FlextTypes,
            FlextUtilities,
            FlextModels,
        ):
            assert issubclass(facade, FlextModelsNamespace), (
                f"{facade.__name__} missing FlextModelsNamespace in MRO"
            )

    def test_downstream_projects_load_cleanly(self) -> None:
        """Flext-core facades load without any violations."""
        assert c.__name__ == "TestsFlextCoreConstants"
        assert m.__name__ == "TestsFlextCoreModels"
        assert p.__name__ == "TestsFlextCoreProtocols"
        assert t.__name__ == "TestsFlextCoreTypes"
        assert u.__name__ == "TestsFlextCoreUtilities"

    def test_exempt_flag_works_across_layers(self) -> None:
        """_flext_enforcement_exempt disables layer checks."""
        target = type("_ExemptCls", (), {"_flext_enforcement_exempt": True})
        assert FlextUtilitiesEnforcement._is_layer_exempt(target)
