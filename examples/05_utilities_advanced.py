"""u advanced features demonstration.

Demonstrates Args, Enum, Model, Text, Guards, Mapper,
Domain, Pagination, and Configuration utilities using Python 3.13+ strict
patterns with PEP 695 type aliases and collections.abc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import Field

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
    s,
    t,
    u,
)

# ═══════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS (Python 3.13+ PEP 695 strict)
# ═══════════════════════════════════════════════════════════════════


class StatusEnum(StrEnum):
    """Status enumeration using StrEnum."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class UserModel(FlextModels.ArbitraryTypesModel):
    """User model for demonstration using FlextModels."""

    name: str = Field(min_length=1)
    status: StatusEnum = StatusEnum.PENDING
    age: int = Field(ge=0, le=150)


# ═══════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════

TEST_DATA: Mapping[str, t.GeneralValueType] = {
    "name": "John Doe",
    "status": "active",
    "age": 30,
    "text": "  Hello   World  ",
    "long_text": "A" * 200,
    "source_dict": {"old_key": "value", "foo": "bar"},
    "key_mapping": {"old_key": "new_key", "foo": "bar"},
}


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class AdvancedUtilitiesService(s[t.Types.ServiceMetadataMapping]):
    """Service demonstrating advanced u features."""

    def execute(
        self,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Execute advanced utilities demonstrations."""
        print("Starting advanced utilities demonstration")

        try:
            self._demonstrate_args_validation()
            self._demonstrate_enum_utilities()
            self._demonstrate_model_utilities()
            self._demonstrate_text_processing()
            self._demonstrate_type_guards()
            self._demonstrate_data_mapping()
            self._demonstrate_domain_utilities()
            self._demonstrate_pagination()
            self._demonstrate_configuration()

            return FlextResult[t.Types.ServiceMetadataMapping].ok({
                "utilities_demonstrated": [
                    "args_validation",
                    "enum_utilities",
                    "model_utilities",
                    "text_processing",
                    "guards",
                    "data_mapping",
                    "domain_utilities",
                    "pagination",
                    "configuration",
                ],
                "utility_categories": 9,
                "advanced_features": [
                    "decorator_validation",
                    "strenum_parsing",
                    "model_creation",
                    "text_normalization",
                    "type_narrowing",
                    "data_transformation",
                    "entity_comparison",
                    "api_pagination",
                    "parameter_access",
                ],
            })

        except Exception as e:
            error_msg = f"Advanced utilities demonstration failed: {e}"
            return FlextResult[t.Types.ServiceMetadataMapping].fail(error_msg)

    @staticmethod
    def _demonstrate_args_validation() -> None:
        """Show Args validation utilities."""
        print("\n=== Args Validation ===")

        @u.Args.validated
        def process_status(status: StatusEnum) -> str:
            """Process status with automatic validation."""
            return f"Status: {status.value}"

        status_enum = StatusEnum.ACTIVE
        result = process_status(status_enum)
        print(f"✅ Validated function: {result}")

        @u.Args.validated_with_result
        def process_with_result(status: StatusEnum) -> FlextResult[str]:
            """Process with result validation."""
            return FlextResult[str].ok(f"Processed: {status.value}")

        status_enum_pending = StatusEnum.PENDING
        result_obj = process_with_result(status_enum_pending)
        if result_obj.is_success:
            print(f"✅ Validated with result: {result_obj.unwrap()}")

    @staticmethod
    def _demonstrate_enum_utilities() -> None:
        """Show Enum utilities."""
        print("\n=== Enum Utilities ===")

        # Parse enum from string
        parse_result = u.Enum.parse(StatusEnum, "active")
        if parse_result.is_success:
            status = parse_result.unwrap()
            print(f"✅ Enum parsing: {status.value}")

        # Type guard for enum membership
        test_value: t.GeneralValueType = "pending"
        if u.Enum.is_member(StatusEnum, test_value):
            print(f"✅ Type guard: {test_value} is valid StatusEnum")

        # Subset validation - using string value for type guard
        active_states = frozenset({StatusEnum.ACTIVE, StatusEnum.PENDING})
        test_status_str: str = "active"
        # Business Rule: is_subset accepts enum class (type[E]), frozenset of enum members, and value to check
        # StatusEnum is the enum class type. Use type() to ensure we pass the class, not an instance.
        # This pattern ensures type checker understands it's a class type for proper type inference.
        if u.Enum.is_subset(type(StatusEnum.ACTIVE), active_states, test_status_str):
            print("✅ Subset validation: 'active' is in active states")

    @staticmethod
    def _demonstrate_model_utilities() -> None:
        """Show Model utilities."""
        print("\n=== Model Utilities ===")

        # Create model from dict
        user_data: Mapping[str, t.FlexibleValue] = {
            "name": "Alice",
            "status": "active",
            "age": 25,
        }
        model_result = u.Model.from_dict(UserModel, user_data)
        if model_result.is_success:
            user = model_result.unwrap()
            status_value = (
                user.status.value
                if u.Validation.guard(user.status, StatusEnum, return_value=True)
                is not None
                else str(user.status)
            )
            print(f"✅ Model from dict: {user.name} ({status_value})")

        # Create model from kwargs
        kwargs_result = u.Model.from_kwargs(
            UserModel,
            name="Bob",
            status=StatusEnum.PENDING,
            age=30,
        )
        if kwargs_result.is_success:
            user = kwargs_result.unwrap()
            status_value = (
                user.status.value
                if u.Validation.guard(user.status, StatusEnum, return_value=True)
                is not None
                else str(user.status)
            )
            print(f"✅ Model from kwargs: {user.name} ({status_value})")

        # Merge defaults
        defaults: Mapping[str, t.FlexibleValue] = {
            "status": StatusEnum.PENDING,
            "age": 0,
        }
        overrides: Mapping[str, t.FlexibleValue] = {"name": "Charlie"}
        merge_result = u.Model.merge_defaults(UserModel, defaults, overrides)
        if merge_result.is_success:
            user = merge_result.unwrap()
            status_value = (
                user.status.value
                if u.Validation.guard(user.status, StatusEnum, return_value=True)
                is not None
                else str(user.status)
            )
            print(f"✅ Merged defaults: {user.name} ({status_value})")

    @staticmethod
    def _demonstrate_text_processing() -> None:
        """Show Text utilities."""
        print("\n=== Text Processing ===")

        # Clean text
        dirty_text = str(TEST_DATA["text"])
        cleaned = u.Text.clean_text(dirty_text)
        print(f"✅ Text cleaning: '{dirty_text}' → '{cleaned}'")

        # Truncate text
        long_text = str(TEST_DATA["long_text"])
        truncate_result = u.Text.truncate_text(long_text, max_length=50)
        if truncate_result.is_success:
            truncated = truncate_result.unwrap()
            print(f"✅ Text truncation: {len(truncated)} chars")

        # Safe string validation
        try:
            safe = u.Text.safe_string("  valid  ")
            print(f"✅ Safe string: '{safe}'")
        except ValueError as e:
            print(f"⚠️  Safe string validation: {e}")

    @staticmethod
    def _demonstrate_type_guards() -> None:
        """Show Guards utilities."""
        print("\n=== Type Guards ===")

        # String non-empty guard
        if u.is_type("hello", "string_non_empty"):
            print("✅ String non-empty guard: 'hello' is valid")

        # Dict non-empty guard
        test_dict: dict[str, str] = {"key": "value"}
        if u.is_type(test_dict, "dict_non_empty"):
            print("✅ Dict non-empty guard: dict is valid")

        # List non-empty guard
        test_list: list[int] = [1, 2, 3]
        if u.is_type(test_list, "list_non_empty"):
            print("✅ List non-empty guard: list is valid")

    @staticmethod
    def _demonstrate_data_mapping() -> None:
        """Show Mapper utilities."""
        print("\n=== Data Mapping ===")

        # Map dictionary keys
        source_value = TEST_DATA["source_dict"]
        mapping_value = TEST_DATA["key_mapping"]
        map_result: FlextResult[dict[str, t.GeneralValueType]] = FlextResult[
            dict[str, t.GeneralValueType]
        ].fail("Invalid data types")
        if (
            u.Validation.guard(source_value, Mapping, return_value=True) is not None
            and u.Validation.guard(mapping_value, Mapping, return_value=True)
            is not None
        ):
            # Type-safe dictionary creation from Mapping
            source_dict: dict[str, t.GeneralValueType] = (
                {str(k): v for k, v in source_value.items()}
                if isinstance(source_value, Mapping)
                else {}
            )
            # u.map expects dict/Mapping, ensure proper type
            if isinstance(mapping_value, Mapping):
                mapping_dict: dict[str, t.GeneralValueType] = {
                    str(k): v for k, v in mapping_value.items()
                }
                mapped_dict = u.Mapper.transform_values(mapping_dict, str)
                key_mapping: dict[str, str] = {
                    str(k): str(v) for k, v in mapped_dict.items()
                }
                map_result = u.Mapper.map_dict_keys(source_dict, key_mapping)
        if map_result.is_success:
            mapped = map_result.unwrap()
            print(f"✅ Key mapping: {list(mapped.keys())}")

        # Convert to int safe using parse()
        int_result = u.Parser.parse("123", int, default=0)
        print(
            f"✅ Safe int conversion: '123' → {int_result.value if int_result.is_success else 0}",
        )

        # Build flags dict
        flags: list[str] = ["read", "write"]
        flag_mapping: dict[str, str] = {
            "read": "can_read",
            "write": "can_write",
        }
        flags_result = u.Mapper.build_flags_dict(flags, flag_mapping)
        if flags_result.is_success:
            flags_dict = flags_result.unwrap()
            print(f"✅ Flags dict: {list(flags_dict.keys())}")

    @staticmethod
    def _demonstrate_domain_utilities() -> None:
        """Show Domain utilities."""
        print("\n=== Domain Utilities ===")

        # Create test entities for comparison demonstration
        user1 = UserModel(name="Alice", status=StatusEnum.ACTIVE, age=25)
        user2 = UserModel(name="Bob", status=StatusEnum.ACTIVE, age=30)

        # Compare value objects by value
        comparison = u.Domain.compare_value_objects_by_value(user1, user2)
        print(f"✅ Value object comparison: {comparison}")

        # Entity comparison utilities available
        print("✅ Entity comparison utilities available")
        print("✅ Entity hashing utilities available")

    @staticmethod
    def _demonstrate_pagination() -> None:
        """Show Pagination utilities."""
        print("\n=== Pagination ===")

        # Extract page params
        query_params: dict[str, str] = {"page": "2", "page_size": "10"}
        page_result = u.Pagination.extract_page_params(
            query_params,
            default_page=1,
            default_page_size=FlextConstants.Pagination.DEFAULT_PAGE_SIZE,
            max_page_size=FlextConstants.Pagination.MAX_PAGE_SIZE,
        )
        if page_result.is_success:
            page, page_size = page_result.unwrap()
            print(f"✅ Page params: page={page}, size={page_size}")

        # Validate pagination params
        validate_result = u.Pagination.validate_pagination_params(
            page=1,
            page_size=20,
            max_page_size=FlextConstants.Pagination.MAX_PAGE_SIZE,
        )
        if validate_result.is_success:
            params = validate_result.unwrap()
            print(f"✅ Validated params: {params}")

    @staticmethod
    def _demonstrate_configuration() -> None:
        """Show Configuration utilities."""
        print("\n=== Configuration ===")

        # Get parameter from model
        user = UserModel(name="Test", status=StatusEnum.ACTIVE, age=30)
        try:
            name_param = u.Configuration.get_parameter(user, "name")
            print(f"✅ Get parameter: name={name_param}")
        except Exception as e:
            print(f"⚠️  Get parameter: {e}")

        # Get parameter from dict
        config_dict: dict[str, t.GeneralValueType] = {
            "timeout": 30,
            "retries": 3,
        }
        try:
            timeout = u.Configuration.get_parameter(config_dict, "timeout")
            print(f"✅ Get from dict: timeout={timeout}")
        except Exception as e:
            print(f"⚠️  Get from dict: {e}")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT UTILITIES - ADVANCED FEATURES")
    print("Args, Enum, Model, Text, Guards, Mapper, Domain, Pagination, Configuration")
    print("=" * 60)

    service = AdvancedUtilitiesService()
    result = service.execute()

    if result.is_success:
        data = result.unwrap()
        utilities = data["utilities_demonstrated"]
        categories = data["utility_categories"]
        if isinstance(utilities, Sequence) and isinstance(categories, int):
            utilities_list = list(utilities)
            print(f"\n✅ Demonstrated {categories} utility categories")
            print(f"✅ Covered {len(utilities_list)} utility types")
    else:
        print(f"\n❌ Failed: {result.error}")

    print("\n" + "=" * 60)
    print("🎯 Advanced Utilities: Args, Enum, Model, Text")
    print("🎯 Type Safety: Guards, Mapper, Domain, Pagination")
    print("🎯 Configuration: Parameter access and manipulation")
    print("🎯 Python 3.13+: PEP 695 type aliases, collections.abc")
    print("=" * 60)


if __name__ == "__main__":
    main()
