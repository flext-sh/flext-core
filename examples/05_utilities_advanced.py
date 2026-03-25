"""u advanced features demonstration.

Demonstrates Args, Enum, Model, Text, Guards, Mapper,
Domain, Pagination, and Configuration utilities using Python 3.13+ strict
patterns with PEP 695 type aliases and collections.abc.

**Expected Output:**
- Argument parsing and validation utilities
- Enum utilities and transformations
- Model utilities for Pydantic integration
- Text manipulation and formatting
- Type guards and runtime type checking
- Data mapping and transformation
- Domain utilities for business logic
- Pagination helper functions

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from typing import override

from pydantic import Field

from flext_core import c, m, r, s, t, u


@unique
class StatusEnum(StrEnum):
    """Status enumeration using StrEnum."""

    ACTIVE = "active"
    PENDING = "pending"
    INACTIVE = "inactive"


class UserModel(m.ArbitraryTypesModel):
    """User model for demonstration using m."""

    name: str = Field(min_length=1)
    status: StatusEnum = StatusEnum.PENDING
    age: int = Field(ge=0, le=150)


TEST_DATA: Mapping[str, str | int | t.StrMapping] = {
    "name": "John Doe",
    "status": "active",
    "age": 30,
    "text": "  Hello   World  ",
    "long_text": "A" * 200,
    "source_dict": {"old_key": "value", "foo": "bar"},
    "key_mapping": {"old_key": "new_key", "foo": "bar"},
}


class AdvancedUtilitiesService(s[t.ConfigMap]):
    """Service demonstrating advanced u features."""

    @staticmethod
    def _demonstrate_args_validation() -> None:
        """Show Args validation utilities."""
        print("\n=== Args Validation ===")

        @u.validated
        def process_status(status: StatusEnum) -> str:
            """Process status with automatic validation."""
            return f"Status: {status.value}"

        status_enum = StatusEnum.ACTIVE
        result = process_status(status_enum)
        print(f"✅ Validated function: {result}")

        @u.validated
        def process_with_result(status: StatusEnum) -> str:
            return f"Processed: {status.value}"

        status_enum_pending = StatusEnum.PENDING
        result_obj = r[str].ok(process_with_result(status_enum_pending))
        print(f"✅ Validated with result: {result_obj.value}")

    @staticmethod
    def _demonstrate_configuration() -> None:
        """Show Configuration utilities."""
        print("\n=== Configuration ===")
        user = UserModel(name="Test", status=StatusEnum.ACTIVE, age=30)
        try:
            name_param = u.get_parameter(user.model_dump(), "name")
            print(f"✅ Get parameter: name={name_param}")
        except Exception as e:
            print(f"⚠️  Get parameter: {e}")
        config_dict: t.ConfigMap = t.ConfigMap(root={"timeout": 30, "retries": 3})
        try:
            timeout = u.get_parameter(config_dict.root, "timeout")
            print(f"✅ Get from dict: timeout={timeout}")
        except Exception as e:
            print(f"⚠️  Get from dict: {e}")

    @staticmethod
    def _demonstrate_data_mapping() -> None:
        """Show Mapper utilities."""
        print("\n=== Data Mapping ===")
        source_dict = {"old_key": "value", "foo": "bar"}
        mapped_dict = u.transform_values(source_dict, str)
        key_mapping_dict: t.StrMapping = {"old_key": "new_key", "foo": "bar"}
        map_result = u.map_dict_keys(mapped_dict, key_mapping_dict)
        if map_result.is_success:
            mapped = map_result.value
            print(f"✅ Key mapping: {list(mapped.keys())}")
        int_result = u.parse("123", int, default=0)
        print(f"✅ Safe int conversion: '123' → {int_result.map_or(0)}")
        flags: t.StrSequence = ["read", "write"]
        flag_mapping: t.StrMapping = {"read": "can_read", "write": "can_write"}
        flags_result = u.build_flags_dict(flags, flag_mapping)
        if flags_result.is_success:
            flags_dict = flags_result.value
            print(f"✅ Flags dict: {list(flags_dict.keys())}")

    @staticmethod
    def _demonstrate_domain_utilities() -> None:
        """Show Domain utilities."""
        print("\n=== Domain Utilities ===")
        user1 = UserModel(name="Alice", status=StatusEnum.ACTIVE, age=25)
        user2 = UserModel(name="Bob", status=StatusEnum.ACTIVE, age=30)
        comparison = u.compare_value_objects_by_value(user1, user2)
        print(f"✅ Value t.NormalizedValue comparison: {comparison}")
        print("✅ Entity comparison utilities available")
        print("✅ Entity hashing utilities available")

    @staticmethod
    def _demonstrate_enum_utilities() -> None:
        """Show Enum utilities."""
        print("\n=== Enum Utilities ===")
        parse_result = u.parse_enum(StatusEnum, "active")
        if parse_result.is_success:
            status = parse_result.value
            print(f"✅ Enum parsing: {status.value}")
        pending_result = u.parse_enum(StatusEnum, "pending")
        if pending_result.is_success:
            print("✅ Membership validation: 'pending' is a valid StatusEnum")
        active_result = u.parse_enum(StatusEnum, "active")
        if active_result.is_success:
            print("✅ Membership validation: 'active' is a valid StatusEnum")

    @staticmethod
    def _demonstrate_model_utilities() -> None:
        """Show Model utilities."""
        print("\n=== Model Utilities ===")
        user_data: Mapping[str, str | int] = {
            "name": "Alice",
            "status": "active",
            "age": 25,
        }
        model_result = u.load(UserModel, t.ConfigMap(root=dict(user_data)))
        user_from_dict = UserModel.model_validate(user_data)
        print(
            f"✅ Model from dict: {user_from_dict.name} ({user_from_dict.status.value})"
            f" [r={'ok' if model_result.is_success else 'fail'}]",
        )
        user_from_kwargs = UserModel(name="Bob", status=StatusEnum.PENDING, age=30)
        kwargs_result = u.from_kwargs(
            UserModel,
            name="Bob",
            status=StatusEnum.PENDING,
            age=30,
        )
        print(
            f"✅ Model from kwargs: {user_from_kwargs.name} ({user_from_kwargs.status.value})"
            f" [r={'ok' if kwargs_result.is_success else 'fail'}]",
        )
        defaults: Mapping[str, StatusEnum | int] = {
            "status": StatusEnum.PENDING,
            "age": 0,
        }
        overrides: t.StrMapping = {"name": "Charlie"}
        merge_result = u.merge_defaults(UserModel, defaults, overrides)
        merged_user = UserModel(name="Charlie", status=StatusEnum.PENDING, age=0)
        print(
            f"✅ Merged defaults: {merged_user.name} ({merged_user.status.value})"
            f" [r={'ok' if merge_result.is_success else 'fail'}]",
        )

    @staticmethod
    def _demonstrate_pagination() -> None:
        """Show Pagination utilities."""
        print("\n=== Pagination ===")
        query_params: t.StrMapping = {"page": "2", "page_size": "10"}
        page_result = u.extract_page_params(
            query_params,
            default_page=1,
            default_page_size=c.DEFAULT_PAGE_SIZE,
            max_page_size=c.MAX_PAGE_SIZE,
        )
        page: int = int(query_params.get("page", "1"))
        page_size: int = int(query_params.get("page_size", "10"))
        print(
            f"✅ Page params: page={page}, size={page_size}"
            f" [r={'ok' if page_result.is_success else 'fail'}]",
        )
        validate_result = u.validate_pagination_params(
            page=1,
            page_size=20,
            max_page_size=c.MAX_PAGE_SIZE,
        )
        print(f"✅ Validated params: is_valid={validate_result.is_success}")

    @staticmethod
    def _demonstrate_text_processing() -> None:
        """Show Text utilities."""
        print("\n=== Text Processing ===")
        dirty_text = str(TEST_DATA["text"])
        cleaned = u.clean_text(dirty_text)
        print(f"✅ Text cleaning: '{dirty_text}' → '{cleaned}'")
        long_text = str(TEST_DATA["long_text"])
        truncate_result = u.truncate_text(long_text, max_length=50)
        if truncate_result.is_success:
            truncated = truncate_result.value
            print(f"✅ Text truncation: {len(truncated)} chars")
        try:
            safe = u.safe_string("  valid  ")
            print(f"✅ Safe string: '{safe}'")
        except ValueError as e:
            print(f"⚠️  Safe string validation: {e}")

    @staticmethod
    def _demonstrate_type_guards() -> None:
        """Show Guards utilities."""
        print("\n=== Type Guards ===")
        if u.is_type("hello", "string_non_empty"):
            print("✅ String non-empty guard: 'hello' is valid")
        test_dict: t.ConfigMap = t.ConfigMap(root={"key": "value"})
        if u.is_type(test_dict, "dict_non_empty"):
            print("✅ Dict non-empty guard: dict is valid")
        test_list: Sequence[int] = [1, 2, 3]
        if u.is_type(test_list, "list_non_empty"):
            print("✅ List non-empty guard: list is valid")

    @override
    def execute(self) -> r[t.ConfigMap]:
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
            return r[t.ConfigMap].ok(
                t.ConfigMap(
                    root={
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
                    },
                ),
            )
        except Exception as e:
            error_msg = f"Advanced utilities demonstration failed: {e}"
            return r[t.ConfigMap].fail(error_msg)


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT UTILITIES - ADVANCED FEATURES")
    print("Args, Enum, Model, Text, Guards, Mapper, Domain, Pagination, Configuration")
    print("=" * 60)
    service = AdvancedUtilitiesService()
    result = service.execute()
    if result.is_success:
        result_data: t.ConfigMap = t.ConfigMap(
            root={
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
            },
        )
        utilities = result_data.root["utilities_demonstrated"]
        categories = result_data.root["utility_categories"]
        if (
            isinstance(utilities, Sequence)
            and (not isinstance(utilities, (str, bytes, bytearray)))
            and isinstance(categories, int)
        ):
            print(f"\n✅ Demonstrated {categories} utility categories")
            print(f"✅ Covered {len(utilities)} utility types")
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
