"""Main demonstration runner for FLEXT utilities.

Orchestrates utility demonstrations following SOLID principles
with helper classes to reduce complexity.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import time
from decimal import Decimal

from examples.shared_domain import SharedDomainFactory
from examples.utilities.complexity_helpers import (
    DemonstrationSectionHelper,
    ValidationHelper,
)
from examples.utilities.domain_models import UtilityDemoUser
from examples.utilities.formatting_helpers import (
    generate_hash_id,
    generate_prefixed_id,
    generate_short_id,
)
from examples.utilities.validation_utilities import (
    calculate_discount_price,
)
from flext_core import FlextTypes, FlextUtilities

# =============================================================================
# DEMONSTRATION FUNCTIONS - Core utility demonstrations
# =============================================================================


def demonstrate_id_generation() -> None:
    """Demonstrate various ID generation strategies using helper.

    Reduced complexity through utility functions.
    """
    DemonstrationSectionHelper.print_section_header(1, "ID Generation Strategies")

    # Basic entity ID generation
    entity_id: FlextTypes.Domain.EntityId = FlextUtilities.generate_entity_id()
    DemonstrationSectionHelper.log_operation("Entity ID", entity_id)

    # UUID generation
    uuid_id: FlextTypes.Domain.EntityId = FlextUtilities.generate_uuid()
    DemonstrationSectionHelper.log_operation("UUID", uuid_id)

    # Correlation ID generation
    correlation_id: FlextTypes.Domain.EntityId = (
        FlextUtilities.generate_correlation_id()
    )
    DemonstrationSectionHelper.log_operation("Correlation ID", correlation_id)

    # Custom prefixed ID
    prefixed_id: FlextTypes.Domain.EntityId = generate_prefixed_id("USER", 8)
    DemonstrationSectionHelper.log_operation("Prefixed ID", prefixed_id)

    # Hash-based ID
    hash_id: FlextTypes.Domain.EntityId = generate_hash_id("user@example.com")
    DemonstrationSectionHelper.log_operation("Hash ID", hash_id)

    # Short ID
    short_id: FlextTypes.Domain.EntityId = generate_short_id(6)
    DemonstrationSectionHelper.log_operation("Short ID", short_id)

    DemonstrationSectionHelper.log_success("ID generation demonstration completed")


def demonstrate_timestamp_generation() -> None:
    """Demonstrate timestamp generation and formatting."""
    DemonstrationSectionHelper.print_section_header(2, "Timestamp Generation")

    # ISO timestamp generation
    iso_timestamp = FlextUtilities.generate_iso_timestamp()
    DemonstrationSectionHelper.log_operation("ISO Timestamp", iso_timestamp)

    # Unix timestamp
    unix_timestamp = int(time.time())
    DemonstrationSectionHelper.log_operation("Unix Timestamp", unix_timestamp)

    # Formatted timestamp
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    DemonstrationSectionHelper.log_operation("Formatted Time", formatted_time)

    DemonstrationSectionHelper.log_success("Timestamp demonstration completed")


def demonstrate_type_checking() -> None:
    """Demonstrate type checking utilities."""
    DemonstrationSectionHelper.print_section_header(3, "Type Checking Utilities")

    # Test various types
    test_values = [
        ("Hello World", "string"),
        (42, "integer"),
        ([1, 2, 3], "list"),
        ({"key": "value"}, "dict"),
        (math.pi, "float"),
    ]

    for _value, _expected_type in test_values:
        DemonstrationSectionHelper.print_separator()

        # Test type checking functions

    DemonstrationSectionHelper.log_success("Type checking demonstration completed")


def demonstrate_validation() -> None:
    """Demonstrate data validation patterns."""
    DemonstrationSectionHelper.print_section_header(4, "Data Validation")

    # Test user data validation
    valid_user_data: dict[str, object] = {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
    }

    invalid_user_data: dict[str, object] = {
        "name": "",
        "email": "invalid-email",
        "age": "not-a-number",
    }

    valid_errors = ValidationHelper.validate_user_data(valid_user_data)
    ValidationHelper.report_validation_result(valid_errors)

    DemonstrationSectionHelper.print_separator()

    invalid_errors = ValidationHelper.validate_user_data(invalid_user_data)
    ValidationHelper.report_validation_result(invalid_errors)

    DemonstrationSectionHelper.log_success("Validation demonstration completed")


def demonstrate_shared_domain_usage() -> None:
    """Demonstrate shared domain models with enhanced mixins."""
    DemonstrationSectionHelper.print_section_header(5, "Shared Domain Models")

    # Create user using shared domain
    user_result = SharedDomainFactory.create_user(
        name="John Doe",
        email="john.doe@example.com",
        age=30,
    )

    if not user_result.success:
        DemonstrationSectionHelper.log_error(
            f"Failed to create user: {user_result.error}",
        )
        return

    shared_user = user_result.value
    if shared_user is None:
        DemonstrationSectionHelper.log_error("User data is None")
        return

    enhanced_user = UtilityDemoUser(
        id=shared_user.id,
        name=shared_user.name,
        email_address=shared_user.email_address,
        age=shared_user.age,
        phone=shared_user.phone,
        status=shared_user.status,
        created_at=shared_user.created_at,
    )

    # Demonstrate enhanced functionality

    # Serialize the user
    enhanced_user.to_serializable()

    DemonstrationSectionHelper.log_success("Shared domain demonstration completed")


def demonstrate_business_logic() -> None:
    """Demonstrate business logic with proper error handling."""
    DemonstrationSectionHelper.print_section_header(6, "Business Logic Functions")

    # Create a product for discount calculation
    product_result = SharedDomainFactory.create_product(
        name="Sample Product",
        price_amount=Decimal("100.00"),
        description="Test product for discount calculation",
    )

    if not product_result.success:
        DemonstrationSectionHelper.log_error(
            f"Failed to create product: {product_result.error}",
        )
        return

    product = product_result.value
    if product is None:
        DemonstrationSectionHelper.log_error("Product data is None")
        return

    # Test discount calculation
    discount_result = calculate_discount_price(product, 20.0)
    if discount_result.success:
        final_price = discount_result.value
        if final_price is not None:
            pass
    else:
        DemonstrationSectionHelper.log_error(
            f"Discount calculation failed: {discount_result.error}",
        )

    # Test invalid discount
    invalid_discount_result = calculate_discount_price(product, 150.0)
    if not invalid_discount_result.success:
        DemonstrationSectionHelper.log_error(
            f"Expected error: {invalid_discount_result.error}",
        )

    DemonstrationSectionHelper.log_success("Business logic demonstration completed")


def run_all_demonstrations() -> None:
    """Main function to run all utility demonstrations."""
    try:
        demonstrate_id_generation()
        demonstrate_timestamp_generation()
        demonstrate_type_checking()
        demonstrate_validation()
        demonstrate_shared_domain_usage()
        demonstrate_business_logic()

        DemonstrationSectionHelper.log_success(
            "ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!",
        )

    except Exception as e:
        DemonstrationSectionHelper.log_error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    run_all_demonstrations()
