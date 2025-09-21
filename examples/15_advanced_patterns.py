#!/usr/bin/env python3
"""FLEXT Core - Advanced Examples.

Complex patterns and production scenarios using FLEXT Core.
This module demonstrates complex usage patterns and can be run independently
when flext_core is properly installed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import sys
from decimal import Decimal

from pydantic_settings import SettingsConfigDict

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextResult,
    FlextTypes,
)

OrderData = FlextTypes.Core.Dict
ItemData = FlextTypes.Core.Dict


MAX_ORDER_ITEMS = 100
MIN_ORDER_VALUE = Decimal("0.01")
MAX_ORDER_VALUE = Decimal("100000.00")


class AdvancedExamplesConfig(FlextConfig):
    """Configuration for complex examples with business validation."""

    # Service configuration
    service_name: str = "flext-advanced-examples"
    service_version: str = "1.0.0"
    debug_mode: bool = False

    # Business rule configuration
    max_order_items: int = MAX_ORDER_ITEMS
    min_order_value: Decimal = MIN_ORDER_VALUE
    max_order_value: Decimal = MAX_ORDER_VALUE

    # Security settings
    require_email_verification: bool = True
    password_complexity_required: bool = True

    model_config = SettingsConfigDict(
        extra="allow",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rule constraints.

        Returns:
            FlextResult[None]: Success or failure result

        """
        if self.max_order_items <= 0:
            return FlextResult[None].fail("max_order_items must be positive")

        if self.min_order_value <= 0:
            return FlextResult[None].fail("min_order_value must be positive")

        if self.max_order_value <= self.min_order_value:
            return FlextResult[None].fail(
                "max_order_value must be greater than min_order_value",
            )

        return FlextResult[None].ok(None)


class OrderProcessor:
    """Order processing with validation and error handling."""

    def __init__(self, config: AdvancedExamplesConfig) -> None:
        """Initialize order processor with configuration."""
        self._config = config
        self.container = FlextContainer()
        self.context = FlextContext()
        self.container.register("config", config)

    def create_order(
        self,
        user_id: str,
        items_data: list[ItemData],
    ) -> FlextResult[OrderData]:
        """Create and validate order with business rules.

        Args:
            user_id: ID of the user creating the order
            items_data: List of order items data

        Returns:
            FlextResult[OrderData]: Created order data or error

        """
        return (
            self._validate_items_count(items_data)
            .flat_map(lambda _: self._process_all_items(items_data))
            .flat_map(
                lambda items_and_total: self._validate_and_create_order(
                    user_id,
                    items_and_total[0],
                    items_and_total[1],
                ),
            )
        )

    def _validate_items_count(
        self,
        items_data: list[ItemData],
    ) -> FlextResult[list[ItemData]]:
        """Validate order items count.

        Args:
            items_data: List of order items to validate

        Returns:
            FlextResult[list[ItemData]]: Validated items or error

        """
        if len(items_data) > self._config.max_order_items:
            return FlextResult[list[ItemData]].fail(
                f"Too many items: {len(items_data)} > {self._config.max_order_items}",
            )
        if not items_data:
            return FlextResult[list[ItemData]].fail("Order must have at least one item")
        return FlextResult[list[ItemData]].ok(items_data)

    def _process_all_items(
        self,
        items_data: list[ItemData],
    ) -> FlextResult[tuple[list[ItemData], Decimal]]:
        """Process all items and calculate total.

        Args:
            items_data: List of order items to process

        Returns:
            FlextResult[tuple[list[ItemData], Decimal]]: Processed items and total or error

        """
        processed_items: list[ItemData] = []
        total_amount = Decimal("0.00")

        for item_data in items_data:
            item_result = self._process_order_item(item_data)
            if not item_result.success:
                return FlextResult[tuple[list[ItemData], Decimal]].fail(
                    f"Item validation failed: {item_result.error}",
                )

            item = item_result.value
            processed_items.append(item)
            if "total_price" in item:
                total_amount += Decimal(str(item["total_price"]))

        return FlextResult[tuple[list[ItemData], Decimal]].ok(
            (
                processed_items,
                total_amount,
            ),
        )

    def _validate_and_create_order(
        self,
        user_id: str,
        processed_items: list[ItemData],
        total_amount: Decimal,
    ) -> FlextResult[OrderData]:
        """Validate amount and create final order.

        Args:
            user_id: ID of the user creating the order
            processed_items: List of processed order items
            total_amount: Total amount of the order

        Returns:
            FlextResult[OrderData]: Created order data or error

        """
        amount_validation = self._validate_order_amount(total_amount)
        if not amount_validation.success:
            return FlextResult[OrderData].fail(
                amount_validation.error or "Amount validation failed",
            )

        order: OrderData = {
            "id": f"ORDER-{user_id}-001",
            "user_id": user_id,
            "items": processed_items,
            "total_amount": float(total_amount),
            "status": "created",
        }
        return FlextResult[OrderData].ok(order)

    def _validate_order_amount(self, amount: Decimal) -> FlextResult[None]:
        """Validate order total amount against business rules.

        Args:
            amount: Order amount to validate

        Returns:
            FlextResult[None]: Success or failure result

        """
        if amount < self._config.min_order_value:
            return FlextResult[None].fail(
                f"Order amount too low: {amount} < {self._config.min_order_value}",
            )

        if amount > self._config.max_order_value:
            return FlextResult[None].fail(
                f"Order amount too high: {amount} > {self._config.max_order_value}",
            )

        return FlextResult[None].ok(None)

    def _process_order_item(self, item_data: ItemData) -> FlextResult[ItemData]:
        """Process and validate individual order item using railway pattern.

        Args:
            item_data: Order item data to process

        Returns:
            FlextResult[ItemData]: Processed item data or error

        """
        return self._validate_required_fields(
            item_data,
            ["product_id", "quantity", "unit_price"],
        ).flat_map(lambda _: self._parse_and_validate_item_values(item_data))

    def _validate_required_fields(
        self,
        item_data: ItemData,
        fields: FlextTypes.Core.StringList,
    ) -> FlextResult[ItemData]:
        """Validate required fields exist.

        Args:
            item_data: Item data to validate
            fields: List of required field names

        Returns:
            FlextResult[ItemData]: Validated item data or error

        """
        for field in fields:
            if field not in item_data:
                return FlextResult[ItemData].fail(f"Missing required field: {field}")
        return FlextResult[ItemData].ok(item_data)

    def _parse_and_validate_item_values(
        self,
        item_data: ItemData,
    ) -> FlextResult[ItemData]:
        """Parse and validate item values.

        Args:
            item_data: Item data to parse and validate

        Returns:
            FlextResult[ItemData]: Processed item data or error

        """
        try:
            quantity = int(str(item_data["quantity"]))
            unit_price = Decimal(str(item_data["unit_price"]))

            if quantity <= 0:
                return FlextResult[ItemData].fail("Quantity must be positive")
            if unit_price <= 0:
                return FlextResult[ItemData].fail("Unit price must be positive")

            total_price = unit_price * quantity
            processed_item: ItemData = {
                **item_data,
                "quantity": quantity,
                "unit_price": float(unit_price),
                "total_price": float(total_price),
            }
            return FlextResult[ItemData].ok(processed_item)

        except (ValueError, TypeError) as e:
            return FlextResult[ItemData].fail(f"Invalid item data: {e}")


def demonstrate_order_processing() -> FlextResult[None]:
    """Demonstrate order processing with error handling.

    Returns:
        FlextResult[None]: Success or failure result

    """
    print("Starting order processing demonstration")

    try:
        # Create configuration
        config = AdvancedExamplesConfig(log_level="INFO")
        validation_result = config.validate_business_rules()
        if not validation_result.success:
            return FlextResult[None].fail(
                f"Configuration validation failed: {validation_result.error}",
            )

        # Create order processor
        processor = OrderProcessor(config)

        # Create test order items
        test_items: list[ItemData] = [
            {
                "product_id": "PROD-001",
                "name": "Test Product 1",
                "quantity": 2,
                "unit_price": "29.99",
            },
            {
                "product_id": "PROD-002",
                "name": "Test Product 2",
                "quantity": 1,
                "unit_price": "49.99",
            },
        ]

        # Create order
        order_result = processor.create_order("USER-123", test_items)

        if not order_result.success:
            print(f"Order creation failed: {order_result.error}")
            return FlextResult[None].fail(
                f"Order creation failed: {order_result.error}",
            )

        order = order_result.value
        print(f"Order created successfully: ID={order['id']}")

        return FlextResult[None].ok(None)

    except Exception as e:
        error_msg = f"Demonstration failed with unexpected error: {e}"
        print(error_msg)
        return FlextResult[None].fail(error_msg)


def demonstrate_configuration_validation() -> FlextResult[None]:
    """Demonstrate configuration validation with various scenarios.

    Returns:
        FlextResult[None]: Success or failure result

    """
    print("Starting configuration validation demonstration")

    try:
        # Test valid configuration
        print("Testing valid configuration...")
        valid_config = AdvancedExamplesConfig(log_level="INFO")
        validation_result = valid_config.validate_business_rules()

        if not validation_result.success:
            return FlextResult[None].fail(
                f"Valid configuration failed validation: {validation_result.error}",
            )

        print("✅ Valid configuration passed validation")

        # Test invalid configuration - negative max_order_items
        print("Testing invalid configuration (negative max_order_items)...")
        with contextlib.suppress(Exception):
            invalid_config1 = AdvancedExamplesConfig(
                max_order_items=-1, log_level="INFO"
            )
            invalid_result1 = invalid_config1.validate_business_rules()

            if invalid_result1.success:
                print("⚠️  Invalid configuration unexpectedly passed validation")
            else:
                print(
                    f"✅ Invalid configuration correctly failed: "
                    f"{invalid_result1.error}",
                )

        print("Configuration validation demonstration completed successfully")
        return FlextResult[None].ok(None)

    except Exception as e:
        error_msg = f"Configuration demonstration failed: {e}"
        print(error_msg)
        return FlextResult[None].fail(error_msg)


def main() -> int:
    """Main execution function with error handling.

    Returns:
        int: Exit code (0 for success, 1 for failure)

    """
    print("Starting FLEXT Core Advanced Examples")

    try:
        # Run configuration demonstration
        config_result = demonstrate_configuration_validation()
        if not config_result.success:
            print(f"Configuration demonstration failed: {config_result.error}")
            return 1

        # Run order processing demonstration
        order_result = demonstrate_order_processing()
        if not order_result.success:
            print(f"Order processing demonstration failed: {order_result.error}")
            return 1

        print("✅ All demonstrations completed successfully")
        return 0

    except Exception as e:
        print(f"❌ Demonstration failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
