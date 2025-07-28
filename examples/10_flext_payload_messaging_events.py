#!/usr/bin/env python3
"""FLEXT Payload Messaging and Events Example.

Comprehensive demonstration of FlextPayload system showing enterprise-grade
message and event patterns for structured data transport, validation, and metadata management.

Features demonstrated:
    - Generic payload containers with type safety
    - Message payloads with level validation and source tracking
    - Domain event payloads with aggregate tracking and versioning
    - Metadata management and enrichment patterns
    - Payload validation and error handling
    - Enterprise messaging patterns for distributed systems
    - Event sourcing foundations for domain-driven design

Key Components:
    - FlextPayload[T]: Generic type-safe payload container
    - FlextMessage: Specialized string message payload with levels
    - FlextEvent: Domain event payload with aggregate correlation
    - Payload factory methods with comprehensive validation
    - Metadata operations for transport context and debugging
    - Serialization support for cross-service communication

This example shows real-world enterprise messaging scenarios
demonstrating the power and flexibility of the FlextPayload system.
"""

import time

from flext_core.payload import FlextEvent, FlextMessage, FlextPayload


def demonstrate_generic_payloads() -> None:
    """Demonstrate generic payload containers with type safety."""
    print("\n" + "=" * 80)
    print("📦 GENERIC PAYLOAD CONTAINERS")
    print("=" * 80)

    # 1. Basic payload creation
    print("\n1. Basic payload creation:")

    # Simple string payload
    text_payload = FlextPayload(data="Hello, World!")
    print(f"✅ Text payload: {text_payload}")
    print(f"   Data: {text_payload.data}")
    print(f"   Metadata: {text_payload.metadata}")

    # Dictionary payload with metadata
    user_data = {"id": "user123", "name": "John Doe", "email": "john@example.com"}
    user_payload = FlextPayload(
        data=user_data,
        metadata={
            "source": "user_service",
            "version": "1.0",
            "timestamp": time.time(),
        },
    )
    print(f"✅ User payload: {user_payload}")

    # 2. Type-safe payload creation with factory method
    print("\n2. Type-safe payload creation:")

    # Create payload with validation
    order_data = {
        "order_id": "ORD001",
        "customer_id": "CUST123",
        "items": [
            {"product": "laptop", "quantity": 1, "price": 999.99},
            {"product": "mouse", "quantity": 2, "price": 29.99},
        ],
        "total": 1059.97,
    }

    order_result = FlextPayload.create(
        order_data,
        source="order_service",
        correlation_id="req_456",
        processing_stage="created",
    )

    if order_result.is_success:
        order_payload = order_result.data
        print(f"✅ Order payload created: {order_payload}")
        print(f"   Order ID: {order_payload.data.get('order_id')}")
        print(f"   Total: ${order_payload.data.get('total')}")
        print(f"   Source: {order_payload.get_metadata('source')}")
    else:
        print(f"❌ Order payload creation failed: {order_result.error}")

    # 3. Payload metadata operations
    print("\n3. Payload metadata operations:")

    if order_result.is_success:
        order_payload = order_result.data

        # Add processing metadata
        enriched_payload = order_payload.with_metadata(
            processed_at=time.time(),
            processor_id="worker_001",
            validation_status="passed",
            estimated_delivery="2025-01-30",
        )

        print("📋 Enhanced payload metadata:")
        for key in ["source", "processed_at", "processor_id", "validation_status"]:
            value = enriched_payload.get_metadata(key)
            print(f"   {key}: {value}")

        # Check metadata existence
        has_correlation = enriched_payload.has_metadata("correlation_id")
        has_delivery = enriched_payload.has_metadata("estimated_delivery")
        print(f"   Has correlation ID: {has_correlation}")
        print(f"   Has delivery estimate: {has_delivery}")

    # 4. Complex payload with nested data
    print("\n4. Complex payload with nested data:")

    analytics_data = {
        "event_type": "user_action",
        "session": {
            "id": "sess_789",
            "user_id": "user123",
            "start_time": time.time() - 3600,
            "current_page": "/dashboard",
        },
        "action": {
            "type": "button_click",
            "element_id": "export_button",
            "coordinates": {"x": 150, "y": 300},
            "timestamp": time.time(),
        },
        "context": {
            "browser": "Chrome",
            "platform": "macOS",
            "screen_resolution": "1920x1080",
            "referrer": "/reports",
        },
    }

    analytics_result = FlextPayload.create(
        analytics_data,
        tenant_id="tenant_001",
        service="analytics_collector",
        version="2.1.0",
        batch_id="batch_456",
    )

    if analytics_result.is_success:
        analytics_payload = analytics_result.data
        print("✅ Analytics payload created successfully")
        print(f"   Event type: {analytics_payload.data.get('event_type')}")
        print(f"   User: {analytics_payload.data.get('session', {}).get('user_id')}")
        print(f"   Action: {analytics_payload.data.get('action', {}).get('type')}")
        print(f"   Service: {analytics_payload.get_metadata('service')}")
    else:
        print(f"❌ Analytics payload failed: {analytics_result.error}")


def demonstrate_message_payloads() -> None:
    """Demonstrate specialized message payloads with level validation."""
    print("\n" + "=" * 80)
    print("💬 MESSAGE PAYLOADS WITH LEVEL VALIDATION")
    print("=" * 80)

    # 1. Basic message creation
    print("\n1. Basic message creation:")

    # Info message
    info_result = FlextMessage.create_message("System startup completed successfully")
    if info_result.is_success:
        info_msg = info_result.data
        print(f"✅ Info message: {info_msg.data}")
        print(f"   Level: {info_msg.get_metadata('level')}")
    else:
        print(f"❌ Info message failed: {info_result.error}")

    # Warning message with source
    warning_result = FlextMessage.create_message(
        "Database connection pool approaching capacity",
        level="warning",
        source="database_service",
    )
    if warning_result.is_success:
        warning_msg = warning_result.data
        print(f"⚠️ Warning message: {warning_msg.data}")
        print(f"   Level: {warning_msg.get_metadata('level')}")
        print(f"   Source: {warning_msg.get_metadata('source')}")
    else:
        print(f"❌ Warning message failed: {warning_result.error}")

    # 2. Error messages with detailed context
    print("\n2. Error messages with detailed context:")

    error_result = FlextMessage.create_message(
        "Payment processing failed: Invalid credit card number",
        level="error",
        source="payment_gateway",
    )
    if error_result.is_success:
        error_msg = error_result.data

        # Add detailed error context
        detailed_error = error_msg.with_metadata(
            error_code="PAY_001",
            transaction_id="txn_789",
            customer_id="cust_456",
            retry_count=3,
            timestamp=time.time(),
        )

        print(f"❌ Error message: {detailed_error.data}")
        print(f"   Error code: {detailed_error.get_metadata('error_code')}")
        print(f"   Transaction: {detailed_error.get_metadata('transaction_id')}")
        print(f"   Customer: {detailed_error.get_metadata('customer_id')}")
        print(f"   Retries: {detailed_error.get_metadata('retry_count')}")
    else:
        print(f"❌ Error message creation failed: {error_result.error}")

    # 3. Debug messages for development
    print("\n3. Debug messages for development:")

    debug_result = FlextMessage.create_message(
        "Cache hit ratio: 85%, average response time: 45ms",
        level="debug",
        source="cache_monitor",
    )
    if debug_result.is_success:
        debug_msg = debug_result.data

        # Add performance metrics
        perf_debug = debug_msg.with_metadata(
            cache_hits=8500,
            cache_misses=1500,
            avg_response_ms=45,
            memory_usage_mb=256,
            uptime_hours=72,
        )

        print(f"🔍 Debug message: {perf_debug.data}")
        print(f"   Cache hits: {perf_debug.get_metadata('cache_hits')}")
        print(f"   Avg response: {perf_debug.get_metadata('avg_response_ms')}ms")
        print(f"   Memory: {perf_debug.get_metadata('memory_usage_mb')}MB")
    else:
        print(f"❌ Debug message failed: {debug_result.error}")

    # 4. Critical messages requiring immediate attention
    print("\n4. Critical messages:")

    critical_result = FlextMessage.create_message(
        "Primary database server is down, failover initiated",
        level="critical",
        source="health_monitor",
    )
    if critical_result.is_success:
        critical_msg = critical_result.data

        # Add incident context
        incident_msg = critical_msg.with_metadata(
            incident_id="INC_001",
            severity="P1",
            affected_services=["user_service", "order_service", "payment_service"],
            failover_time_seconds=15,
            estimated_recovery="30 minutes",
            escalated_to="on_call_engineer",
        )

        print(f"🚨 Critical message: {incident_msg.data}")
        print(f"   Incident ID: {incident_msg.get_metadata('incident_id')}")
        print(f"   Severity: {incident_msg.get_metadata('severity')}")
        print(
            f"   Affected services: {len(incident_msg.get_metadata('affected_services', []))} services",
        )
        print(
            f"   Recovery estimate: {incident_msg.get_metadata('estimated_recovery')}",
        )
    else:
        print(f"❌ Critical message failed: {critical_result.error}")

    # 5. Message validation and error handling
    print("\n5. Message validation:")

    # Test empty message (should fail)
    empty_result = FlextMessage.create_message("")
    if empty_result.is_success:
        print("⚠️ Empty message was created (unexpected)")
    else:
        print(f"✅ Empty message rejected: {empty_result.error}")

    # Test invalid level (should use default)
    invalid_level_result = FlextMessage.create_message(
        "Test message with invalid level",
        level="invalid_level",
    )
    if invalid_level_result.is_success:
        msg = invalid_level_result.data
        print(f"✅ Invalid level handled, used: {msg.get_metadata('level')}")
    else:
        print(f"❌ Message with invalid level failed: {invalid_level_result.error}")


def demonstrate_domain_events() -> None:
    """Demonstrate domain event payloads with aggregate tracking."""
    print("\n" + "=" * 80)
    print("🎯 DOMAIN EVENT PAYLOADS FOR EVENT SOURCING")
    print("=" * 80)

    # 1. User registration event
    print("\n1. User registration domain event:")

    user_registered_data = {
        "user_id": "user_123",
        "email": "alice@example.com",
        "first_name": "Alice",
        "last_name": "Johnson",
        "registration_method": "email",
        "terms_accepted": True,
        "marketing_consent": False,
    }

    user_event_result = FlextEvent.create_event(
        event_type="UserRegistered",
        event_data=user_registered_data,
        aggregate_id="user_123",
        version=1,
    )

    if user_event_result.is_success:
        user_event = user_event_result.data
        print("✅ User registration event created")
        print(f"   Event type: {user_event.get_metadata('event_type')}")
        print(f"   User ID: {user_event.data.get('user_id')}")
        print(f"   Email: {user_event.data.get('email')}")
        print(f"   Aggregate: {user_event.get_metadata('aggregate_id')}")
        print(f"   Version: {user_event.get_metadata('version')}")
    else:
        print(f"❌ User event creation failed: {user_event_result.error}")

    # 2. Order processing events with enrichment
    print("\n2. Order processing event chain:")

    # Order created event
    order_created_data = {
        "order_id": "order_456",
        "customer_id": "user_123",
        "items": [
            {"sku": "LAPTOP_001", "quantity": 1, "unit_price": 999.99},
            {"sku": "MOUSE_002", "quantity": 1, "unit_price": 29.99},
        ],
        "subtotal": 1029.98,
        "tax": 82.40,
        "total": 1112.38,
        "currency": "USD",
    }

    order_created_result = FlextEvent.create_event(
        event_type="OrderCreated",
        event_data=order_created_data,
        aggregate_id="order_456",
        version=1,
    )

    if order_created_result.is_success:
        order_event = order_created_result.data

        # Enrich with processing metadata
        enriched_order_event = order_event.with_metadata(
            created_by="customer",
            channel="web",
            session_id="sess_789",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 Chrome/91.0",
            correlation_id="req_12345",
            processing_timestamp=time.time(),
        )

        print("✅ Order created event:")
        print(f"   Order ID: {enriched_order_event.data.get('order_id')}")
        print(f"   Customer: {enriched_order_event.data.get('customer_id')}")
        print(f"   Total: ${enriched_order_event.data.get('total')}")
        print(f"   Channel: {enriched_order_event.get_metadata('channel')}")
        print(f"   Correlation: {enriched_order_event.get_metadata('correlation_id')}")
    else:
        print(f"❌ Order event creation failed: {order_created_result.error}")

    # Order paid event (version 2)
    payment_data = {
        "order_id": "order_456",
        "payment_method": "credit_card",
        "card_last_four": "1234",
        "amount_charged": 1112.38,
        "currency": "USD",
        "transaction_id": "txn_789",
        "payment_gateway": "stripe",
        "authorization_code": "AUTH123456",
    }

    order_paid_result = FlextEvent.create_event(
        event_type="OrderPaid",
        event_data=payment_data,
        aggregate_id="order_456",
        version=2,
    )

    if order_paid_result.is_success:
        payment_event = order_paid_result.data
        print("✅ Order payment event:")
        print(f"   Transaction: {payment_event.data.get('transaction_id')}")
        print(f"   Amount: ${payment_event.data.get('amount_charged')}")
        print(f"   Method: {payment_event.data.get('payment_method')}")
        print(f"   Version: {payment_event.get_metadata('version')}")
    else:
        print(f"❌ Payment event creation failed: {order_paid_result.error}")

    # 3. Inventory events for stock management
    print("\n3. Inventory management events:")

    inventory_updated_data = {
        "sku": "LAPTOP_001",
        "previous_quantity": 50,
        "new_quantity": 49,
        "change_quantity": -1,
        "change_reason": "sale",
        "warehouse_id": "WH_001",
        "location": "A-15-3",
        "reserved_quantity": 5,
        "available_quantity": 44,
    }

    inventory_result = FlextEvent.create_event(
        event_type="InventoryUpdated",
        event_data=inventory_updated_data,
        aggregate_id="inventory_LAPTOP_001",
        version=147,  # High version number indicating many updates
    )

    if inventory_result.is_success:
        inventory_event = inventory_result.data

        # Add operational context
        ops_inventory_event = inventory_event.with_metadata(
            triggered_by="order_fulfillment",
            operator_id="system",
            batch_operation=False,
            audit_trail_id="audit_789",
            low_stock_alert=inventory_event.data.get("available_quantity", 0) < 10,
        )

        print("✅ Inventory update event:")
        print(f"   SKU: {ops_inventory_event.data.get('sku')}")
        print(f"   Change: {ops_inventory_event.data.get('change_quantity')}")
        print(f"   Available: {ops_inventory_event.data.get('available_quantity')}")
        print(f"   Version: {ops_inventory_event.get_metadata('version')}")
        print(
            f"   Low stock alert: {ops_inventory_event.get_metadata('low_stock_alert')}",
        )
    else:
        print(f"❌ Inventory event creation failed: {inventory_result.error}")

    # 4. Event validation and error handling
    print("\n4. Event validation:")

    # Test invalid event type
    invalid_event_result = FlextEvent.create_event(
        event_type="",  # Empty event type
        event_data={"test": "data"},
    )
    if invalid_event_result.is_success:
        print("⚠️ Invalid event type was accepted (unexpected)")
    else:
        print(f"✅ Empty event type rejected: {invalid_event_result.error}")

    # Test invalid aggregate ID
    invalid_aggregate_result = FlextEvent.create_event(
        event_type="TestEvent",
        event_data={"test": "data"},
        aggregate_id="",  # Empty aggregate ID
    )
    if invalid_aggregate_result.is_success:
        print("⚠️ Invalid aggregate ID was accepted (unexpected)")
    else:
        print(f"✅ Empty aggregate ID rejected: {invalid_aggregate_result.error}")

    # Test negative version
    negative_version_result = FlextEvent.create_event(
        event_type="TestEvent",
        event_data={"test": "data"},
        version=-1,  # Negative version
    )
    if negative_version_result.is_success:
        print("⚠️ Negative version was accepted (unexpected)")
    else:
        print(f"✅ Negative version rejected: {negative_version_result.error}")


def demonstrate_payload_serialization() -> None:
    """Demonstrate payload serialization for cross-service communication."""
    print("\n" + "=" * 80)
    print("📡 PAYLOAD SERIALIZATION FOR CROSS-SERVICE COMMUNICATION")
    print("=" * 80)

    # 1. Basic payload serialization
    print("\n1. Basic payload serialization:")

    service_request = {
        "request_id": "req_12345",
        "service": "user_profile",
        "operation": "get_profile",
        "parameters": {"user_id": "user_123", "include_preferences": True},
        "timeout_ms": 5000,
    }

    request_payload_result = FlextPayload.create(
        service_request,
        source_service="api_gateway",
        target_service="user_service",
        trace_id="trace_456",
        span_id="span_789",
    )

    if request_payload_result.is_success:
        request_payload = request_payload_result.data

        # Serialize to dictionary
        serialized = request_payload.to_dict()
        print("✅ Serialized payload:")
        print(f"   Data keys: {list(serialized['data'].keys())}")
        print(f"   Metadata keys: {list(serialized['metadata'].keys())}")
        print(f"   Request ID: {serialized['data']['request_id']}")
        print(f"   Source: {serialized['metadata']['source_service']}")
    else:
        print(f"❌ Request payload creation failed: {request_payload_result.error}")

    # 2. Message serialization for logging
    print("\n2. Message serialization for logging:")

    audit_message_result = FlextMessage.create_message(
        "User profile accessed by administrator",
        level="info",
        source="audit_service",
    )

    if audit_message_result.is_success:
        audit_message = audit_message_result.data

        # Add audit context
        audit_with_context = audit_message.with_metadata(
            audit_id="audit_001",
            admin_user_id="admin_456",
            target_user_id="user_123",
            action="profile_view",
            ip_address="10.0.1.50",
            session_duration_minutes=15,
            compliance_category="data_access",
        )

        # Serialize for structured logging
        log_data = audit_with_context.to_dict()
        print("✅ Audit log serialization:")
        print(f"   Message: {log_data['data']}")
        print(f"   Audit ID: {log_data['metadata']['audit_id']}")
        print(f"   Admin: {log_data['metadata']['admin_user_id']}")
        print(f"   Action: {log_data['metadata']['action']}")
    else:
        print(f"❌ Audit message creation failed: {audit_message_result.error}")

    # 3. Event serialization for event store
    print("\n3. Event serialization for event store:")

    customer_updated_data = {
        "customer_id": "cust_789",
        "changes": {
            "email": {"old": "old@example.com", "new": "new@example.com"},
            "phone": {"old": "+1-555-0100", "new": "+1-555-0200"},
            "address": {
                "old": {"street": "123 Old St", "city": "Old City"},
                "new": {"street": "456 New Ave", "city": "New City"},
            },
        },
        "change_reason": "customer_request",
        "verified": True,
    }

    customer_event_result = FlextEvent.create_event(
        event_type="CustomerUpdated",
        event_data=customer_updated_data,
        aggregate_id="customer_789",
        version=5,
    )

    if customer_event_result.is_success:
        customer_event = customer_event_result.data

        # Add event store metadata
        event_store_ready = customer_event.with_metadata(
            stream_name="customer_789",
            expected_version=4,
            causation_id="cmd_123",
            correlation_id="process_456",
            event_store_timestamp=time.time(),
            checkpoint_position=10047,
        )

        # Serialize for event store persistence
        event_store_data = event_store_ready.to_dict()
        print("✅ Event store serialization:")
        print(f"   Event type: {event_store_data['metadata']['event_type']}")
        print(f"   Stream: {event_store_data['metadata']['stream_name']}")
        print(f"   Version: {event_store_data['metadata']['version']}")
        print(f"   Changes: {len(event_store_data['data']['changes'])} fields")
        print(f"   Checkpoint: {event_store_data['metadata']['checkpoint_position']}")
    else:
        print(f"❌ Customer event creation failed: {customer_event_result.error}")


def demonstrate_enterprise_messaging_patterns() -> None:
    """Demonstrate enterprise messaging patterns and best practices."""
    print("\n" + "=" * 80)
    print("🏢 ENTERPRISE MESSAGING PATTERNS AND BEST PRACTICES")
    print("=" * 80)

    # 1. Request-Response pattern
    print("\n1. Request-Response messaging pattern:")

    # Service request
    calculation_request = {
        "operation": "risk_assessment",
        "parameters": {
            "customer_id": "cust_123",
            "loan_amount": 250000,
            "loan_term_years": 30,
            "down_payment": 50000,
            "credit_score": 750,
            "annual_income": 85000,
        },
    }

    request_result = FlextPayload.create(
        calculation_request,
        message_type="request",
        service="risk_engine",
        version="2.1.0",
        timeout_seconds=30,
        reply_to="risk_responses",
        request_id="risk_req_001",
    )

    if request_result.is_success:
        request = request_result.data
        print("📤 Risk assessment request:")
        print(f"   Customer: {request.data['parameters']['customer_id']}")
        print(f"   Loan amount: ${request.data['parameters']['loan_amount']:,}")
        print(f"   Request ID: {request.get_metadata('request_id')}")
        print(f"   Reply to: {request.get_metadata('reply_to')}")
    else:
        print(f"❌ Request creation failed: {request_result.error}")

    # Service response
    risk_response = {
        "request_id": "risk_req_001",
        "status": "completed",
        "result": {
            "risk_level": "low",
            "approval_probability": 0.92,
            "recommended_rate": 3.25,
            "conditions": ["income_verification", "appraisal"],
            "max_approved_amount": 300000,
        },
        "processing_time_ms": 1250,
    }

    response_result = FlextPayload.create(
        risk_response,
        message_type="response",
        service="risk_engine",
        correlation_id="risk_req_001",
        processing_node="risk_node_03",
        cache_hit=False,
    )

    if response_result.is_success:
        response = response_result.data
        print("📥 Risk assessment response:")
        print(f"   Risk level: {response.data['result']['risk_level']}")
        print(
            f"   Approval probability: {response.data['result']['approval_probability']:.0%}",
        )
        print(f"   Processing time: {response.data['processing_time_ms']}ms")
        print(f"   Node: {response.get_metadata('processing_node')}")
    else:
        print(f"❌ Response creation failed: {response_result.error}")

    # 2. Event streaming pattern
    print("\n2. Event streaming pattern:")

    # Real-time transaction event
    transaction_event_data = {
        "transaction_id": "txn_001",
        "account_id": "acc_456",
        "amount": -125.50,
        "merchant": "Coffee Shop Downtown",
        "category": "food_beverage",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "card_present": True,
        "authorization_code": "AUTH789",
    }

    transaction_event_result = FlextEvent.create_event(
        event_type="TransactionAuthorized",
        event_data=transaction_event_data,
        aggregate_id="acc_456",
        version=1047,
    )

    if transaction_event_result.is_success:
        transaction_event = transaction_event_result.data

        # Add streaming metadata
        stream_event = transaction_event.with_metadata(
            stream="transactions",
            partition_key="acc_456",
            sequence_number=10047891,
            producer_id="pos_terminal_001",
            timestamp=time.time(),
            headers={"content-type": "application/json", "schema-version": "1.0"},
        )

        print("🌊 Transaction stream event:")
        print(f"   Transaction: {stream_event.data['transaction_id']}")
        print(f"   Amount: ${abs(stream_event.data['amount'])}")
        print(f"   Merchant: {stream_event.data['merchant']}")
        print(f"   Partition: {stream_event.get_metadata('partition_key')}")
        print(f"   Sequence: {stream_event.get_metadata('sequence_number')}")
    else:
        print(f"❌ Transaction event creation failed: {transaction_event_result.error}")

    # 3. Command pattern for CQRS
    print("\n3. Command pattern for CQRS:")

    create_account_command = {
        "command_id": "cmd_create_account_001",
        "customer_id": "cust_789",
        "account_type": "checking",
        "initial_deposit": 1000.00,
        "overdraft_protection": True,
        "statements_electronic": True,
        "beneficiary": "Jane Doe",
    }

    command_result = FlextPayload.create(
        create_account_command,
        message_type="command",
        command_type="CreateAccount",
        aggregate_type="Account",
        expected_version=0,
        idempotency_key="idem_001",
        user_id="user_789",
        role="customer",
    )

    if command_result.is_success:
        command = command_result.data

        # Add command execution metadata
        executable_command = command.with_metadata(
            handler="AccountCommandHandler",
            validation_rules=["min_deposit", "customer_verification", "account_limits"],
            execution_priority="normal",
            max_retry_attempts=3,
            timeout_seconds=60,
        )

        print("⚡ Account creation command:")
        print(f"   Command: {executable_command.data['command_id']}")
        print(f"   Customer: {executable_command.data['customer_id']}")
        print(f"   Deposit: ${executable_command.data['initial_deposit']}")
        print(f"   Handler: {executable_command.get_metadata('handler')}")
        print(f"   Idempotency: {executable_command.get_metadata('idempotency_key')}")
    else:
        print(f"❌ Command creation failed: {command_result.error}")

    # 4. Error handling in messaging
    print("\n4. Error handling in messaging:")

    # Failed processing notification
    error_notification_result = FlextMessage.create_message(
        "Payment processing failed due to insufficient funds",
        level="error",
        source="payment_processor",
    )

    if error_notification_result.is_success:
        error_msg = error_notification_result.data

        # Add comprehensive error context
        detailed_error = error_msg.with_metadata(
            error_code="INSUFFICIENT_FUNDS",
            transaction_id="txn_failed_001",
            customer_id="cust_123",
            account_balance=45.50,
            attempted_amount=125.00,
            retry_strategy="none",
            customer_notified=True,
            fraud_check_passed=True,
        )

        print("💥 Payment failure notification:")
        print(f"   Error: {detailed_error.get_metadata('error_code')}")
        print(f"   Transaction: {detailed_error.get_metadata('transaction_id')}")
        print(f"   Balance: ${detailed_error.get_metadata('account_balance')}")
        print(f"   Attempted: ${detailed_error.get_metadata('attempted_amount')}")
        print(
            f"   Customer notified: {detailed_error.get_metadata('customer_notified')}",
        )
    else:
        print(
            f"❌ Error notification creation failed: {error_notification_result.error}",
        )


def main() -> None:
    """Execute all FlextPayload demonstrations."""
    print("🚀 FLEXT PAYLOAD - MESSAGING AND EVENTS EXAMPLE")
    print("Demonstrating comprehensive payload patterns for enterprise messaging")

    try:
        demonstrate_generic_payloads()
        demonstrate_message_payloads()
        demonstrate_domain_events()
        demonstrate_payload_serialization()
        demonstrate_enterprise_messaging_patterns()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT PAYLOAD DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of patterns demonstrated:")
        print("   📦 Generic payload containers with type safety and metadata")
        print("   💬 Message payloads with level validation and source tracking")
        print("   🎯 Domain event payloads for event sourcing and DDD")
        print("   📡 Payload serialization for cross-service communication")
        print("   🏢 Enterprise messaging patterns (request-response, streaming, CQRS)")
        print("   💥 Comprehensive error handling and validation")
        print("\n💡 FlextPayload provides enterprise-grade messaging infrastructure")
        print(
            "   with type safety, validation, metadata management, and serialization!",
        )

    except Exception as e:
        print(f"\n❌ Error during FlextPayload demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
