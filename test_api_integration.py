#!/usr/bin/env python3
"""Teste real de integração das APIs públicas do flext-core."""
from __future__ import annotations

from flext_core import *


def test_core_apis() -> bool:
    """Teste das APIs principais."""
    # 1. FlextResult
    result = FlextResult.ok("test_data")
    assert result.is_success
    assert result.data == "test_data"

    # 2. FlextContainer
    container = FlextContainer()
    reg_result = container.register("test_service", "service_instance")
    assert reg_result.is_success

    get_result = container.get("test_service")
    assert get_result.is_success
    assert get_result.data == "service_instance"

    # 3. Validators
    email_result = validate_email("user@example.com")
    assert email_result.is_success

    invalid_email = validate_email("invalid")
    assert invalid_email.is_failure

    # 4. Validation Chain
    chain_result = validate("test").validate_with(NotEmptyValidator()).result()
    assert chain_result.is_success

    empty_chain = validate("").validate_with(NotEmptyValidator()).result()
    assert empty_chain.is_failure

    # 5. Builders
    config = FlextConfigBuilder().set("key", "value").build()
    assert config.is_success

    # 6. Domain objects (basic creation)
    # Não testamos funcionalidade completa pois precisaria de implementações
    assert FlextEntity is not None
    assert FlextValueObject is not None

    return True

if __name__ == "__main__":
    try:
        test_core_apis()
    except Exception:
        raise
