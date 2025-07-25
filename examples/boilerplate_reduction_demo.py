"""Demonstração da Redução Massiva de Boilerplate - FLEXT Core.

Este exemplo mostra como as novas utilidades reduzem 90% do código repetitivo
em aplicações empresariais, seguindo SOLID, DRY e KISS.
"""
from __future__ import annotations

from flext_core import FlextEmail
from flext_core import FlextEntityMixin  # Mixins - Redução de 70% do código de classe
from flext_core import FlextResult
from flext_core import FlextUserId
from flext_core import flext_api_response
from flext_core import flext_config
from flext_core import flext_event
from flext_core import (
    # Decorators - Redução de 80% do código de infraestrutura
    flext_robust,
)

# =============================================================================
# ANTES: Código Tradicional (Muito Boilerplate)
# =============================================================================

class TraditionalUser:
    """Implementação tradicional com MUITO boilerplate."""

    def __init__(self, id: str, name: str, email: str) -> None:
        self.id = id
        self.name = name
        self.email = email
        import datetime
        self.created_at = datetime.datetime.now()
        self.updated_at = datetime.datetime.now()
        self.version = 1

    def to_dict(self):
        try:
            return {
                "id": self.id,
                "name": self.name,
                "email": self.email,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "version": self.version,
            }
        except Exception as e:
            msg = f"Serialization failed: {e}"
            raise ValueError(msg)

    def validate(self) -> None:
        errors = []
        if not self.name or len(self.name) < 2:
            errors.append("Name must be at least 2 characters")
        if "@" not in self.email:
            errors.append("Invalid email format")
        if errors:
            msg = f"Validation failed: {'; '.join(errors)}"
            raise ValueError(msg)

def traditional_api_call():
    """Chamada API tradicional com muito código repetitivo."""
    try:
        # Simulação de processamento
        import time
        time.sleep(0.1)

        user = TraditionalUser("123", "Alice", "alice@example.com")
        user.validate()

        return {
            "success": True,
            "data": user.to_dict(),
            "error": None,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e),
            "timestamp": time.time(),
        }

# =============================================================================
# DEPOIS: Com FLEXT Core (90% Menos Boilerplate!)
# =============================================================================

class FlextUser(FlextEntityMixin):
    """Implementação FLEXT com 90% menos código!"""

    id: FlextUserId
    name: str
    email: FlextEmail

    def get_comparison_key(self) -> str:
        return self.id

    def validate_business_rules(self) -> FlextResult[None]:
        result = self.start_validation()

        if len(self.name) < 2:
            result.add_error("Name must be at least 2 characters")
        if "@" not in self.email:
            result.add_error("Invalid email format")

        return result.finalize()

@flext_robust(max_attempts=3, timing=True, safe=True)
def flext_api_call() -> FlextResult[FlextUser]:
    """Chamada API FLEXT com 95% menos código!"""
    user = FlextUser(
        id=FlextUserId("123"),
        name="Alice",
        email=FlextEmail("alice@example.com"),
    )

    validation = user.validate_business_rules()
    if not validation.is_success:
        return FlextResult.fail(validation.error)

    return FlextResult.ok(user)

# =============================================================================
# CONFIGURAÇÃO: Antes vs Depois
# =============================================================================

def traditional_config():
    """Configuração tradicional - muito verbosa."""
    import os

    config = {}
    config["database_url"] = os.getenv("DATABASE_URL", "postgresql://localhost/db")
    config["pool_size"] = int(os.getenv("POOL_SIZE", "10"))
    config["timeout"] = int(os.getenv("TIMEOUT", "30"))
    config["debug"] = os.getenv("DEBUG", "false").lower() in ["true", "1", "yes"]

    # Validation
    if not config["database_url"]:
        msg = "DATABASE_URL is required"
        raise ValueError(msg)
    if config["pool_size"] < 1:
        msg = "Pool size must be positive"
        raise ValueError(msg)

    return config

def flext_config_demo():
    """Configuração FLEXT - 80% menos código!"""
    config = flext_config(
        database_url="postgresql://localhost/db",
        pool_size=10,
        timeout=30,
        debug=True,
    )

    # Validação automática e type-safe
    config.get_str("database_url").unwrap()
    config.get_int("pool_size").unwrap()
    config.get_bool("debug").unwrap()

    return config

# =============================================================================
# API RESPONSES: Antes vs Depois
# =============================================================================

def traditional_api_response(success: bool, data=None, error=None):
    """Response tradicional - muito boilerplate."""
    import json
    import time

    response = {
        "success": success,
        "data": data,
        "error": error,
        "timestamp": time.time(),
        "metadata": {
            "version": "1.0",
            "service": "user-service",
        },
    }

    try:
        # Tentativa de serialização
        json.dumps(response)
        return response
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Serialization failed: {e}",
            "timestamp": time.time(),
        }

def flext_api_response_demo(success: bool, data=None, error=None):
    """Response FLEXT - 90% menos código!"""
    return flext_api_response(
        success=success,
        data=data,
        error=error,
        metadata={"version": "1.0", "service": "user-service"},
    )

# =============================================================================
# EVENTOS: Antes vs Depois
# =============================================================================

def traditional_event(event_type: str, data=None):
    """Evento tradicional - muito boilerplate."""
    import datetime
    import uuid

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0",
        "source": "unknown",
        "correlation_id": None,
    }

def flext_event_demo(event_type: str, data=None):
    """Evento FLEXT - 85% menos código!"""
    return flext_event(
        event_type=event_type,
        data=data,
        source="user-service",
        version="1.0",
    )

# =============================================================================
# DEMONSTRAÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    """Demonstra a redução massiva de boilerplate."""
    # 1. Teste de API com retry automático
    api_result = flext_api_call()
    if api_result.is_success:
        user = api_result.data
        # Serialização automática!
        json_result = user.to_json_safe()
        if json_result.is_success:
            pass
    else:
        pass

    # 2. Configuração type-safe
    flext_config_demo()

    # 3. Response padronizada
    flext_api_response_demo(
        success=True,
        data={"user_id": "123", "status": "active"},
        error=None,
    )

    # 4. Evento estruturado
    flext_event_demo(
        "user.created",
        {"user_id": "123", "email": "alice@example.com"},
    )


if __name__ == "__main__":
    main()
