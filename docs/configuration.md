# Configuration

Centralised configuration is a cornerstone of the 1.0.0 release. `FlextConfig`, `FlextContainer`, and `FlextDomainService` combine to deliver a predictable runtime bootstrap for every package in the ecosystem.

---

## FlextConfig Basics

```python
from flext_core import FlextConfig

class AppConfig(FlextConfig):
    database_url: str
    debug: bool = False

    model_config = {
        "env_file": ".env",
        "env_prefix": "APP_",
    }

settings = AppConfig()
```

Features:

- `.env` support via `dotenv` (lazy-loaded when the class is instantiated).
- Automatic type validation using Pydantic Settings.
- Case-insensitive environment variable lookups by default.

---

## Injecting Configuration

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()
container.register(AppConfig.__name__, settings)

config = container.get(AppConfig.__name__).unwrap()
```

Using the container aligns with the modernization pillar that all domain services resolve configuration through a shared surface.

---

## Domain Services & Config

```python
from flext_core import FlextDomainService, FlextResult

class SendWelcomeEmail(FlextDomainService[FlextResult[None]]):
    def execute(self, user_email: str) -> FlextResult[None]:
        config = self.container.get(AppConfig.__name__).unwrap()
        if not config.debug:
            self.logger.info("sending_email", to=user_email)
        return FlextResult[None].ok(None)
```

`FlextDomainService` exposes `self.container` and `self.logger` to keep domain services aligned with container and context expectations.

---

## Configuration Sources

`FlextConfig` subclasses can combine multiple sources:

- `.env` files (default when `env_file` is set)
- Environment variables
- TOML or YAML payloads via helper methods (`FlextConfigIO` within `config.py`)
- JSON strings or dictionaries passed to factory constructors

Helper functions within `config.py` (for example `merge_settings`) simplify overriding defaults in tests.

---

## FlextProcessing Limits

`FlextProcessing` consults `FlextConfig` for two limits that govern runtime behaviour:

- `max_batch_size` &ndash; caps how many items a pipeline processes in a single batch. The default mirrors `FlextConstants.Performance.DEFAULT_BATCH_SIZE` and must be at least `1`.
- `max_handlers` &ndash; restricts how many handlers can be registered in a processing registry. The default aligns with `FlextConstants.Container.MAX_SERVICES` and also requires a minimum value of `1`.

Update these fields on your configuration subclass (or via environment variables) to tune processing throughput without editing helper code. Tests that depend on different limits can override them by instantiating a custom `FlextConfig` and setting it as the global instance.

---

## Modernization Checklist

1. Downstream packages load settings through `FlextConfig` subclasses – no bespoke loaders.
2. Register configuration instances with `FlextContainer` immediately after instantiation.
3. Domain services and handlers retrieve configuration from the container instead of accessing environment variables directly.
4. Document new configuration fields in package-specific READMEs to retain alignment across the ecosystem.

Following this checklist keeps configuration workflows consistent throughout the 1.0.0 migration.
