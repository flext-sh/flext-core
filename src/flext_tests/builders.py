"""Test data builders for FLEXT ecosystem tests.

Provides ultra-powerful builder pattern for creating complex test data structures.
Supports r, lists, dicts, mappings, and generic classes with fluent interface.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Self, TypeGuard, overload

from pydantic import BaseModel, ValidationError

from flext_core import r
from flext_tests import c, m, t, tt, u


class FlextTestsBuilders:
    """Ultra-powerful test data builder with fluent interface.

    Example:
        from flexts import tb

        dataset = tb().add("users", count=5).add("config", production=True).build()
        result = tb().add("data", value=42).to_result()
        model = tb().add("name", "test").add("value", 100).build(as_model=MyModel)
    """

    def __init__(self, **data: t.Tests.object) -> None:
        """Initialize builder with optional initial data."""
        super().__init__()
        self._data: t.Tests.Builders.BuilderDict = dict(data)

    @staticmethod
    def _is_result_obj(value: t.Tests.object) -> TypeGuard[r[t.Tests.object]]:
        return isinstance(value, r)

    def add(
        self,
        key: str,
        value: t.Tests.Builders.BuilderValue | None = None,
        **kwargs: t.Tests.object,
    ) -> Self:
        """Add data to builder with smart type inference.

        Resolution order (first match wins):
        1. result_ok → r[T].ok()  2. result_fail → r[T].fail()
        3. cls → class instantiation  4. items → list with map/filter
        5. entries → dict with map/filter  6. factory → FlextTestsFactories
        7. model → Pydantic model  8. production/debug → config
        9. mapping → dict  10. sequence → list  11. value/default → direct
        """
        value_for_kwargs = self._normalize_input(value)
        try:
            params = m.Tests.AddParams.model_validate({
                "key": key, "value": value_for_kwargs, **kwargs,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid add() parameters: {exc}") from exc

        resolved_value = self._resolve_add_value(params)

        if params.transform is not None and resolved_value is not None:
            resolved_value = self._apply_transform(params.transform, resolved_value)
        if (
            params.validate_func is not None
            and resolved_value is not None
            and not params.validate_func(resolved_value)
        ):
            raise ValueError(
                f"Validation failed for key '{params.key}' with value: {resolved_value}"
            )
        self._store_value(params, resolved_value)
        return self

    def batch(
        self,
        key: str,
        scenarios: Sequence[tuple[str, t.Tests.object]],
        **kwargs: t.Tests.object,
    ) -> Self:
        """Build batch of test scenarios."""
        try:
            params = m.Tests.BuildersBatchParams.model_validate({
                "key": key, "scenarios": scenarios, **kwargs,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid batch() parameters: {exc}") from exc
        self._ensure_data()
        batch_data: list[t.Tests.object] = []
        for scenario_id, scenario_data in params.scenarios:
            if params.as_results:
                batch_data.append({
                    "_id": scenario_id, "_result_ok": scenario_data,
                    "_is_result_marker": True,
                })
            else:
                batch_data.append(scenario_data)
        if params.with_failures:
            for fail_id, fail_error in params.with_failures:
                batch_data.append({
                    "_id": fail_id, "_result_fail": fail_error,
                    "_is_result_marker": True,
                })
        self._data[params.key] = batch_data
        return self

    def build(
        self, **kwargs: t.Tests.object
    ) -> (
        t.Tests.Builders.BuildOutputValue
        | Sequence[t.Tests.Builders.BuildOutputValue]
        | Sequence[tuple[str, t.Tests.Builders.BuildOutputValue]]
    ):
        """Build the dataset with output type control."""
        try:
            params = m.Tests.BuildParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid build() parameters: {exc}") from exc
        self._ensure_data()
        data = self._process_batch_results(dict(self._data))
        if params.filter_none:
            data = {k: v for k, v in data.items() if v is not None}
        if params.flatten:
            flat: t.Tests.Builders.BuilderDict = {}
            for k, v in data.items():
                if not isinstance(v, r):
                    flat[k] = u.Tests.to_payload(v)
            data = dict(self._flatten_dict(flat))
        hooks: t.Tests.Builders.BuilderOutputDict = {
            str(k): v for k, v in data.items()
        }
        if params.validate_with is not None and not params.validate_with(hooks):
            raise ValueError("Validation failed during build")
        if params.assert_with is not None:
            params.assert_with(hooks)
        if params.map_result is not None:
            return params.map_result(hooks)
        if params.keys_only:
            return [*data.keys()]
        if params.values_only:
            return [*data.values()]
        if params.as_list:
            return list(data.items())
        if params.as_parametrized:
            test_id = str(data.get(params.parametrize_key, "default"))
            return [(test_id, data)]
        if params.as_model is not None:
            return params.as_model(**{
                k: u.Tests.to_payload(v) for k, v in data.items()
            })
        return data

    def copy_builder(self) -> Self:
        """Create independent copy of builder state."""
        self._ensure_data()
        new_builder = type(self)()
        new_builder._data = dict(self._data)
        return new_builder

    def execute(self) -> r[t.Tests.Builders.BuilderDict]:
        """Execute service - builds and returns as r."""
        self._ensure_data()
        return r[t.Tests.Builders.BuilderDict].ok(dict(self._data))

    def fork(self, **updates: t.Tests.object) -> Self:
        """Copy and immediately add updates."""
        new_builder = self.copy_builder()
        for key, value in updates.items():
            _ = new_builder.add(key, value=value)
        return new_builder

    @overload
    def get(self, path: str) -> t.Tests.Builders.BuilderValue | None: ...

    @overload
    def get[T](self, path: str, default: T) -> t.Tests.Builders.BuilderValue | T: ...

    @overload
    def get[T](
        self, path: str, default: T | None = None, *, as_type: type[T]
    ) -> T | None: ...

    def get[T](
        self, path: str, default: T | None = None, *, as_type: type[T] | None = None
    ) -> t.Tests.Builders.BuilderValue | T | None:
        """Get value from dot-separated path."""
        self._ensure_data()
        current: t.Tests.Builders.BuilderValue = self._data
        for part in path.split("."):
            if not isinstance(current, Mapping):
                return default
            if part not in current:
                return default
            current = current[part]
        if current is None:
            return default
        if as_type is not None:
            if isinstance(current, as_type):
                typed: T = current
                return typed
            return default
        return u.Tests.to_payload(current)

    def merge_from(
        self,
        other: FlextTestsBuilders,
        *,
        strategy: str = "deep",
        exclude_keys: frozenset[str] | None = None,
    ) -> Self:
        """Merge data from another builder using u.merge()."""
        try:
            params = m.Tests.MergeFromParams.model_validate({
                "strategy": strategy,
                "exclude_keys": list(exclude_keys) if exclude_keys else None,
            })
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid merge_from() parameters: {exc}") from exc
        self._ensure_data()
        other._ensure_data()
        other_data = dict(other._data)
        if params.exclude_keys:
            other_data = {
                k: v for k, v in other_data.items() if k not in params.exclude_keys
            }
        self_dict = {
            k: v for k, v in self._data.items() if t.Guards.is_general_value(v)
        }
        other_dict = {
            k: v for k, v in other_data.items() if t.Guards.is_general_value(v)
        }
        merge_result = u.merge(
            u.Tests.to_normalized_dict(self_dict),
            u.Tests.to_normalized_dict(other_dict),
            strategy=params.strategy,
        )
        if merge_result.is_success:
            for k, v in merge_result.value.items():
                self._data[k] = u.Tests.to_payload(v)
        return self

    def reset(self) -> Self:
        """Reset builder state."""
        self._data = {}
        return self

    def scenarios(
        self, *cases: tuple[str, Mapping[str, t.Tests.Builders.BuilderValue]]
    ) -> list[t.Tests.Builders.ParametrizedCase]:
        """Build pytest.mark.parametrize compatible scenarios."""
        return list(cases)

    def set(
        self,
        path: str,
        value: t.Tests.Builders.BuilderValue | None = None,
        *,
        create_parents: bool = True,
        **kwargs: t.Tests.object,
    ) -> Self:
        """Set value at nested path using dot notation."""
        self._ensure_data()
        final_value: t.Tests.Builders.BuilderValue
        if kwargs:
            if value is not None and isinstance(value, Mapping):
                merged: dict[str, t.Tests.object] = dict(value.items())
                merged.update(kwargs)
                final_value = merged
            else:
                final_value = dict(kwargs)
        else:
            final_value = value
        parts = path.split(".")
        if len(parts) == 1:
            self._data[path] = final_value
            return self
        current: t.Tests.Builders.BuilderDict = self._data
        for part in parts[:-1]:
            if part not in current:
                if not create_parents:
                    raise KeyError(f"Path '{part}' not found in '{path}'")
                current[part] = {}
            next_val = current[part]
            if not isinstance(next_val, dict):
                if not create_parents:
                    raise TypeError(f"Path '{part}' is not a dict in '{path}'")
                current[part] = {}
                next_val = current[part]
            if isinstance(next_val, dict):
                current = next_val
        current[parts[-1]] = final_value
        return self

    def to_result(
        self, **kwargs: t.Tests.object
    ) -> r[t.Tests.Builders.BuilderValue] | t.Tests.Builders.BuilderValue:
        """Build data wrapped in r."""
        try:
            params = m.Tests.ToResultParams.model_validate(kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid to_result() parameters: {exc}") from exc
        if params.error is not None:
            return r[t.Tests.object].fail(
                params.error, error_code=params.error_code, error_data=params.error_data
            )
        self._ensure_data()
        data: t.Tests.Builders.BuilderDict = dict(self._data)
        if params.validate_func is not None and not params.validate_func(data):
            return r[t.Tests.object].fail(
                "Validation failed",
                error_code=params.error_code, error_data=params.error_data,
            )
        if params.map_fn is not None:
            transformed = params.map_fn(data)
            if params.unwrap:
                return transformed if t.Guards.is_builder_value(transformed) else None
            return r[t.Tests.object].ok(transformed)
        if params.as_cls is not None:
            try:
                instance = params.as_cls(*(params.cls_args or ()), **data)
                if params.unwrap:
                    return instance if t.Guards.is_builder_value(instance) else None
                return r[t.Tests.object].ok(
                    instance if t.Guards.is_builder_value(instance) else None
                )
            except (TypeError, ValueError, AttributeError) as exc:
                return r[t.Tests.object].fail(
                    str(exc), error_code=params.error_code, error_data=params.error_data
                )
        if params.as_model is not None:
            try:
                model_instance = params.as_model(**data)
                return model_instance if params.unwrap else r[t.Tests.object].ok(model_instance)
            except (TypeError, ValueError, AttributeError) as exc:
                return r[t.Tests.object].fail(
                    str(exc), error_code=params.error_code, error_data=params.error_data
                )
        if params.as_list_result:
            values: list[t.Tests.object] = list(data.values())
            return values if params.unwrap else r[t.Tests.object].ok(values)
        if params.as_dict_result:
            if params.unwrap:
                return data
            return r[t.Tests.object].ok(data)
        result: r[t.Tests.Builders.BuilderValue] = r[t.Tests.Builders.BuilderValue].ok(data)
        if params.unwrap:
            if result.is_failure:
                raise ValueError(
                    params.unwrap_msg or f"Failed to unwrap result: {result.error}"
                )
            return result.value
        return result

    def with_configs(self, *, production: bool = False) -> Self:
        """Add configuration to builder."""
        return self.add("configs", value={
            "environment": "production" if production else "development",
            "debug": not production, "service_type": "api", "timeout": 30,
        })

    def with_users(self, count: int = 5) -> Self:
        """Add test users to builder."""
        return self.add("users", value=[
            {"id": f"user_{i}", "name": f"User {i}",
             "email": f"user{i}@example.com", "active": True}
            for i in range(count)
        ])

    def with_validation_fields(self, count: int = 5) -> Self:
        """Add validation test fields to builder."""
        return self.add("validation_fields", value={
            "valid_emails": [f"user{i}@example.com" for i in range(count)],
            "invalid_emails": ["invalid", "no-at-sign.com", "@missing-local.com"],
            "valid_hostnames": ["example.com", c.Platform.DEFAULT_HOST],
        })

    # -- Private helpers --

    @staticmethod
    def _apply_transform(
        transform: Callable[[t.Tests.object], t.Tests.object],
        value: t.Tests.object,
    ) -> t.Tests.object:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [transform(u.Tests.to_payload(item)) for item in value]
        return transform(u.Tests.to_payload(value))

    def _create_config(
        self, *, production: bool, debug: bool
    ) -> t.Tests.Builders.BuilderValue:
        """Create configuration data via tt.model('config')."""
        environment = (
            c.Tests.Builders.DEFAULT_ENVIRONMENT_PRODUCTION
            if production
            else c.Tests.Builders.DEFAULT_ENVIRONMENT_DEVELOPMENT
        )
        config_result = tt.model(
            "config",
            service_type=c.Tests.Factory.DEFAULT_SERVICE_TYPE,
            environment=environment, debug=debug,
            timeout=c.Tests.Factory.DEFAULT_TIMEOUT,
        )
        config = self._extract_model(config_result, m.Tests.Config)
        cb = c.Tests.Builders
        return {
            cb.KEY_SERVICE_TYPE: config.service_type,
            cb.KEY_ENVIRONMENT: config.environment,
            cb.KEY_DEBUG: config.debug,
            cb.KEY_LOG_LEVEL: config.log_level,
            cb.KEY_TIMEOUT: config.timeout,
            cb.KEY_MAX_RETRIES: config.max_retries,
            cb.KEY_DATABASE_URL: cb.DEFAULT_DATABASE_URL,
            cb.KEY_MAX_CONNECTIONS: cb.DEFAULT_MAX_CONNECTIONS,
        }

    def _ensure_data(self) -> None:
        if not self._data:
            self._data = {}

    @staticmethod
    def _extract_model[T: BaseModel](
        result: t.Tests.object, expected: type[T]
    ) -> T:
        """Extract a BaseModel from various tt.model() return shapes."""
        if isinstance(result, expected):
            return result
        if isinstance(result, r):
            if result.is_success and isinstance(result.value, expected):
                return result.value
        if isinstance(result, list) and len(result) == 1:
            if isinstance(result[0], expected):
                return result[0]
        if isinstance(result, dict) and len(result) == 1:
            val = next(iter(result.values()))
            if isinstance(val, expected):
                return val
        raise TypeError(f"Expected {expected.__name__}, got {type(result)}")

    def _flatten_dict(
        self, data: t.Tests.Builders.BuilderDict,
        parent_key: str = "", sep: str = ".",
    ) -> t.Tests.Builders.BuilderDict:
        """Flatten nested dict using dot notation keys."""
        items: list[tuple[str, t.Tests.Builders.BuilderValue]] = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def _generate_from_factory(
        self, factory: str, count: int
    ) -> t.Tests.Builders.BuilderValue:
        """Generate data using factory methods."""
        cb = c.Tests.Builders
        if factory == "users":
            return [
                {cb.KEY_ID: item.id, cb.KEY_NAME: item.name,
                 cb.KEY_EMAIL: item.email, cb.KEY_ACTIVE: item.active}
                for item in tt.batch("user", count=count)
                if isinstance(item, m.Tests.User)
            ]
        if factory == "configs":
            return self._create_config(production=False, debug=True)
        if factory == "services":
            services: list[dict[str, str]] = []
            for i in range(count):
                svc = tt.model("service", name=f"service_{i}")
                if isinstance(svc, m.Tests.Service):
                    services.append({
                        "id": svc.id, "name": svc.name,
                        "type": svc.type, "status": svc.status,
                    })
            return services
        if factory == "results":
            return [
                {"success": res.is_success,
                 "value": res.value if res.is_success else None}
                for res in tt.results(list(range(count)))
            ]
        raise ValueError(f"Unknown factory: {factory}")

    @staticmethod
    def _get_model_kind(model: type[BaseModel]) -> str:
        """Map Pydantic model class to factory kind string."""
        name = model.__name__.lower()
        for kind in ("user", "config", "service", "entity", "value"):
            if kind in name:
                return kind
        raise ValueError(f"Unknown model kind for {model.__name__}")

    @staticmethod
    def _normalize_input(value: t.Tests.Builders.BuilderValue | None) -> t.Tests.object:
        """Normalize input value for AddParams validation."""
        if value is None:
            return None
        if type(value) in {str, int, float, bool} or BaseModel in type(value).__mro__:
            return value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return list(value)
        if isinstance(value, dict):
            return dict(value)
        return str(value)

    def _process_batch_results(
        self, data: t.Tests.Builders.BuilderDict
    ) -> t.Tests.Builders.BuilderOutputDict:
        """Convert batch result markers to actual r objects."""
        processed: dict[
            str,
            t.Tests.object | r[t.Tests.object]
            | list[t.Tests.object | r[t.Tests.object]]
            | Mapping[str, t.Tests.object],
        ] = {}
        for key, value in data.items():
            if isinstance(value, list):
                processed[key] = [self._convert_marker(item) for item in value]
            elif isinstance(value, dict) and value.get("_is_result_marker"):
                processed[key] = self._convert_marker(value)
            else:
                processed[key] = value
        return processed

    @staticmethod
    def _convert_marker(item: t.Tests.object) -> t.Tests.object | r[t.Tests.object]:
        """Convert a single result marker dict to r."""
        if isinstance(item, dict) and item.get("_is_result_marker"):
            if "_result_ok" in item:
                return r[t.Tests.object].ok(item["_result_ok"])
            if "_result_fail" in item:
                return r[t.Tests.object].fail(str(item["_result_fail"]))
        return item

    def _resolve_add_value(self, params: m.Tests.AddParams) -> t.Tests.object:
        """Core resolution logic for add() — returns resolved value."""
        if params.result_ok is not None:
            return {"_result_ok": params.result_ok, "_is_result_marker": True}
        if params.result_fail is not None:
            return {
                "_result_fail": params.result_fail,
                "_result_code": params.result_code or c.Errors.VALIDATION_ERROR,
                "_is_result_marker": True,
            }
        if params.cls is not None:
            return self._resolve_cls(params)
        if params.items is not None:
            return self._resolve_items(params)
        if params.entries is not None:
            return self._resolve_entries(params)
        if params.factory is not None:
            return self._generate_from_factory(
                params.factory, params.count or c.Tests.Factory.DEFAULT_BATCH_COUNT
            )
        if params.model is not None:
            return self._resolve_model(params)
        if params.production is not None or params.debug is not None:
            return self._create_config(
                production=params.production or False,
                debug=params.debug if params.debug is not None else not (params.production or False),
            )
        if params.mapping is not None:
            return dict(params.mapping.items())
        if params.sequence is not None:
            return list(params.sequence)
        if params.value is not None:
            return params.value
        if params.default is not None:
            return params.default
        return None

    def _resolve_cls(self, params: m.Tests.AddParams) -> t.Tests.object:
        """Resolve cls= parameter in add()."""
        cls_type = params.cls
        assert cls_type is not None
        cls_kwargs = params.cls_kwargs or {}

        def is_entity(c: type) -> TypeGuard[type[m.Tests.Entity]]:
            return issubclass(c, m.Tests.Entity)

        def is_value(c: type) -> TypeGuard[type[m.Tests.Value]]:
            return issubclass(c, m.Tests.Value)

        if is_entity(cls_type):
            entity_cls = cls_type
            return u.Tests.DomainHelpers.create_test_entity_instance(
                name=str(cls_kwargs.get("name", "")),
                value=cls_kwargs.get("value", ""),
                entity_class=lambda *, name, value, **kw: entity_cls(name=name, value=value),
            )
        if is_value(cls_type):
            value_cls = cls_type
            data_val = str(cls_kwargs.get("data", ""))
            count_val = cls_kwargs.get("count", 1)
            return u.Tests.DomainHelpers.create_test_value_object_instance(
                data=data_val,
                count=int(count_val) if isinstance(count_val, int | float) else 1,
                value_class=lambda *, data, count: value_cls(data=data, count=count),
            )
        args = params.cls_args or ()
        instance = cls_type.__call__(*args, **cls_kwargs) if args or cls_kwargs else cls_type.__call__()
        return u.Tests.to_payload(instance)

    @staticmethod
    def _resolve_items(params: m.Tests.AddParams) -> list[t.Tests.object]:
        items = list(params.items) if params.items else []
        if params.items_filter is not None:
            items = [i for i in items if params.items_filter(i)]
        if params.items_map is not None:
            items = [params.items_map(i) for i in items]
        return items

    @staticmethod
    def _resolve_entries(params: m.Tests.AddParams) -> dict[str, t.Tests.object]:
        entries = dict(params.entries.items()) if params.entries else {}
        if params.entries_filter is not None:
            entries = {k: v for k, v in entries.items() if k in params.entries_filter}
        if params.entries_map is not None:
            entries = {k: params.entries_map(v) for k, v in entries.items()}
        return entries

    def _resolve_model(self, params: m.Tests.AddParams) -> t.Tests.object:
        """Resolve model= parameter in add()."""
        assert params.model is not None
        data_dict = dict(params.model_data.items()) if params.model_data else {}
        model_kind_str = self._get_model_kind(params.model)
        filtered: dict[str, t.Tests.TestResultValue] = {}
        for k, v in data_dict.items():
            if type(v) in {str, int, float, bool, type(None)}:
                filtered[k] = v
            elif isinstance(v, Sequence) and not isinstance(v, str | bytes | bytearray):
                filtered[k] = list(v)
            elif isinstance(v, dict):
                filtered[k] = v
        model_kind: Literal["user", "config", "service", "entity", "value"]
        match model_kind_str:
            case "user" | "config" | "service" | "entity" | "value":
                model_kind = model_kind_str
            case _:
                model_kind = "user"
        result = tt.model(model_kind, **filtered)
        if isinstance(result, BaseModel):
            return result
        if self._is_result_obj(result):
            if result.is_success:
                val = result.value
                if isinstance(val, BaseModel):
                    return val
                if isinstance(val, Sequence) and not isinstance(val, str | bytes):
                    return list(val)
                if isinstance(val, Mapping):
                    return dict(val.items())
            return None
        if isinstance(result, (list, dict)):
            return result
        return None

    def _store_value(self, params: m.Tests.AddParams, resolved_value: t.Tests.object) -> None:
        """Store resolved value into builder data, handling merge."""
        self._ensure_data()
        builder_data = self._data
        if params.merge and params.key in builder_data:
            existing = builder_data[params.key]
            if isinstance(existing, Mapping) and isinstance(resolved_value, Mapping):
                existing_dict: dict[str, t.Tests.object] = {
                    str(k): v for k, v in existing.items()
                    if not self._is_result_obj(v)
                }
                resolved_dict: dict[str, t.Tests.object] = {
                    str(k): u.Tests.to_payload(v) for k, v in resolved_value.items()
                    if not self._is_result_obj(u.Tests.to_payload(v))
                }
                merge_result = u.merge(
                    u.Tests.to_normalized_dict(existing_dict),
                    u.Tests.to_normalized_dict(resolved_dict),
                )
                if merge_result.is_success:
                    resolved_value = u.Tests.to_payload(merge_result.value)
            else:
                builder_data[params.key] = resolved_value
        else:
            builder_data[params.key] = resolved_value
        self._data = builder_data

    # -- Inner namespace: Tests --

    class Tests:
        """Test-specific builder helpers under tb.Tests.*."""

        class Result:
            """r building helpers - tb.Tests.Result.*."""

            @staticmethod
            def all_success[T](results: Sequence[r[T]]) -> bool:
                return all(res.is_success for res in results)

            @staticmethod
            def assert_failure(result: r[t.Tests.object]) -> str:
                return u.Tests.Result.assert_failure(result)

            @staticmethod
            def assert_success[T](result: r[T]) -> T:
                return u.Tests.Result.assert_success(result)

            @staticmethod
            def batch_fail[T](
                errors: Sequence[str], expected_type: type[T] | None = None
            ) -> list[r[T]]:
                _ = expected_type
                return tt.results(values=[], errors=list(errors))

            @staticmethod
            def batch_ok[T](values: Sequence[T]) -> list[r[T]]:
                return tt.results(values=list(values))

            @staticmethod
            def fail[T](
                error: str, code: str | None = None,
                data: m.ConfigMap | None = None,
                expected_type: type[T] | None = None,
            ) -> r[T]:
                _ = expected_type
                return r[T].fail(error, error_code=code or c.Errors.VALIDATION_ERROR, error_data=data)

            @staticmethod
            def mixed[T](successes: Sequence[T], errors: Sequence[str]) -> list[r[T]]:
                return tt.results(values=list(successes), errors=list(errors))

            @staticmethod
            def ok[T](value: T) -> r[T]:
                return r[T].ok(value)

            @staticmethod
            def partition[T](results: Sequence[r[T]]) -> tuple[list[T], list[str]]:
                return (
                    [res.value for res in results if res.is_success],
                    [str(res.error) for res in results if res.is_failure],
                )

        class Batch:
            """Batch operations - tb.Tests.Batch.*."""

            @staticmethod
            def from_dict[T](mapping: Mapping[str, T]) -> list[tuple[str, T]]:
                return list(mapping.items())

            @staticmethod
            def parametrized(
                success_values: Sequence[t.Tests.object],
                failure_errors: Sequence[str],
            ) -> list[tuple[str, Mapping[str, t.Tests.object]]]:
                """Create parametrized cases - DELEGATES to u.Tests.GenericHelpers."""
                cases = u.Tests.GenericHelpers.create_parametrized_cases(
                    success_values=list(success_values),
                    failure_errors=list(failure_errors),
                )
                return [
                    (f"case_{i}", {
                        "result_is_success": ok, "result_value": val,
                        "result_error": err, "is_success": ok,
                        "value": val, "error": err,
                    })
                    for i, (_res, ok, val, err) in enumerate(cases)
                ]

            @staticmethod
            def scenarios[T](*cases: tuple[str, T]) -> list[tuple[str, T]]:
                return list(cases)

            @staticmethod
            def test_cases(
                operation: str, descriptions: Sequence[str],
                inputs: Sequence[Mapping[str, t.Tests.object]],
                expected: Sequence[t.Tests.object],
            ) -> list[Mapping[str, t.Tests.object]]:
                """Create batch test cases - DELEGATES to u.Tests.TestCaseHelpers."""
                raw = u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
                    operation=operation, descriptions=list(descriptions),
                    input_data_list=list(inputs), expected_results=list(expected),
                )
                return [dict(case.items()) for case in raw]

        class Data:
            """Data generation helpers - tb.Tests.Data.*."""

            @staticmethod
            def flatten(
                nested: Mapping[str, t.Tests.object], separator: str = "."
            ) -> Mapping[str, t.Tests.object]:
                """Flatten nested dict."""
                def _flatten(
                    data: Mapping[str, t.Tests.object], parent: str = ""
                ) -> Mapping[str, t.Tests.object]:
                    items: list[tuple[str, t.Tests.object]] = []
                    for key, value in data.items():
                        new_key = f"{parent}{separator}{key}" if parent else key
                        if isinstance(value, Mapping):
                            value_dict: dict[str, t.Tests.object] = {
                                str(k): u.Tests.to_payload(v) for k, v in value.items()
                            }
                            items.extend(_flatten(value_dict, new_key).items())
                        else:
                            items.append((new_key, value))
                    return dict(items)
                return _flatten(dict(nested.items()))

            @staticmethod
            def id() -> str:
                return u.Tests.Factory.generate_id()

            @staticmethod
            def merged(
                *dicts: Mapping[str, t.Tests.object],
            ) -> Mapping[str, t.Tests.object]:
                """Merge dictionaries - DELEGATES to u.merge()."""
                result: MutableMapping[str, t.Tests.object] = {}
                for d in dicts:
                    mr = u.merge(
                        u.Tests.to_normalized_dict({
                            k: u.Tests.to_payload(v) for k, v in result.items()
                        }),
                        u.Tests.to_normalized_dict({
                            str(k): u.Tests.to_payload(v) for k, v in d.items()
                        }),
                    )
                    if mr.is_success:
                        result = {
                            str(k): u.Tests.to_payload(v)
                            for k, v in mr.value.items()
                        }
                return result

            @staticmethod
            def short_id(length: int = 8) -> str:
                return u.Tests.Factory.generate_short_id(length)

            @staticmethod
            def transform[T, U](items: Sequence[T], func: Callable[[T], U]) -> list[U]:
                return [func(item) for item in items]

            @staticmethod
            def typed(**kwargs: t.Tests.object) -> Mapping[str, t.Tests.object]:
                return dict(kwargs)

        class Model:
            """Model creation helpers - tb.Tests.Model.*."""

            @staticmethod
            def batch_entities[T: m.Tests.Entity](
                entity_class: type[T], names: Sequence[str],
                values: Sequence[t.Tests.object],
            ) -> list[T]:
                result: r[list[T]] = u.Tests.DomainHelpers.create_test_entities_batch(
                    names=list(names), values=list(values),
                    entity_class=lambda *, name, value, **kw: entity_class(name=name, value=value),
                )
                if result.is_success:
                    return result.value
                raise ValueError(result.error)

            @staticmethod
            def batch_users(
                count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
            ) -> list[m.Tests.User]:
                return [i for i in tt.batch("user", count=count) if isinstance(i, m.Tests.User)]

            @staticmethod
            def config(**overrides: t.Tests.object) -> m.Tests.Config:
                result = tt.model("config", **overrides)
                if isinstance(result, m.Tests.Config):
                    return result
                raise TypeError(f"Expected Config, got {type(result).__name__}")

            @staticmethod
            def entity[T: m.Tests.Entity](
                entity_class: type[T], name: str = "", value: t.Tests.object = "",
            ) -> T:
                return u.Tests.DomainHelpers.create_test_entity_instance(
                    name=name, value=value,
                    entity_class=lambda *, name, value, **kw: entity_class(name=name, value=value),
                )

            @staticmethod
            def service(**overrides: t.Tests.object) -> m.Tests.Service:
                result = tt.model("service", **overrides)
                if isinstance(result, m.Tests.Service):
                    return result
                raise TypeError(f"Expected Service, got {type(result).__name__}")

            @staticmethod
            def user(**overrides: t.Tests.object) -> m.Tests.User:
                result = tt.model("user", **overrides)
                if isinstance(result, m.Tests.User):
                    return result
                raise TypeError(f"Expected User, got {type(result).__name__}")

            @staticmethod
            def value_object[T: m.Tests.Value](
                value_class: type[T], data: str = "", count: int = 1
            ) -> T:
                return u.Tests.DomainHelpers.create_test_value_object_instance(
                    data=data, count=count,
                    value_class=lambda *, data, count: value_class(data=data, count=count),
                )

        class Operation:
            """Operation helpers - tb.Tests.Operation.*."""

            @staticmethod
            def add() -> Callable[[t.Tests.object, t.Tests.object], t.Tests.object]:
                return u.Tests.Factory.add_operation

            @staticmethod
            def error(message: str) -> Callable[[], t.Tests.object]:
                return u.Tests.Factory.create_error_operation(message)

            @staticmethod
            def execute_service(
                overrides: Mapping[str, t.Tests.object] | None = None,
            ) -> r[t.Tests.object]:
                return u.Tests.Factory.execute_user_service(overrides or {})

            @staticmethod
            def format() -> Callable[[str, int], str]:
                return u.Tests.Factory.format_operation

            @staticmethod
            def simple() -> Callable[[], t.Tests.object]:
                return u.Tests.Factory.simple_operation


tb = FlextTestsBuilders
__all__ = ["FlextTestsBuilders", "tb"]
