"""Test data builders for FLEXT ecosystem tests.

Provides ultra-powerful builder pattern for creating complex test data structures.
Supports r, lists, dicts, mappings, and generic classes with fluent interface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, Self, overload

from pydantic import BaseModel

from flext_core import r
from flext_tests import c, m, t, tt, u

_SCALAR = (str, int, float, bool)
_KINDS: tuple[str, ...] = ("user", "config", "service", "entity", "value")


class FlextTestsBuilders:
    """Ultra-powerful test data builder with fluent interface.

    Example:
        from flexts import tb

        dataset = tb().add("users", count=5).add("config", production=True).build()
        result = tb().add("data", value=42).to_result()
        model = tb().add("name", "test").add("value", 100).build(as_model=MyModel)

    """

    _EXC = (TypeError, ValueError, AttributeError)

    def __init__(self, **data: t.Tests.object) -> None:
        """Initialize builder with optional initial data."""
        super().__init__()
        self._data: t.Tests.Builders.BuilderDict = dict(data)

    @classmethod
    def _v[M: BaseModel](
        cls, model: type[M], name: str, kw: dict[str, t.Tests.object]
    ) -> M:
        try:
            return model.model_validate(kw)
        except cls._EXC as e:
            raise ValueError(f"Invalid {name}() parameters: {e}") from e

    # -- Public API --

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
        nv: t.Tests.object = value
        if value is not None:
            if isinstance(value, (*_SCALAR, BaseModel, list)):
                nv = value
            elif isinstance(value, tuple):
                converted: list[t.Tests.object] = [v for v in value]  # noqa: C416
                nv = converted
            elif isinstance(value, dict):
                nv = dict(value)
            else:
                nv = str(value)
        params = self._v(m.Tests.AddParams, "add", {"key": key, "value": nv, **kwargs})
        rv = self._resolve(params)
        if params.transform is not None and rv is not None:
            rv = (
                [params.transform(u.Tests.to_payload(i)) for i in rv]
                if isinstance(rv, (list, tuple)) and not isinstance(rv, (str, bytes))
                else params.transform(u.Tests.to_payload(rv))
            )
        if (
            params.validate_func is not None
            and rv is not None
            and not params.validate_func(rv)
        ):
            raise ValueError(
                f"Validation failed for key '{params.key}' with value: {rv}"
            )
        self._store(params, rv)
        return self

    def batch(
        self,
        key: str,
        scenarios: Sequence[tuple[str, t.Tests.object]],
        **kwargs: t.Tests.object,
    ) -> Self:
        """Build batch of test scenarios."""
        params = self._v(
            m.Tests.BuildersBatchParams,
            "batch",
            {"key": key, "scenarios": scenarios, **kwargs},
        )
        self._data = self._data or {}
        bd: list[t.Tests.object] = []
        for sid, sd in params.scenarios:
            bd.append(
                {"_id": sid, "_result_ok": sd, "_is_result_marker": True}
                if params.as_results
                else sd
            )
        if params.with_failures:
            for fid, fe in params.with_failures:
                bd.append({"_id": fid, "_result_fail": fe, "_is_result_marker": True})
        self._data[params.key] = bd
        return self

    def build(
        self,
        **kwargs: t.Tests.object,
    ) -> (
        t.Tests.Builders.BuildOutputValue
        | Sequence[t.Tests.Builders.BuildOutputValue]
        | Sequence[tuple[str, t.Tests.Builders.BuildOutputValue]]
    ):
        """Build the dataset with output type control."""
        params = self._v(m.Tests.BuildParams, "build", kwargs)
        self._data = self._data or {}
        data = self._to_results(dict(self._data))
        if params.filter_none:
            data = {k: v for k, v in data.items() if v is not None}
        if params.flatten:
            data = dict(
                self._flat({
                    k: u.Tests.to_payload(v)
                    for k, v in data.items()
                    if not isinstance(v, r)
                })
            )
        hooks: t.Tests.Builders.BuilderOutputDict = {str(k): v for k, v in data.items()}
        if params.validate_with is not None and not params.validate_with(hooks):
            msg = "Validation failed during build"
            raise ValueError(msg)
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
            return [(str(data.get(params.parametrize_key, "default")), data)]
        if params.as_model is not None:
            return params.as_model(**{
                k: u.Tests.to_payload(v) for k, v in data.items()
            })
        return data

    def copy_builder(self) -> Self:
        """Create independent copy of builder state."""
        self._data = self._data or {}
        new = type(self)()
        new._data = dict(self._data)
        return new

    def execute(self) -> r[t.Tests.Builders.BuilderDict]:
        """Execute service - builds and returns as r."""
        self._data = self._data or {}
        return r[t.Tests.Builders.BuilderDict].ok(dict(self._data))

    def fork(self, **updates: t.Tests.object) -> Self:
        """Copy and immediately add updates."""
        new = self.copy_builder()
        for k, v in updates.items():
            _ = new.add(k, value=v)
        return new

    @overload
    def get(self, path: str) -> t.Tests.Builders.BuilderValue | None: ...
    @overload
    def get[T](self, path: str, default: T) -> t.Tests.Builders.BuilderValue | T: ...
    @overload
    def get[T](
        self, path: str, default: T | None = None, *, as_type: type[T]
    ) -> T | None: ...

    def get[T](
        self,
        path: str,
        default: T | None = None,
        *,
        as_type: type[T] | None = None,
    ) -> t.Tests.Builders.BuilderValue | T | None:
        """Get value from dot-separated path."""
        self._data = self._data or {}
        current: t.Tests.Builders.BuilderValue = self._data
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        if current is None:
            return default
        if as_type is not None:
            return current if isinstance(current, as_type) else default
        return u.Tests.to_payload(current)

    def merge_from(
        self,
        other: FlextTestsBuilders,
        *,
        strategy: str = "deep",
        exclude_keys: frozenset[str] | None = None,
    ) -> Self:
        """Merge data from another builder using u.merge()."""
        params = self._v(
            m.Tests.MergeFromParams,
            "merge_from",
            {
                "strategy": strategy,
                "exclude_keys": list(exclude_keys) if exclude_keys else None,
            },
        )
        self._data, other._data = self._data or {}, other._data or {}
        od = {
            k: v for k, v in other._data.items() if k not in (params.exclude_keys or [])
        }
        merged = u.Tests.merge_test_dicts(
            {k: v for k, v in self._data.items() if t.Guards.is_general_value(v)},
            {k: v for k, v in od.items() if t.Guards.is_general_value(v)},
            strategy=params.strategy,
        )
        for k, v in merged.items():
            self._data[k] = v
        return self

    def reset(self) -> Self:
        """Reset builder state."""
        self._data = {}
        return self

    def scenarios(
        self,
        *cases: tuple[str, Mapping[str, t.Tests.Builders.BuilderValue]],
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
        self._data = self._data or {}
        fv: t.Tests.Builders.BuilderValue
        if kwargs:
            fv = (
                {**value, **kwargs}
                if value is not None and isinstance(value, Mapping)
                else dict(kwargs)
            )
        else:
            fv = value
        parts = path.split(".")
        if len(parts) == 1:
            self._data[path] = fv
            return self
        current: t.Tests.Builders.BuilderDict = self._data
        for part in parts[:-1]:
            if part not in current:
                if not create_parents:
                    raise KeyError(f"Path '{part}' not found in '{path}'")
                current[part] = {}
            nv = current[part]
            if not isinstance(nv, dict):
                if not create_parents:
                    raise TypeError(f"Path '{part}' is not a dict in '{path}'")
                current[part] = {}
                nv = current[part]
            if isinstance(nv, dict):
                current = nv
        current[parts[-1]] = fv
        return self

    def to_result(
        self,
        **kwargs: t.Tests.object,
    ) -> r[t.Tests.Builders.BuilderValue] | t.Tests.Builders.BuilderValue:
        """Build data wrapped in r."""
        params = self._v(m.Tests.ToResultParams, "to_result", kwargs)
        if params.error is not None:
            return r[t.Tests.object].fail(
                params.error,
                error_code=params.error_code,
                error_data=params.error_data,
            )
        self._data = self._data or {}
        data: t.Tests.Builders.BuilderDict = dict(self._data)

        def fail(msg: str) -> r[t.Tests.object]:
            return r[t.Tests.object].fail(
                msg, error_code=params.error_code, error_data=params.error_data
            )

        if params.validate_func is not None and not params.validate_func(data):
            return fail("Validation failed")
        if params.map_fn is not None:
            tr = params.map_fn(data)
            return (
                u.Tests.coerce_to_test_object(tr)
                if params.unwrap
                else r[t.Tests.object].ok(tr)
            )
        if params.as_cls is not None:
            try:
                bv: t.Tests.object = u.Tests.coerce_to_test_object(
                    params.as_cls(*(params.cls_args or ()), **data)
                )
                return bv if params.unwrap else r[t.Tests.object].ok(bv)
            except self._EXC as e:
                return fail(str(e))
        if params.as_model is not None:
            try:
                mi = params.as_model(**data)
                return mi if params.unwrap else r[t.Tests.object].ok(mi)
            except self._EXC as e:
                return fail(str(e))
        if params.as_list_result:
            vals: list[t.Tests.object] = list(data.values())
            return vals if params.unwrap else r[t.Tests.object].ok(vals)
        if params.as_dict_result:
            return data if params.unwrap else r[t.Tests.object].ok(data)
        result = r[t.Tests.Builders.BuilderValue].ok(data)
        if params.unwrap:
            if result.is_failure:
                raise ValueError(
                    params.unwrap_msg or f"Failed to unwrap result: {result.error}"
                )
            return result.value
        return result

    def with_configs(self, *, production: bool = False) -> Self:
        """Add configuration to builder."""
        return self.add(
            "configs",
            value={
                "environment": "production" if production else "development",
                "debug": not production,
                "service_type": "api",
                "timeout": 30,
            },
        )

    def with_users(self, count: int = 5) -> Self:
        """Add test users to builder."""
        return self.add(
            "users",
            value=[
                {
                    "id": f"user_{i}",
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "active": True,
                }
                for i in range(count)
            ],
        )

    def with_validation_fields(self, count: int = 5) -> Self:
        """Add validation test fields to builder."""
        return self.add(
            "validation_fields",
            value={
                "valid_emails": [f"user{i}@example.com" for i in range(count)],
                "invalid_emails": list(c.Tests.Builders.INVALID_EMAIL_SAMPLES),
                "valid_hostnames": list(c.Tests.Builders.VALID_HOSTNAME_SAMPLES),
            },
        )

    # -- Private helpers --

    def _config(
        self, *, production: bool, debug: bool
    ) -> t.Tests.Builders.BuilderValue:
        """Create configuration data via tt.model('config')."""
        env = (
            c.Tests.Builders.DEFAULT_ENVIRONMENT_PRODUCTION
            if production
            else c.Tests.Builders.DEFAULT_ENVIRONMENT_DEVELOPMENT
        )
        config = u.Tests.extract_model(
            tt.model(
                "config",
                service_type=c.Tests.Factory.DEFAULT_SERVICE_TYPE,
                environment=env,
                debug=debug,
                timeout=c.Tests.Factory.DEFAULT_TIMEOUT,
            ),
            m.Tests.Config,
        )
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

    @staticmethod
    def _flat(
        data: t.Tests.Builders.BuilderDict, parent_key: str = "", sep: str = "."
    ) -> t.Tests.Builders.BuilderDict:
        """Flatten nested dict using dot notation keys."""
        items: list[tuple[str, t.Tests.Builders.BuilderValue]] = []
        for k, v in data.items():
            nk = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(
                FlextTestsBuilders._flat(v, nk, sep).items()
                if isinstance(v, dict)
                else [(nk, v)]
            )
        return dict(items)

    def _factory(self, factory: str, count: int) -> t.Tests.Builders.BuilderValue:
        """Generate data using factory methods."""
        cb = c.Tests.Builders
        if factory == "users":
            return [
                {
                    cb.KEY_ID: i.id,
                    cb.KEY_NAME: i.name,
                    cb.KEY_EMAIL: i.email,
                    cb.KEY_ACTIVE: i.active,
                }
                for i in tt.batch("user", count=count)
                if isinstance(i, m.Tests.User)
            ]
        if factory == "configs":
            return self._config(production=False, debug=True)
        if factory == "services":
            svcs: list[dict[str, str]] = []
            for i in range(count):
                svc = tt.model("service", name=f"service_{i}")
                if isinstance(svc, m.Tests.Service):
                    svcs.append({
                        "id": svc.id,
                        "name": svc.name,
                        "type": svc.type,
                        "status": svc.status,
                    })
            return svcs
        if factory == "results":
            return [
                {
                    "success": rs.is_success,
                    "value": rs.value if rs.is_success else None,
                }
                for rs in tt.results(list(range(count)))
            ]
        raise ValueError(f"Unknown factory: {factory}")

    def _model(self, params: m.Tests.AddParams) -> t.Tests.object:
        """Resolve model= parameter in add()."""
        assert params.model is not None
        dd: dict[str, t.Tests.object] = (
            dict(params.model_data.items()) if params.model_data else {}
        )
        name = params.model.__name__.lower()
        mk_str = next((k for k in _KINDS if k in name), None)
        if mk_str is None:
            raise ValueError(f"Unknown model kind for {params.model.__name__}")
        filtered: dict[str, t.Tests.TestResultValue] = {}
        for k, v in dd.items():
            if isinstance(v, (*_SCALAR, type(None), list)):
                filtered[k] = v
            elif isinstance(v, tuple):
                filtered[k] = [i for i in v]  # noqa: C416
            elif isinstance(v, dict):
                filtered[k] = v
        mk: Literal["user", "config", "service", "entity", "value"]
        match mk_str:
            case "user" | "config" | "service" | "entity" | "value":
                mk = mk_str
            case _:
                mk = "user"
        result = tt.model(mk, **filtered)
        if isinstance(result, BaseModel):
            return result
        if u.Tests.is_flext_result(result):
            if result.is_success:
                rval: BaseModel = result.value
                if isinstance(rval, BaseModel):
                    return rval
            return None
        return result if isinstance(result, (list, dict)) else None

    def _resolve(self, params: m.Tests.AddParams) -> t.Tests.object:
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
            items = list(params.items) if params.items else []
            if params.items_filter is not None:
                items = [i for i in items if params.items_filter(i)]
            if params.items_map is not None:
                items = [params.items_map(i) for i in items]
            return items
        if params.entries is not None:
            entries: dict[str, t.Tests.object] = (
                dict(params.entries.items()) if params.entries else {}
            )
            if params.entries_filter is not None:
                entries = {
                    k: v for k, v in entries.items() if k in params.entries_filter
                }
            if params.entries_map is not None:
                entries = {k: params.entries_map(v) for k, v in entries.items()}
            return entries
        if params.factory is not None:
            return self._factory(
                params.factory, params.count or c.Tests.Factory.DEFAULT_BATCH_COUNT
            )
        if params.model is not None:
            return self._model(params)
        if params.production is not None or params.debug is not None:
            return self._config(
                production=params.production or False,
                debug=params.debug
                if params.debug is not None
                else not (params.production or False),
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
        """Resolve cls= parameter in add() — uses DRY factory helpers."""
        cls_type = params.cls
        assert cls_type is not None
        kw = params.cls_kwargs or {}
        if issubclass(cls_type, m.Tests.Entity):
            return u.Tests.DomainHelpers.create_test_entity_instance(
                name=str(kw.get("name", "")),
                value=kw.get("value", ""),
                entity_class=u.Tests.entity_factory_for(cls_type),
            )
        if issubclass(cls_type, m.Tests.Value):
            cv = kw.get("count", 1)
            return u.Tests.DomainHelpers.create_test_value_object_instance(
                data=str(kw.get("data", "")),
                count=int(cv) if isinstance(cv, (int, float)) else 1,
                value_class=u.Tests.value_factory_for(cls_type),
            )
        args = params.cls_args or ()
        return u.Tests.to_payload(
            cls_type.__call__(*args, **kw) if args or kw else cls_type.__call__()
        )

    def _store(self, params: m.Tests.AddParams, resolved_value: t.Tests.object) -> None:
        """Store resolved value into builder data, handling merge."""
        self._data = self._data or {}
        if params.merge and params.key in self._data:
            ex, rv = self._data[params.key], resolved_value
            if isinstance(ex, Mapping) and isinstance(rv, Mapping):
                merged = u.Tests.merge_test_dicts(
                    {str(k): v for k, v in ex.items() if not isinstance(v, r)},
                    {
                        str(k): u.Tests.to_payload(v)
                        for k, v in rv.items()
                        if not isinstance(u.Tests.to_payload(v), r)
                    },
                )
                resolved_value = u.Tests.to_payload(merged)
        self._data[params.key] = resolved_value

    def _to_results(
        self, data: t.Tests.Builders.BuilderDict
    ) -> t.Tests.Builders.BuilderOutputDict:
        """Convert batch result markers to actual r objects."""

        def cm(item: t.Tests.object) -> t.Tests.object | r[t.Tests.object]:
            if isinstance(item, dict) and item.get("_is_result_marker"):
                if "_result_ok" in item:
                    return r[t.Tests.object].ok(item["_result_ok"])
                if "_result_fail" in item:
                    return r[t.Tests.object].fail(str(item["_result_fail"]))
            return item

        processed: dict[
            str,
            t.Tests.object
            | r[t.Tests.object]
            | list[t.Tests.object | r[t.Tests.object]]
            | Mapping[str, t.Tests.object],
        ] = {}
        for k, v in data.items():
            if isinstance(v, list):
                processed[k] = [cm(i) for i in v]
            elif isinstance(v, dict) and v.get("_is_result_marker"):
                processed[k] = cm(v)
            else:
                processed[k] = v
        return processed

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
                error: str,
                code: str | None = None,
                data: m.ConfigMap | None = None,
                expected_type: type[T] | None = None,
            ) -> r[T]:
                _ = expected_type
                return r[T].fail(
                    error,
                    error_code=code or c.Errors.VALIDATION_ERROR,
                    error_data=data,
                )

            @staticmethod
            def mixed[T](successes: Sequence[T], errors: Sequence[str]) -> list[r[T]]:
                return tt.results(values=list(successes), errors=list(errors))

            @staticmethod
            def ok[T](value: T) -> r[T]:
                return r[T].ok(value)

            @staticmethod
            def partition[T](
                results: Sequence[r[T]],
            ) -> tuple[list[T], list[str]]:
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
                return [
                    (
                        f"case_{i}",
                        {
                            "result_is_success": ok,
                            "result_value": val,
                            "result_error": err,
                            "is_success": ok,
                            "value": val,
                            "error": err,
                        },
                    )
                    for i, (_res, ok, val, err) in enumerate(
                        u.Tests.GenericHelpers.create_parametrized_cases(
                            success_values=list(success_values),
                            failure_errors=list(failure_errors),
                        )
                    )
                ]

            @staticmethod
            def scenarios[T](*cases: tuple[str, T]) -> list[tuple[str, T]]:
                return list(cases)

            @staticmethod
            def test_cases(
                operation: str,
                descriptions: Sequence[str],
                inputs: Sequence[Mapping[str, t.Tests.object]],
                expected: Sequence[t.Tests.object],
            ) -> list[Mapping[str, t.Tests.object]]:
                """Create batch test cases - DELEGATES to u.Tests.TestCaseHelpers."""
                return [
                    dict(c.items())
                    for c in u.Tests.TestCaseHelpers.create_batch_operation_test_cases(
                        operation=operation,
                        descriptions=list(descriptions),
                        input_data_list=list(inputs),
                        expected_results=list(expected),
                    )
                ]

        class Data:
            """Data generation helpers - tb.Tests.Data.*."""

            @staticmethod
            def flatten(
                nested: Mapping[str, t.Tests.object], separator: str = "."
            ) -> Mapping[str, t.Tests.object]:
                """Flatten nested dict — delegates to FlextTestsBuilders._flat."""
                return FlextTestsBuilders._flat(
                    {str(k): v for k, v in nested.items()}, sep=separator
                )

            @staticmethod
            def id() -> str:
                return u.generate()

            @staticmethod
            def merged(
                *dicts: Mapping[str, t.Tests.object],
            ) -> Mapping[str, t.Tests.object]:
                """Merge dictionaries - DELEGATES to u.Tests.merge_test_dicts."""
                result: dict[str, t.Tests.object] = {}
                for d in dicts:
                    result = u.Tests.merge_test_dicts(result, dict(d.items()))
                return result

            @staticmethod
            def short_id(length: int = 8) -> str:
                return u.generate("ulid", length=length)

            @staticmethod
            def transform[T, U](items: Sequence[T], func: Callable[[T], U]) -> list[U]:
                return [func(i) for i in items]

            @staticmethod
            def typed(**kwargs: t.Tests.object) -> Mapping[str, t.Tests.object]:
                return dict(kwargs)

        class Model:
            """Model creation helpers - tb.Tests.Model.*."""

            @staticmethod
            def batch_entities[T: m.Tests.Entity](
                entity_class: type[T],
                names: Sequence[str],
                values: Sequence[t.Tests.object],
            ) -> list[T]:
                return u.Tests.Result.assert_success(
                    u.Tests.DomainHelpers.create_test_entities_batch(
                        names=list(names),
                        values=list(values),
                        entity_class=u.Tests.entity_factory_for(entity_class),
                    )
                )

            @staticmethod
            def batch_users(
                count: int = c.Tests.Factory.DEFAULT_BATCH_COUNT,
            ) -> list[m.Tests.User]:
                return [
                    i
                    for i in tt.batch("user", count=count)
                    if isinstance(i, m.Tests.User)
                ]

            @staticmethod
            def config(**overrides: t.Tests.object) -> m.Tests.Config:
                return u.Tests.extract_model(
                    tt.model("config", **overrides), m.Tests.Config
                )

            @staticmethod
            def entity[T: m.Tests.Entity](
                entity_class: type[T],
                name: str = "",
                value: t.Tests.object = "",
            ) -> T:
                return u.Tests.DomainHelpers.create_test_entity_instance(
                    name=name,
                    value=value,
                    entity_class=u.Tests.entity_factory_for(entity_class),
                )

            @staticmethod
            def service(**overrides: t.Tests.object) -> m.Tests.Service:
                return u.Tests.extract_model(
                    tt.model("service", **overrides), m.Tests.Service
                )

            @staticmethod
            def user(**overrides: t.Tests.object) -> m.Tests.User:
                return u.Tests.extract_model(
                    tt.model("user", **overrides), m.Tests.User
                )

            @staticmethod
            def value_object[T: m.Tests.Value](
                value_class: type[T], data: str = "", count: int = 1
            ) -> T:
                return u.Tests.DomainHelpers.create_test_value_object_instance(
                    data=data,
                    count=count,
                    value_class=u.Tests.value_factory_for(value_class),
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
