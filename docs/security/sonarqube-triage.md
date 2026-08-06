# Triagem SonarCloud — flext-sh/flext-core

Gerado do dump da plataforma SonarCloud (2026-08-06).

Bead: `mro-2wjm.2`

## Resumo

**133 issues** — BLOCKER 5, CRITICAL 24, MAJOR 27, MINOR 77
Tipos: VULNERABILITY 4, BUG 6, CODE_SMELL 123 · **Debt total: 630min**

| regra | issues |
|---|---|
| `python:S116` | 66 |
| `python:S3776` | 15 |
| `python:S1192` | 8 |
| `python:S108` | 6 |
| `python:S930` | 5 |
| `python:S5713` | 4 |
| `python:S3358` | 3 |
| `python:S6796` | 3 |
| `python:S8963` | 3 |
| `githubactions:S8233` | 2 |

## Como usar

Cada issue traz a **mensagem do SonarQube** (descreve o problema e o impacto), o **código real** (linha `>>>`), o tipo e o effort estimado.
**Decisão**: `corrigir` / `falso-positivo` (marcar na plataforma com justificativa) / `risco-aceito`. Ordem: BLOCKER → CRITICAL → VULNERABILITY → MAJOR. CODE_SMELL em volume pede correção de padrão.

## Issues

### 1 · 🔴 BLOCKER · BUG · `python:S930`
**Local**: `src/flext_core/__version__.py:35` · **Effort**: 10min

> Remove this unexpected named argument 'strict'.

```python
       31      @staticmethod
       32      def _resolve_author(package_metadata: PackageMetadata) -> tuple[str, str]:
       33          """Return the first normalized author identity from package metadata."""
       34          raw_email = package_metadata.get("Author-Email", "")
>>>    35          author_name, author_email = parseaddr(raw_email, strict=True)
       36          if raw_email and not author_email:
       37              msg = f"invalid Author-Email package metadata: {raw_email!r}"
       38              raise ValueError(msg)
       39          return (
```

**Decisão**: 

### 2 · 🔴 BLOCKER · BUG · `python:S930`
**Local**: `src/flext_core/_result/base.py:87` · **Effort**: 10min

> Remove this unexpected named argument 'error'.

```python
       83          exception: BaseException | None = None,
       84      ) -> None:
       85          type(self).reject_banned_result_parameterization()
       86          super().__init__(
>>>    87              error=error,
       88              error_code=error_code,
       89              success=success,
       90              error_data=self.validate_error_data(error_data),
       91          )
```

**Decisão**: 

### 3 · 🔴 BLOCKER · BUG · `python:S930`
**Local**: `src/flext_core/_result/base.py:88` · **Effort**: 10min

> Remove this unexpected named argument 'error_code'.

```python
       84      ) -> None:
       85          type(self).reject_banned_result_parameterization()
       86          super().__init__(
       87              error=error,
>>>    88              error_code=error_code,
       89              success=success,
       90              error_data=self.validate_error_data(error_data),
       91          )
       92          if success:
```

**Decisão**: 

### 4 · 🔴 BLOCKER · BUG · `python:S930`
**Local**: `src/flext_core/_result/base.py:89` · **Effort**: 10min

> Remove this unexpected named argument 'success'.

```python
       85          type(self).reject_banned_result_parameterization()
       86          super().__init__(
       87              error=error,
       88              error_code=error_code,
>>>    89              success=success,
       90              error_data=self.validate_error_data(error_data),
       91          )
       92          if success:
       93              self.reject_banned_success_payload(value)
```

**Decisão**: 

### 5 · 🔴 BLOCKER · BUG · `python:S930`
**Local**: `src/flext_core/_result/base.py:90` · **Effort**: 10min

> Remove this unexpected named argument 'error_data'.

```python
       86          super().__init__(
       87              error=error,
       88              error_code=error_code,
       89              success=success,
>>>    90              error_data=self.validate_error_data(error_data),
       91          )
       92          if success:
       93              self.reject_banned_success_payload(value)
       94              self._payload = cast("T", value)
```

**Decisão**: 

### 6 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_01.py:24` · **Effort**: 12min

> Refactor this function to reduce its Cognitive Complexity from 22 to the 15 allowed.

```python
       20  
       21  
       22  class FlextBaseErrorMetadataMixin:
       23      @staticmethod
>>>    24      def normalize_metadata(
       25          metadata: pr.HasModelDump | tb.JsonValue | None,
       26          merged_kwargs: tb.MappingKV[str, ts.JsonPayload],
       27      ) -> m.Metadata:
       28          """Normalize metadata from various input types to m.Metadata model."""
```

**Decisão**: 

### 7 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_03.py:33` · **Effort**: 33min

> Refactor this function to reduce its Cognitive Complexity from 43 to the 15 allowed.

```python
       29  
       30      params_cls: ClassVar[ts.ModelClass[mp.BaseModel] | None] = None
       31      excluded_context_keys: ClassVar[set[str] | frozenset[str] | None] = None
       32  
>>>    33      def __init__(
       34          self,
       35          message: str,
       36          *,
       37          error_code: str = cv.ErrorCode.UNKNOWN_ERROR,
```

**Decisão**: 

### 8 · 🟠 CRITICAL · CODE_SMELL · `python:S1186`
**Local**: `src/flext_core/_result/behavior.py:32` · **Effort**: 5min

> Add a nested comment explaining why this method is empty, or complete the implementation.

```python
       28  
       29      def __enter__(self) -> Self:
       30          return self
       31  
>>>    32      def __exit__(
       33          self,
       34          _exc_type: type[BaseException] | None,
       35          _exc_val: BaseException | None,
       36          _exc_tb: object,
```

**Decisão**: 

### 9 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_result/composition.py:64` · **Effort**: 10min

> Define a constant instead of duplicating this literal "p.Result[U]" 5 times.

```python
       60              try:
       61                  all_results.append(cls.from_result(func(item)))
       62              except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
       63                  all_results.append(
>>>    64                      cast("p.Result[U]", cls.fail(str(exc), exception=exc))
       65                  )
       66          return cls.accumulate_errors(*all_results)
       67  
       68      @classmethod
```

**Decisão**: 

### 10 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_result/construction.py:119` · **Effort**: 10min

> Define a constant instead of duplicating this literal "p.Result[V]" 5 times.

```python
      115          try:
      116              value = func()
      117              if value is None:
      118                  return cast(
>>>   119                      "p.Result[V]",
      120                      cls.fail("Callable returned None", error_code=error_code),
      121                  )
      122              return cls.ok(value)
      123          except c.EXC_BROAD_RUNTIME as exc:
```

**Decisão**: 

### 11 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_result/transforms.py:39` · **Effort**: 12min

> Define a constant instead of duplicating this literal "p.Result[U]" 6 times.

```python
       35  
       36      def flat_map[U](self, func: Callable[[T], p.Result[U]]) -> p.Result[U]:
       37          if self.failure:
       38              return cast(
>>>    39                  "p.Result[U]",
       40                  self.__class__.fail(
       41                      self.require_error(self._as_result()),
       42                      error_code=self.error_code,
       43                      error_data=self.error_data,
```

**Decisão**: 

### 12 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_result/transforms.py:76` · **Effort**: 10min

> Define a constant instead of duplicating this literal "p.Result[T | U]" 5 times.

```python
       72      def lash[U](self, func: Callable[[str], p.Result[U]]) -> p.Result[T | U]:
       73          if self.failure:
       74              try:
       75                  return cast(
>>>    76                      "p.Result[T | U]",
       77                      self.__class__.from_result(
       78                          func(self.require_error(self._as_result()))
       79                      ),
       80                  )
```

**Decisão**: 

### 13 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_settings.py:95` · **Effort**: 6min

> Define a constant instead of duplicating this literal "Application Support" 3 times.

```python
       91      ``~/Library/Application Support``; Windows uses ``%LOCALAPPDATA%`` (default
       92      ``~/AppData/Local``). Module-level + stdlib-only for layer-0 purity.
       93      """
       94      if sys.platform == "darwin":
>>>    95          return Path.home() / "Library" / "Application Support"
       96      if sys.platform == "win32":
       97          local_app_data = os.environ.get("LOCALAPPDATA")
       98          return (
       99              Path(local_app_data)
```

**Decisão**: 

### 14 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_beartype/attr_visitor.py:106` · **Effort**: 6min

> Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed.

```python
      102              return False
      103          return FlextUtilitiesBeartypeAttrVisitor._is_constant_value(value)
      104  
      105      @staticmethod
>>>   106      def v_classvar_constant(
      107          params: me.ClassVarConstantParams, target: type
      108      ) -> t.StrMapping | None:
      109          """CLASSVAR_CONSTANT — flag constants declared outside _constants."""
      110          module_name = getattr(target, "__module__", "") or ""
```

**Decisão**: 

### 15 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_beartype/deprecated_visitor.py:39` · **Effort**: 18min

> Refactor this function to reduce its Cognitive Complexity from 28 to the 15 allowed.

```python
       35                  return {"name": fn.__name__, "file": Path(src_file).name}
       36          return _NO_VIOLATION
       37  
       38      @staticmethod
>>>    39      def v_deprecated_syntax(
       40          params: me.DeprecatedSyntaxParams, target: type
       41      ) -> t.StrMapping | None:
       42          """DEPRECATED_SYNTAX — runtime introspection routed by ``params.ast_shape``."""
       43          shape = params.ast_shape
```

**Decisão**: 

### 16 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_beartype/field_visitor.py:45` · **Effort**: 8min

> Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed.

```python
       41          ))
       42          return None if has_description else {}
       43  
       44      @staticmethod
>>>    45      def _field_violation(
       46          params: me.FieldShapeParams, info: FieldInfo
       47      ) -> t.StrMapping | None:
       48          violation: t.StrMapping | None = None
       49          if params.forbid_any and _ubh.contains_any_recursive(
```

**Decisão**: 

### 17 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_beartype/import_visitor.py:61` · **Effort**: 8min

> Refactor this function to reduce its Cognitive Complexity from 18 to the 15 allowed.

```python
       57  class _ImportBlacklistVisitor:
       58      """IMPORT_BLACKLIST implementation extracted for LOC cap."""
       59  
       60      @staticmethod
>>>    61      def v_import_blacklist(
       62          params: me.ImportBlacklistParams, target: type
       63      ) -> t.StrMapping | None:
       64          """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline."""
       65          no_violation: t.StrMapping | None = None
```

**Decisão**: 

### 18 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_beartype/module_visitor.py:52` · **Effort**: 6min

> Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed.

```python
       48  class FlextUtilitiesBeartypeModuleVisitor:
       49      """LOC_CAP + MODULE_ALIAS + DUPLICATE_SYMBOL visitors."""
       50  
       51      @staticmethod
>>>    52      def v_loc_cap(params: me.LocCapParams, target: type) -> t.StrMapping | None:
       53          """LOC_CAP — module logical-LOC ceiling + top-level class census (§3.1)."""
       54          module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
       55          if module is None:
       56              return _NO_VIOLATION
```

**Decisão**: 

### 19 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_checker_parts/checker_part_03.py:112` · **Effort**: 7min

> Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed.

```python
      108                  message_types.append(explicit_type_result.unwrap())
      109          return tuple(message_types)
      110  
      111      @classmethod
>>>   112      def resolve_message_route(cls, msg: pb.Routable | type[pb.Routable] | str) -> str:
      113          """Resolve route name from Routable attributes or string.
      114  
      115          Raises:
      116              TypeError: If message does not provide a valid route.
```

**Decisão**: 

### 20 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_enforcement_collect_parts/enforcement_collect_part_01.py:98` · **Effort**: 6min

> Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed.

```python
       94              return None
       95          return upm.build_project_metadata(project_root, document).package_name
       96  
       97      @staticmethod
>>>    98      def _project(target: type) -> t.StrPair | None:
       99          """Return (derived_prefix, inner_namespace) or None if unknowable."""
      100          top = (getattr(target, "__module__", "") or "").split(".", 1)[0]
      101          if not top:
      102              return None
```

**Decisão**: 

### 21 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/_enforcement_parts/enforcement_part_02.py:54` · **Effort**: 11min

> Refactor this function to reduce its Cognitive Complexity from 21 to the 15 allowed.

```python
       50              if (detail := ub.apply(kind, params, *args)) is not None
       51          ]
       52  
       53      @staticmethod
>>>    54      def _items_for(
       55          target: type, tag: str, category: c.EnforcementCategory, effective_layer: str
       56      ) -> Iterator[tuple[str, tuple[p.AttributeProbe, ...]]]:
       57          """Return category-specific (location, args) pairs for one rule tag.
       58  
```

**Decisão**: 

### 22 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_utilities/_mapper_access_parts/mapper_access_part_02.py:48` · **Effort**: 6min

> Define a constant instead of duplicating this literal "extract array index" 3 times.

```python
       44          ):
       45              sequence = current
       46          else:
       47              return r[t.JsonPayload | None].fail_op(
>>>    48                  "extract array index", c.ERR_MAPPER_NOT_A_SEQUENCE
       49              )
       50          index_result = FlextUtilitiesMapperAccess._normalize_array_index(
       51              array_match, len(sequence)
       52          )
```

**Decisão**: 

### 23 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/_utilities/dispatcher_execute.py:27` · **Effort**: 6min

> Define a constant instead of duplicating this literal "validate handler return payload" 3 times.

```python
       23  ) -> p.Result[t.JsonPayload]:
       24      result: p.Result[t.JsonPayload]
       25      if raw_output is None:
       26          result = dispatch_result.fail_op(
>>>    27              "validate handler return payload", c.ERR_HANDLER_RETURNED_NONE
       28          )
       29      elif isinstance(raw_output, p.Result):
       30          if raw_output.failure:
       31              result = dispatch_result.from_failure(raw_output)
```

**Decisão**: 

### 24 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/domain.py:66` · **Effort**: 21min

> Refactor this function to reduce its Cognitive Complexity from 31 to the 15 allowed.

```python
       62              result = id_a is not None and id_a == id_b
       63          return result
       64  
       65      @staticmethod
>>>    66      def compare_value_objects_by_value(
       67          obj_a: t.JsonPayload | prt.HasModelDump, obj_b: t.JsonPayload | prt.HasModelDump
       68      ) -> bool:
       69          """Compare two value objects by all attributes (value, not identity).
       70  
```

**Decisão**: 

### 25 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/guards.py:98` · **Effort**: 23min

> Refactor this function to reduce its Cognitive Complexity from 33 to the 15 allowed.

```python
       94              return contains in value
       95          return False
       96  
       97      @staticmethod
>>>    98      def _check_spec_ops(
       99          value: t.GuardInput,
      100          guard_spec: FlextModelsCollections.GuardCheckSpec,
      101          check_val: t.Numeric,
      102      ) -> bool:
```

**Decisão**: 

### 26 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/_utilities/mapper.py:102` · **Effort**: 9min

> Refactor this function to reduce its Cognitive Complexity from 19 to the 15 allowed.

```python
       98  
       99          return accessor
      100  
      101      @staticmethod
>>>   102      def transform(
      103          source: t.JsonMapping | m.ConfigMap,
      104          *,
      105          normalize: bool = False,
      106          strip_none: bool = False,
```

**Decisão**: 

### 27 · 🟠 CRITICAL · CODE_SMELL · `python:S1192`
**Local**: `src/flext_core/dispatcher.py:188` · **Effort**: 6min

> Define a constant instead of duplicating this literal "validate handler return payload" 3 times.

```python
      184              return None
      185          if u.container(raw_candidate) or u.pydantic_model(raw_candidate):
      186              return raw_candidate
      187          return dispatch_result.fail_op(
>>>   188              "validate handler return payload",
      189              c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE,
      190          )
      191  
      192      @staticmethod
```

**Decisão**: 

### 28 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/registry.py:80` · **Effort**: 6min

> Refactor this function to reduce its Cognitive Complexity from 16 to the 15 allowed.

```python
       76          cls._class_plugin_storage = {}  # MutableMapping[str, t.RegistrablePlugin]
       77          cls._class_registered_keys = set()  # set[str]
       78  
       79      @classmethod
>>>    80      def create(
       81          cls,
       82          dispatcher: p.Dispatcher | None = None,
       83          *,
       84          runtime: m.ServiceRuntime | None = None,
```

**Decisão**: 

### 29 · 🟠 CRITICAL · CODE_SMELL · `python:S3776`
**Local**: `src/flext_core/registry.py:431` · **Effort**: 7min

> Refactor this function to reduce its Cognitive Complexity from 17 to the 15 allowed.

```python
      427              else:
      428                  summary.errors.append(r.require_error(result))
      429          return self._finalize_summary(summary)
      430  
>>>   431      def register_plugin(
      432          self,
      433          category: str,
      434          name: str,
      435          plugin: t.RegistrablePlugin,
```

**Decisão**: 

### 30 · 🟡 MAJOR · VULNERABILITY · `githubactions:S8264`
**Local**: `.github/workflows/docs.yml:18` · **Effort**: 5min

> Move this read permission from workflow level to job level.

```yaml
       14        - ".github/workflows/docs.yml"
       15    workflow_dispatch:
       16  
       17  permissions:
>>>    18    contents: read
       19    pages: write
       20    id-token: write
       21  
       22  concurrency:
```

**Decisão**: 

### 31 · 🟡 MAJOR · VULNERABILITY · `githubactions:S8233`
**Local**: `.github/workflows/docs.yml:19` · **Effort**: 5min

> Move this write permission from workflow level to job level.

```yaml
       15    workflow_dispatch:
       16  
       17  permissions:
       18    contents: read
>>>    19    pages: write
       20    id-token: write
       21  
       22  concurrency:
       23    group: pages
```

**Decisão**: 

### 32 · 🟡 MAJOR · VULNERABILITY · `githubactions:S8233`
**Local**: `.github/workflows/docs.yml:20` · **Effort**: 5min

> Move this write permission from workflow level to job level.

```yaml
       16  
       17  permissions:
       18    contents: read
       19    pages: write
>>>    20    id-token: write
       21  
       22  concurrency:
       23    group: pages
       24    cancel-in-progress: false
```

**Decisão**: 

### 33 · 🟡 MAJOR · VULNERABILITY · `text:S8565`
**Local**: `pyproject.toml:-` · **Effort**: 5min

> Dependency versions are not predictable if the lock file (uv.lock, poetry.lock, pdm.lock or pylock.toml) is missing.


**Decisão**: 

### 34 · 🟡 MAJOR · CODE_SMELL · `python:S8786`
**Local**: `src/flext_core/_constants/regex.py:62` · **Effort**: 20min

> Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.

```python
       58      PATTERN_FORBIDDEN_FACADE_IMPORT: Final[str] = (
       59          r"^\s*from\s+(tests|examples|scripts)\.([\w.]+)\s+import\s+([\w,\s]+?)\s*$"
       60      )
       61      "Matches `from <forbidden>.<module> import …` lines (multiline source scan)."
>>>    62      PATTERN_EXAMPLE_RESULT_LINE: Final[str] = r"^[^\[][^\n]+: .+$"
       63      "Matches one normalized PASS/FAIL/GENERATED summary line emitted by examples."
       64  
       65      # === Pre-compiled regex authorities (consumers MUST use these) ===
       66      PATTERN_ENFORCE_RULE_ID_RE: ClassVar[t.RegexPattern] = re.compile(
```

**Decisão**: 

### 35 · 🟡 MAJOR · CODE_SMELL · `python:S3358`
**Local**: `src/flext_core/_decorators/_base.py:70` · **Effort**: 5min

> Extract this nested conditional expression into an independent statement.

```python
       66              return first_arg.logger
       67          module_name = (
       68              func_module
       69              if isinstance(func_module, str)
>>>    70              else (func.__module__ if callable(func) else __name__)
       71          )
       72          logger: pl.Logger = FlextUtilitiesLogging.fetch_logger(module_name)
       73          return logger
       74  
```

**Decisão**: 

### 36 · 🟡 MAJOR · CODE_SMELL · `python:S6796`
**Local**: `src/flext_core/_exceptions/_factories_parts/flextexceptionsfactories_part_01.py:72` · **Effort**: 5min

> Use a generic type parameter for this function instead of a "TypeVar".

```python
       68          )
       69          return resolved_options, resolved_options.error
       70  
       71      @staticmethod
>>>    72      def _normalize_params(
       73          params: TExceptionParams | None,
       74          params_type: type[TExceptionParams],
       75          update: dict[str, object | None],
       76      ) -> TExceptionParams:
```

**Decisão**: 

### 37 · 🟡 MAJOR · CODE_SMELL · `python:S5890`
**Local**: `src/flext_core/_lazy_parts/flextlazy_part_01.py:76` · **Effort**: 5min

> Assign to "_activating_core_beartype" a value of type "bool" instead of "PrivateAttr" or update its type hint.

```python
       72              ).FlextCoreBeartypeBootstrap.activate_package_beartype
       73          )
       74      )
       75  
>>>    76      _activating_core_beartype: bool = PrivateAttr(default=False)
       77  
       78      @computed_field(return_type=dict[str, int])
       79      @property
       80      def cache_stats(self) -> dict[str, int]:
```

**Decisão**: 

### 38 · 🟡 MAJOR · CODE_SMELL · `python:S8963`
**Local**: `src/flext_core/_models/_base_parts/flextmodelsbase_part_03.py:115` · **Effort**: 5min

> Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.

```python
      111                  description="Maximum delay between retries",
      112              ),
      113          ] = c.DEFAULT_MAX_DELAY_SECONDS
      114  
>>>   115      class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
      116          """Model with timestamp fields."""
      117  
      118  
      119  __all__: list[str] = ["FlextModelsBase"]
```

**Decisão**: 

### 39 · 🟡 MAJOR · CODE_SMELL · `python:S8963`
**Local**: `src/flext_core/_models/domain_event.py:29` · **Effort**: 5min

> Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.

```python
       25      Contains DomainEvent and helper utilities for event data normalization.
       26      Split into its own module so Entity can import without forward references.
       27      """
       28  
>>>    29      class Entry(m.IdentifiableMixin, m.TimestampedModel):
       30          """Base class for domain events."""
       31  
       32          message_type: str = mp.Field(
       33              "event",
```

**Decisão**: 

### 40 · 🟡 MAJOR · CODE_SMELL · `python:S8963`
**Local**: `src/flext_core/_models/entity.py:37` · **Effort**: 5min

> Refactor this Pydantic model to avoid multiple inheritance with conflicting configurations.

```python
       33      DomainEvent is imported from FlextModelsDomainEvent to break
       34      the forward-reference cycle that Pydantic cannot resolve.
       35      """
       36  
>>>    37      class Entity(m.TimestampedModel, m.IdentifiableMixin, m.VersionableMixin, Hashable):
       38          """Entity implementation - base class for domain entities with identity.
       39  
       40          Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
       41          - unique_id: Unique identifier (from IdentifiableMixin)
```

**Decisão**: 

### 41 · 🟡 MAJOR · CODE_SMELL · `python:S6794`
**Local**: `src/flext_core/_models/pydantic.py:115` · **Effort**: 5min

> Use a "type" statement instead of this "TypeAlias".

```python
      111      class RootModel[RootValueT](PydanticRootModel[RootValueT]):
      112          """Canonical RootModel exported through the FLEXT models facade."""
      113  
      114      # Pydantic field utilities
>>>   115      ConfigDict: TypeAlias = _PydanticConfigDict
      116      SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict
      117  
      118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
```

**Decisão**: 

### 42 · 🟡 MAJOR · CODE_SMELL · `python:S6794`
**Local**: `src/flext_core/_models/pydantic.py:116` · **Effort**: 5min

> Use a "type" statement instead of this "TypeAlias".

```python
      112          """Canonical RootModel exported through the FLEXT models facade."""
      113  
      114      # Pydantic field utilities
      115      ConfigDict: TypeAlias = _PydanticConfigDict
>>>   116      SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict
      117  
      118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
      120      # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
```

**Decisão**: 

### 43 · 🟡 MAJOR · CODE_SMELL · `python:S6796`
**Local**: `src/flext_core/_protocols/result.py:59` · **Effort**: 5min

> Use a generic type parameter for this function instead of a "TypeVar".

```python
       55          def exception(self) -> BaseException | None: ...
       56          @property
       57          def failure(self) -> bool: ...
       58          @property
>>>    59          def value(self) -> ResultT: ...
       60  
       61          def __enter__(self) -> Self: ...
       62  
       63          def __exit__(
```

**Decisão**: 

### 44 · 🟡 MAJOR · CODE_SMELL · `python:S6796`
**Local**: `src/flext_core/_protocols/result.py:72` · **Effort**: 5min

> Use a generic type parameter for this function instead of a "TypeVar".

```python
       68          ) -> None: ...
       69  
       70          def __or__[D](self, default: D) -> ResultT | D: ...
       71  
>>>    72          def unwrap(self) -> ResultT: ...
       73          def unwrap_or[D](self, default: D) -> ResultT | D: ...
       74          def unwrap_or_else[D](self, func: Callable[[], D]) -> ResultT | D: ...
       75  
       76          def flat_map[U](
```

**Decisão**: 

### 45 · 🟡 MAJOR · CODE_SMELL · `python:S5890`
**Local**: `src/flext_core/_result/base.py:39` · **Effort**: 5min

> Assign to "_exception" a value of type "Optional[BaseException]" instead of "PrivateAttr" or update its type hint.

```python
       35      error_code: str | None = None
       36      error_data: JsonDict | None = None
       37  
       38      _payload: T = PrivateAttr()
>>>    39      _exception: BaseException | None = PrivateAttr(default=None)
       40  
       41      @classmethod
       42      def reject_banned_result_parameterization(cls) -> None:
       43          """Reject ``FlextResult[None]`` and ``FlextResult[object]`` specializations."""
```

**Decisão**: 

### 46 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/_beartype/_alias_visitor.py:101` · **Effort**: 5min

> Either remove or fill this block of code.

```python
       97              case "sibling_models_type_checking":
       98                  if "_models" in src_file:
       99                      violation = _NO_VIOLATION
      100              case _:
>>>   101                  pass
      102          return violation
      103  
      104      @staticmethod
      105      def v_compatibility_alias(
```

**Decisão**: 

### 47 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_01.py:74` · **Effort**: 5min

> Either remove or fill this block of code.

```python
       70                          if not target.__name__.startswith(expected)
       71                          else NO_VIOLATION
       72                      )
       73              case _:
>>>    74                  pass
       75          return violation
       76  
       77      @staticmethod
       78      def _deep_nested(node: type, budget: int) -> str | None:
```

**Decisão**: 

### 48 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_01.py:107` · **Effort**: 5min

> Either remove or fill this block of code.

```python
      103                  ubh.has_runtime_protocol_marker(value)
      104                  or ubh.has_nested_namespace(value)
      105                  or ubh.has_abstract_contract(value)
      106              ):
>>>   107                  pass
      108              else:
      109                  return BARE_VIOLATION
      110          if (
      111              params.require_runtime_checkable
```

**Decisão**: 

### 49 · 🟡 MAJOR · CODE_SMELL · `python:S3358`
**Local**: `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_03.py:69` · **Effort**: 5min

> Extract this nested conditional expression into an independent statement.

```python
       65          )
       66          return (
       67              {"name": target_name}
       68              if settings_violation
>>>    69              else {"expected": expected_prefix_text, "actual": target_name}
       70              if prefix_violation
       71              else NO_VIOLATION
       72          )
       73  
```

**Decisão**: 

### 50 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/_beartype/deprecated_visitor.py:189` · **Effort**: 5min

> Either remove or fill this block of code.

```python
      185                              ),
      186                              _NO_VIOLATION,
      187                          )
      188              case _:
>>>   189                  pass
      190          return violation
```

**Decisão**: 

### 51 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/_beartype/method_visitor.py:109` · **Effort**: 5min

> Either remove or fill this block of code.

```python
      105                              "count": str(count),
      106                              "max": str(params.max_params),
      107                          }
      108              case _:
>>>   109                  pass
      110          return violation
```

**Decisão**: 

### 52 · 🟡 MAJOR · BUG · `python:S3699`
**Local**: `src/flext_core/_utilities/_logging_config_parts/logging_config_part_01.py:130` · **Effort**: 5min

> Remove this use of the output from "flush"; "flush" doesn’t return anything.

```python
      126  
      127          def _write_queued_message(self, msg: str) -> None:
      128              target_stream = self._target_stream
      129              _ = target_stream.write(msg)
>>>   130              _ = target_stream.flush()
      131              self.queue.task_done()
      132  
      133          def _worker(self) -> None:
      134              """Worker thread processing log queue."""
```

**Decisão**: 

### 53 · 🟡 MAJOR · CODE_SMELL · `python:S5864`
**Local**: `src/flext_core/_utilities/_parser_targets_parts/parser_targets_part_01.py:92` · **Effort**: 5min

> Replace this expression; Previous type checks suggest that "target" has type "type[T]" and isn't iterable.

```python
       88              if opts.default_factory is not None:
       89                  return opts.default_factory()
       90              raise ValueError(c.ERR_PARSER_VALUE_IS_NONE.format(field_prefix=fp))
       91          value_str = str(value)
>>>    92          options_text = [member.value for member in target]
       93          if not opts.case_insensitive:
       94              validation_result: p.Result[T] = FlextUtilitiesModel.validate_value(
       95                  target, value_str
       96              )
```

**Decisão**: 

### 54 · 🟡 MAJOR · CODE_SMELL · `python:S5864`
**Local**: `src/flext_core/_utilities/_parser_targets_parts/parser_targets_part_01.py:108` · **Effort**: 5min

> Replace this expression; Previous type checks suggest that "target" has type "type[T]" and isn't iterable.

```python
      104                      target_name=target_name,
      105                      options=options_text,
      106                  )
      107              )
>>>   108          for member in target:
      109              member_val = getattr(member, "value", None)
      110              if member_val is None:
      111                  continue
      112              if str(member_val).lower() == value_str.lower():
```

**Decisão**: 

### 55 · 🟡 MAJOR · CODE_SMELL · `python:S108`
**Local**: `src/flext_core/_utilities/guards.py:139` · **Effort**: 5min

> Either remove or fill this block of code.

```python
      135              result = FlextUtilitiesGuards._check_string_ops(value, guard_spec)
      136          if result:
      137              match guard_spec.contains:
      138                  case None:
>>>   139                      pass
      140                  case contains_value:
      141                      result = FlextUtilitiesGuardsTypeCore.container(
      142                          value
      143                      ) and FlextUtilitiesGuards._check_iterable_contains(
```

**Decisão**: 

### 56 · 🟡 MAJOR · CODE_SMELL · `python:S3358`
**Local**: `src/flext_core/context.py:188` · **Effort**: 5min

> Extract this nested conditional expression into an independent statement.

```python
      184          parent_token = (
      185              u.PARENT_CORRELATION_ID.set(parent_id)
      186              if parent_id
      187              else (
>>>   188                  u.PARENT_CORRELATION_ID.set(current)
      189                  if isinstance(current, str)
      190                  else None
      191              )
      192          )
```

**Decisão**: 

### 57 · ⚪ MINOR · CODE_SMELL · `python:S7504`
**Local**: `conftest.py:33` · **Effort**: 5min

> Remove this unnecessary `list()` call on an already iterable object.

```python
       29          and Path(getattr(existing_package, "__file__", "")).resolve() == init_file
       30      ):
       31          return
       32  
>>>    33      for module_name in list(sys.modules):
       34          if module_name == package_name or module_name.startswith(f"{package_name}."):
       35              sys.modules.pop(module_name, None)
       36  
       37      package_spec = importlib.util.spec_from_file_location(
```

**Decisão**: 

### 58 · ⚪ MINOR · CODE_SMELL · `python:S117`
**Local**: `conftest.py:111` · **Effort**: 2min

> Rename this local variable "T_co" to match the regular expression ^[_a-z][a-z0-9_]*$.

```python
      107      flext_context = core.FlextContext
      108      flext_settings = core.FlextSettings
      109  
      110      T = TypeVar("T")
>>>   111      T_co = TypeVar("T_co", covariant=True)
      112      empty_mapping: dict[str, object] = {}
      113      stub_names = {
      114          "User": _DocsStub,
      115          "Order": _DocsStub,
```

**Decisão**: 

### 59 · ⚪ MINOR · CODE_SMELL · `python:S7500`
**Local**: `examples/ex_01_flext_result_helpers.py:126` · **Effort**: 5min

> Replace this comprehension with passing the iterable to the collection constructor call

```python
      122          self.audit_check("map.failure", base_fail.map(lambda n: n + 1).failure)
      123          self.audit_check(
      124              "map.exception_to_failure",
      125              base_ok.map(
>>>   126                  lambda _: (_ for _ in ()).throw(ValueError("map exploded"))
      127              ).error,
      128          )
      129          self.audit_check(
      130              "flat_map.success",
```

**Decisão**: 

### 60 · ⚪ MINOR · CODE_SMELL · `python:S6353`
**Local**: `src/flext_core/_constants/regex.py:32` · **Effort**: 5min

> Use concise character class syntax '\w' instead of '[a-zA-Z0-9_]'.

```python
       28          r"-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?"
       29          r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?$"
       30      )
       31      "Pattern for SemVer and normalized PEP 440 version strings."
>>>    32      PATTERN_IDENTIFIER_WITH_UNDERSCORE: Final[str] = "^[a-zA-Z_][a-zA-Z0-9_]*$"
       33      "Pattern for identifiers that can start with underscore (context keys)."
       34      PATTERN_ISO8601_TIMESTAMP: Final[str] = (
       35          "^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}[Z+\\-][0-9:]*)?$"
       36      )
```

**Decisão**: 

### 61 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_03.py:160` · **Effort**: 2min

> Rename this field "BaseError" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      156  
      157  class FlextExceptionsBase:
      158      """BaseError and all typed exception subclasses."""
      159  
>>>   160      BaseError = FlextBaseError
      161  
      162  
      163  __all__: list[str] = ["FlextExceptionsBase"]
```

**Decisão**: 

### 62 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/cqrs.py:110` · **Effort**: 2min

> Rename this field "Pagination" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      106              t.NonEmptyStr | None,
      107              Field(description="Identity of the principal that issued this command."),
      108          ] = None
      109  
>>>   110      Pagination = _CqrsPagination
      111  
      112      class Query(m.ArbitraryTypesModel):
      113          """Query model for CQRS query operations."""
      114  
```

**Decisão**: 

### 63 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/domain_event.py:58` · **Effort**: 2min

> Rename this field "DomainEvent" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       54              description="Event data container",
       55              default_factory=lambda: mc.ConfigMap(root={}),
       56          )
       57  
>>>    58      DomainEvent = Entry
       59  
       60  
       61  __all__: t.MutableSequenceOf[str] = ["FlextModelsDomainEvent"]
```

**Decisão**: 

### 64 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:115` · **Effort**: 2min

> Rename this field "ConfigDict" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      111      class RootModel[RootValueT](PydanticRootModel[RootValueT]):
      112          """Canonical RootModel exported through the FLEXT models facade."""
      113  
      114      # Pydantic field utilities
>>>   115      ConfigDict: TypeAlias = _PydanticConfigDict
      116      SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict
      117  
      118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
```

**Decisão**: 

### 65 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:116` · **Effort**: 2min

> Rename this field "SettingsConfigDict" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      112          """Canonical RootModel exported through the FLEXT models facade."""
      113  
      114      # Pydantic field utilities
      115      ConfigDict: TypeAlias = _PydanticConfigDict
>>>   116      SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict
      117  
      118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
      120      # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
```

**Decisão**: 

### 66 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:118` · **Effort**: 2min

> Rename this field "Field" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      114      # Pydantic field utilities
      115      ConfigDict: TypeAlias = _PydanticConfigDict
      116      SettingsConfigDict: TypeAlias = _PydanticSettingsConfigDict
      117  
>>>   118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
      120      # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
      121      # function class attribute called through the facade (mixins.py:59 error).
      122      PrivateAttr = staticmethod(PrivateAttr)
```

**Decisão**: 

### 67 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:122` · **Effort**: 2min

> Rename this field "PrivateAttr" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      118      Field = staticmethod(_field)
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
      120      # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
      121      # function class attribute called through the facade (mixins.py:59 error).
>>>   122      PrivateAttr = staticmethod(PrivateAttr)
      123      SkipValidation = SkipValidation
      124      # Same unwrapped-class-attribute problem as PrivateAttr above: pyright
      125      # binds the bare decorator through the facade and infers the facade type
      126      # for every decorated property (reportIndexIssue on real consumers).
```

**Decisão**: 

### 68 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:123` · **Effort**: 2min

> Rename this field "SkipValidation" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      119      # NOTE (multi-agent): mro-ecfu — staticmethod wrap matches Field above and
      120      # u.PrivateAttr (_utilities/pydantic.py): pyright cannot model an unwrapped
      121      # function class attribute called through the facade (mixins.py:59 error).
      122      PrivateAttr = staticmethod(PrivateAttr)
>>>   123      SkipValidation = SkipValidation
      124      # Same unwrapped-class-attribute problem as PrivateAttr above: pyright
      125      # binds the bare decorator through the facade and infers the facade type
      126      # for every decorated property (reportIndexIssue on real consumers).
      127      computed_field = staticmethod(computed_field)
```

**Decisão**: 

### 69 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:131` · **Effort**: 2min

> Rename this field "AfterValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      127      computed_field = staticmethod(computed_field)
      128      field_validator = field_validator
      129  
      130      # Annotation validators
>>>   131      AfterValidator = AfterValidator
      132      BeforeValidator = BeforeValidator
      133      PlainValidator = PlainValidator
      134      WrapValidator = WrapValidator
      135  
```

**Decisão**: 

### 70 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:132` · **Effort**: 2min

> Rename this field "BeforeValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      128      field_validator = field_validator
      129  
      130      # Annotation validators
      131      AfterValidator = AfterValidator
>>>   132      BeforeValidator = BeforeValidator
      133      PlainValidator = PlainValidator
      134      WrapValidator = WrapValidator
      135  
      136      # Serializers
```

**Decisão**: 

### 71 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:133` · **Effort**: 2min

> Rename this field "PlainValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      129  
      130      # Annotation validators
      131      AfterValidator = AfterValidator
      132      BeforeValidator = BeforeValidator
>>>   133      PlainValidator = PlainValidator
      134      WrapValidator = WrapValidator
      135  
      136      # Serializers
      137      PlainSerializer = PlainSerializer
```

**Decisão**: 

### 72 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:134` · **Effort**: 2min

> Rename this field "WrapValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      130      # Annotation validators
      131      AfterValidator = AfterValidator
      132      BeforeValidator = BeforeValidator
      133      PlainValidator = PlainValidator
>>>   134      WrapValidator = WrapValidator
      135  
      136      # Serializers
      137      PlainSerializer = PlainSerializer
      138      WrapSerializer = WrapSerializer
```

**Decisão**: 

### 73 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:137` · **Effort**: 2min

> Rename this field "PlainSerializer" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      133      PlainValidator = PlainValidator
      134      WrapValidator = WrapValidator
      135  
      136      # Serializers
>>>   137      PlainSerializer = PlainSerializer
      138      WrapSerializer = WrapSerializer
      139  
      140      # Validation and serialization context helpers
      141      FieldInfo = FieldInfo
```

**Decisão**: 

### 74 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:138` · **Effort**: 2min

> Rename this field "WrapSerializer" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      134      WrapValidator = WrapValidator
      135  
      136      # Serializers
      137      PlainSerializer = PlainSerializer
>>>   138      WrapSerializer = WrapSerializer
      139  
      140      # Validation and serialization context helpers
      141      FieldInfo = FieldInfo
      142      FieldSerializationInfo = FieldSerializationInfo
```

**Decisão**: 

### 75 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:141` · **Effort**: 2min

> Rename this field "FieldInfo" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      137      PlainSerializer = PlainSerializer
      138      WrapSerializer = WrapSerializer
      139  
      140      # Validation and serialization context helpers
>>>   141      FieldInfo = FieldInfo
      142      FieldSerializationInfo = FieldSerializationInfo
      143      ValidationInfo = ValidationInfo
      144  
      145      type TypeAdapterType[T] = PydanticTypeAdapter[T]
```

**Decisão**: 

### 76 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:142` · **Effort**: 2min

> Rename this field "FieldSerializationInfo" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      138      WrapSerializer = WrapSerializer
      139  
      140      # Validation and serialization context helpers
      141      FieldInfo = FieldInfo
>>>   142      FieldSerializationInfo = FieldSerializationInfo
      143      ValidationInfo = ValidationInfo
      144  
      145      type TypeAdapterType[T] = PydanticTypeAdapter[T]
      146      TypeAdapter = PydanticTypeAdapter
```

**Decisão**: 

### 77 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:143` · **Effort**: 2min

> Rename this field "ValidationInfo" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      139  
      140      # Validation and serialization context helpers
      141      FieldInfo = FieldInfo
      142      FieldSerializationInfo = FieldSerializationInfo
>>>   143      ValidationInfo = ValidationInfo
      144  
      145      type TypeAdapterType[T] = PydanticTypeAdapter[T]
      146      TypeAdapter = PydanticTypeAdapter
      147  
```

**Decisão**: 

### 78 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:146` · **Effort**: 2min

> Rename this field "TypeAdapter" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      142      FieldSerializationInfo = FieldSerializationInfo
      143      ValidationInfo = ValidationInfo
      144  
      145      type TypeAdapterType[T] = PydanticTypeAdapter[T]
>>>   146      TypeAdapter = PydanticTypeAdapter
      147  
      148      # Schema and validator handlers
      149      GetCoreSchemaHandler = GetCoreSchemaHandler
      150      GetJsonSchemaHandler = GetJsonSchemaHandler
```

**Decisão**: 

### 79 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:149` · **Effort**: 2min

> Rename this field "GetCoreSchemaHandler" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      145      type TypeAdapterType[T] = PydanticTypeAdapter[T]
      146      TypeAdapter = PydanticTypeAdapter
      147  
      148      # Schema and validator handlers
>>>   149      GetCoreSchemaHandler = GetCoreSchemaHandler
      150      GetJsonSchemaHandler = GetJsonSchemaHandler
      151      GetPydanticSchema = GetPydanticSchema
      152  
      153      # Validation exception (re-exported so consumers avoid `import pydantic`)
```

**Decisão**: 

### 80 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:150` · **Effort**: 2min

> Rename this field "GetJsonSchemaHandler" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      146      TypeAdapter = PydanticTypeAdapter
      147  
      148      # Schema and validator handlers
      149      GetCoreSchemaHandler = GetCoreSchemaHandler
>>>   150      GetJsonSchemaHandler = GetJsonSchemaHandler
      151      GetPydanticSchema = GetPydanticSchema
      152  
      153      # Validation exception (re-exported so consumers avoid `import pydantic`)
      154      ValidationError = ValidationError
```

**Decisão**: 

### 81 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:151` · **Effort**: 2min

> Rename this field "GetPydanticSchema" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      147  
      148      # Schema and validator handlers
      149      GetCoreSchemaHandler = GetCoreSchemaHandler
      150      GetJsonSchemaHandler = GetJsonSchemaHandler
>>>   151      GetPydanticSchema = GetPydanticSchema
      152  
      153      # Validation exception (re-exported so consumers avoid `import pydantic`)
      154      ValidationError = ValidationError
      155  
```

**Decisão**: 

### 82 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:154` · **Effort**: 2min

> Rename this field "ValidationError" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      150      GetJsonSchemaHandler = GetJsonSchemaHandler
      151      GetPydanticSchema = GetPydanticSchema
      152  
      153      # Validation exception (re-exported so consumers avoid `import pydantic`)
>>>   154      ValidationError = ValidationError
      155  
      156      # Schema and JSON utilities (from pydantic_core)
      157      SchemaValidator = SchemaValidator
      158  
```

**Decisão**: 

### 83 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:157` · **Effort**: 2min

> Rename this field "SchemaValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      153      # Validation exception (re-exported so consumers avoid `import pydantic`)
      154      ValidationError = ValidationError
      155  
      156      # Schema and JSON utilities (from pydantic_core)
>>>   157      SchemaValidator = SchemaValidator
      158  
      159      # Settings sources (from pydantic_settings)
      160      EnvSettingsSource = EnvSettingsSource
      161      PydanticBaseSettingsSource = PydanticBaseSettingsSource
```

**Decisão**: 

### 84 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:160` · **Effort**: 2min

> Rename this field "EnvSettingsSource" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      156      # Schema and JSON utilities (from pydantic_core)
      157      SchemaValidator = SchemaValidator
      158  
      159      # Settings sources (from pydantic_settings)
>>>   160      EnvSettingsSource = EnvSettingsSource
      161      PydanticBaseSettingsSource = PydanticBaseSettingsSource
```

**Decisão**: 

### 85 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_models/pydantic.py:161` · **Effort**: 2min

> Rename this field "PydanticBaseSettingsSource" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      157      SchemaValidator = SchemaValidator
      158  
      159      # Settings sources (from pydantic_settings)
      160      EnvSettingsSource = EnvSettingsSource
>>>   161      PydanticBaseSettingsSource = PydanticBaseSettingsSource
```

**Decisão**: 

### 86 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_runtime/_base.py:25` · **Effort**: 2min

> Rename this field "Metadata" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       21  # mro-i6nq.8: Keep the runtime base free of unused provider passthroughs.
       22  class FlextRuntimeBase:
       23      """Foundational runtime helpers shared by higher runtime namespaces."""
       24  
>>>    25      Metadata: ClassVar[type[pl.Metadata] | None] = None
       26  
       27      @classmethod
       28      def _require_metadata_model(cls) -> type[pl.Metadata]:
       29          """Return the bound metadata model class or raise a runtime contract error."""
```

**Decisão**: 

### 87 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:126` · **Effort**: 2min

> Rename this field "TypeAdapter" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      122      # attribute as an instance variable, which breaks isinstance narrowing
      123      # for every union containing ``t.BaseModel`` (mypy [unreachable]).
      124      type BaseModel = pydantic.BaseModel
      125      type TypeAdapterType[T] = pydantic.TypeAdapter[T]
>>>   126      TypeAdapter = pydantic.TypeAdapter
      127      ConfigDict = pydantic.ConfigDict
      128      ImportString = pydantic.ImportString
      129      InstanceOf = pydantic.InstanceOf
      130      Secret = pydantic.Secret
```

**Decisão**: 

### 88 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:127` · **Effort**: 2min

> Rename this field "ConfigDict" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      123      # for every union containing ``t.BaseModel`` (mypy [unreachable]).
      124      type BaseModel = pydantic.BaseModel
      125      type TypeAdapterType[T] = pydantic.TypeAdapter[T]
      126      TypeAdapter = pydantic.TypeAdapter
>>>   127      ConfigDict = pydantic.ConfigDict
      128      ImportString = pydantic.ImportString
      129      InstanceOf = pydantic.InstanceOf
      130      Secret = pydantic.Secret
      131      SecretStr = pydantic.SecretStr
```

**Decisão**: 

### 89 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:128` · **Effort**: 2min

> Rename this field "ImportString" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      124      type BaseModel = pydantic.BaseModel
      125      type TypeAdapterType[T] = pydantic.TypeAdapter[T]
      126      TypeAdapter = pydantic.TypeAdapter
      127      ConfigDict = pydantic.ConfigDict
>>>   128      ImportString = pydantic.ImportString
      129      InstanceOf = pydantic.InstanceOf
      130      Secret = pydantic.Secret
      131      SecretStr = pydantic.SecretStr
      132      type SecretBytes = pydantic.SecretBytes
```

**Decisão**: 

### 90 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:129` · **Effort**: 2min

> Rename this field "InstanceOf" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      125      type TypeAdapterType[T] = pydantic.TypeAdapter[T]
      126      TypeAdapter = pydantic.TypeAdapter
      127      ConfigDict = pydantic.ConfigDict
      128      ImportString = pydantic.ImportString
>>>   129      InstanceOf = pydantic.InstanceOf
      130      Secret = pydantic.Secret
      131      SecretStr = pydantic.SecretStr
      132      type SecretBytes = pydantic.SecretBytes
      133  
```

**Decisão**: 

### 91 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:130` · **Effort**: 2min

> Rename this field "Secret" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      126      TypeAdapter = pydantic.TypeAdapter
      127      ConfigDict = pydantic.ConfigDict
      128      ImportString = pydantic.ImportString
      129      InstanceOf = pydantic.InstanceOf
>>>   130      Secret = pydantic.Secret
      131      SecretStr = pydantic.SecretStr
      132      type SecretBytes = pydantic.SecretBytes
      133  
      134      # IP types
```

**Decisão**: 

### 92 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:131` · **Effort**: 2min

> Rename this field "SecretStr" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      127      ConfigDict = pydantic.ConfigDict
      128      ImportString = pydantic.ImportString
      129      InstanceOf = pydantic.InstanceOf
      130      Secret = pydantic.Secret
>>>   131      SecretStr = pydantic.SecretStr
      132      type SecretBytes = pydantic.SecretBytes
      133  
      134      # IP types
      135      type IPvAnyAddress = pydantic.IPvAnyAddress
```

**Decisão**: 

### 93 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:140` · **Effort**: 2min

> Rename this field "StringConstraints" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      136      type IPvAnyInterface = pydantic.IPvAnyInterface
      137      type IPvAnyNetwork = pydantic.IPvAnyNetwork
      138  
      139      # Constraint helper types (runtime markers / classes)
>>>   140      StringConstraints = pydantic.StringConstraints
      141      UrlConstraints = pydantic.UrlConstraints
      142      ErrorDetails = pydantic_core.ErrorDetails
      143      ErrorType = core_schema.ErrorType
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
```

**Decisão**: 

### 94 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:141` · **Effort**: 2min

> Rename this field "UrlConstraints" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      137      type IPvAnyNetwork = pydantic.IPvAnyNetwork
      138  
      139      # Constraint helper types (runtime markers / classes)
      140      StringConstraints = pydantic.StringConstraints
>>>   141      UrlConstraints = pydantic.UrlConstraints
      142      ErrorDetails = pydantic_core.ErrorDetails
      143      ErrorType = core_schema.ErrorType
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
      145      InitErrorDetails = pydantic_core.InitErrorDetails
```

**Decisão**: 

### 95 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:142` · **Effort**: 2min

> Rename this field "ErrorDetails" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      138  
      139      # Constraint helper types (runtime markers / classes)
      140      StringConstraints = pydantic.StringConstraints
      141      UrlConstraints = pydantic.UrlConstraints
>>>   142      ErrorDetails = pydantic_core.ErrorDetails
      143      ErrorType = core_schema.ErrorType
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
      145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
```

**Decisão**: 

### 96 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:143` · **Effort**: 2min

> Rename this field "ErrorType" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      139      # Constraint helper types (runtime markers / classes)
      140      StringConstraints = pydantic.StringConstraints
      141      UrlConstraints = pydantic.UrlConstraints
      142      ErrorDetails = pydantic_core.ErrorDetails
>>>   143      ErrorType = core_schema.ErrorType
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
      145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
```

**Decisão**: 

### 97 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:144` · **Effort**: 2min

> Rename this field "ErrorTypeInfo" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      140      StringConstraints = pydantic.StringConstraints
      141      UrlConstraints = pydantic.UrlConstraints
      142      ErrorDetails = pydantic_core.ErrorDetails
      143      ErrorType = core_schema.ErrorType
>>>   144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
      145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
      148      AliasGenerator = pydantic.AliasGenerator
```

**Decisão**: 

### 98 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:145` · **Effort**: 2min

> Rename this field "InitErrorDetails" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      141      UrlConstraints = pydantic.UrlConstraints
      142      ErrorDetails = pydantic_core.ErrorDetails
      143      ErrorType = core_schema.ErrorType
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
>>>   145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
      148      AliasGenerator = pydantic.AliasGenerator
      149      AliasChoices = pydantic.AliasChoices
```

**Decisão**: 

### 99 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:148` · **Effort**: 2min

> Rename this field "AliasGenerator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      144      ErrorTypeInfo = pydantic_core.ErrorTypeInfo
      145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
>>>   148      AliasGenerator = pydantic.AliasGenerator
      149      AliasChoices = pydantic.AliasChoices
      150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
```

**Decisão**: 

### 100 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:149` · **Effort**: 2min

> Rename this field "AliasChoices" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      145      InitErrorDetails = pydantic_core.InitErrorDetails
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
      148      AliasGenerator = pydantic.AliasGenerator
>>>   149      AliasChoices = pydantic.AliasChoices
      150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
```

**Decisão**: 

### 101 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:150` · **Effort**: 2min

> Rename this field "AliasPath" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      146  
      147      # Annotation and alias helper types (runtime markers / classes)
      148      AliasGenerator = pydantic.AliasGenerator
      149      AliasChoices = pydantic.AliasChoices
>>>   150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
```

**Decisão**: 

### 102 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:151` · **Effort**: 2min

> Rename this field "Discriminator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      147      # Annotation and alias helper types (runtime markers / classes)
      148      AliasGenerator = pydantic.AliasGenerator
      149      AliasChoices = pydantic.AliasChoices
      150      AliasPath = pydantic.AliasPath
>>>   151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
```

**Decisão**: 

### 103 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:152` · **Effort**: 2min

> Rename this field "Tag" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      148      AliasGenerator = pydantic.AliasGenerator
      149      AliasChoices = pydantic.AliasChoices
      150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
>>>   152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
```

**Decisão**: 

### 104 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:153` · **Effort**: 2min

> Rename this field "ValidateAs" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      149      AliasChoices = pydantic.AliasChoices
      150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
>>>   153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
```

**Decisão**: 

### 105 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:154` · **Effort**: 2min

> Rename this field "WithJsonSchema" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      150      AliasPath = pydantic.AliasPath
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
>>>   154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
```

**Decisão**: 

### 106 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:155` · **Effort**: 2min

> Rename this field "SerializeAsAny" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      151      Discriminator = pydantic.Discriminator
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
>>>   155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
      159      FailFast = pydantic.FailFast
```

**Decisão**: 

### 107 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:156` · **Effort**: 2min

> Rename this field "SkipValidation" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      152      Tag = pydantic.Tag
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
>>>   156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
      159      FailFast = pydantic.FailFast
      160      OnErrorOmit = pydantic.OnErrorOmit
```

**Decisão**: 

### 108 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:157` · **Effort**: 2min

> Rename this field "AllowInfNan" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      153      ValidateAs = pydantic.ValidateAs
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
>>>   157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
      159      FailFast = pydantic.FailFast
      160      OnErrorOmit = pydantic.OnErrorOmit
```

**Decisão**: 

### 109 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:158` · **Effort**: 2min

> Rename this field "Strict" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      154      WithJsonSchema = pydantic.WithJsonSchema
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
>>>   158      Strict = pydantic.Strict
      159      FailFast = pydantic.FailFast
      160      OnErrorOmit = pydantic.OnErrorOmit
```

**Decisão**: 

### 110 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:159` · **Effort**: 2min

> Rename this field "FailFast" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      155      SerializeAsAny = pydantic.SerializeAsAny
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
>>>   159      FailFast = pydantic.FailFast
      160      OnErrorOmit = pydantic.OnErrorOmit
```

**Decisão**: 

### 111 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/pydantic.py:160` · **Effort**: 2min

> Rename this field "OnErrorOmit" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      156      SkipValidation = pydantic.SkipValidation
      157      AllowInfNan = pydantic.AllowInfNan
      158      Strict = pydantic.Strict
      159      FailFast = pydantic.FailFast
>>>   160      OnErrorOmit = pydantic.OnErrorOmit
```

**Decisão**: 

### 112 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/services.py:138` · **Effort**: 2min

> Rename this field "MapperInput" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      134  
      135      type ValidatorCallable = Callable[[ScalarOrModel], ScalarOrModel]
      136  
      137      type MapperCallable = Callable[[tp.JsonValue], tp.JsonValue]
>>>   138      MapperInput = MapperCallable | tp.JsonValue
      139      StrictValue = (
      140          t.Scalar
      141          | ConfigurationMapping
      142          | t.JsonList
```

**Decisão**: 

### 113 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_typings/services.py:139` · **Effort**: 2min

> Rename this field "StrictValue" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      135      type ValidatorCallable = Callable[[ScalarOrModel], ScalarOrModel]
      136  
      137      type MapperCallable = Callable[[tp.JsonValue], tp.JsonValue]
      138      MapperInput = MapperCallable | tp.JsonValue
>>>   139      StrictValue = (
      140          t.Scalar
      141          | ConfigurationMapping
      142          | t.JsonList
      143          | tuple[tp.JsonValue | t.Scalar, ...]
```

**Decisão**: 

### 114 · ⚪ MINOR · CODE_SMELL · `python:S5713`
**Local**: `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_01.py:32` · **Effort**: 1min

> Remove this redundant Exception class; it derives from another which is already caught.

```python
       28          package = sys.modules.get(package_name)
       29          if package is None:
       30              try:
       31                  package = importlib.import_module(package_name)
>>>    32              except (ImportError, ModuleNotFoundError):
       33                  return ()
       34          if not hasattr(package, "_LAZY_IMPORTS"):
       35              return ()
       36          lazy_module = importlib.import_module("flext_core.lazy")
```

**Decisão**: 

### 115 · ⚪ MINOR · CODE_SMELL · `python:S7500`
**Local**: `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_02.py:132` · **Effort**: 5min

> Replace this comprehension with passing the iterable to the collection constructor call

```python
      128      @staticmethod
      129      def function_param_names(fn: _types_mod.FunctionType) -> t.StrSequence:
      130          code = getattr(fn, "__code__", None)
      131          return (
>>>   132              tuple(name for name in code.co_varnames[: code.co_argcount])
      133              if isinstance(code, _types_mod.CodeType)
      134              else ()
      135          )
      136  
```

**Decisão**: 

### 116 · ⚪ MINOR · CODE_SMELL · `python:S5713`
**Local**: `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_03.py:92` · **Effort**: 1min

> Remove this redundant Exception class; it derives from another which is already caught.

```python
       88      def alias_contains_any(alias_value: t.TypeHintSpecifier | None) -> bool:
       89          h = FlextUtilitiesBeartypeHelpers
       90          try:
       91              return h.contains_any_recursive(alias_value, seen=set())
>>>    92          except (TypeError, AttributeError, RuntimeError, RecursionError):
       93              return "Any" in str(alias_value)
       94  
       95      @staticmethod
       96      def mutable_kind(value: p.AttributeProbe) -> str | None:
```

**Decisão**: 

### 117 · ⚪ MINOR · CODE_SMELL · `python:S5685`
**Local**: `src/flext_core/_utilities/_beartype/import_visitor.py:114` · **Effort**: 10min

> Move this assignment out of the argument list; ":=" operator is confusing in this context.

```python
      110                          {"import": name, "origin": origin, "file": filename}
      111                          for name, value in vars(module).items()
      112                          if _ImportBlacklistVisitor._is_private_family_import(
      113                              value,
>>>   114                              origin := _ubh.object_module_name_for(value) or "",
      115                              package,
      116                              families,
      117                              consumer_exempt=consumer_exempt,
      118                          )
```

**Decisão**: 

### 118 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/collection_merge.py:109` · **Effort**: 2min

> Rename this field "_MergeHandler" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
      105              if merge_result.failure:
      106                  return r[t.JsonMapping].from_failure(merge_result)
      107          return r[t.JsonMapping].ok(result)
      108  
>>>   109      _MergeHandler = Callable[[t.JsonMapping, t.JsonMapping], "p.Result[t.JsonMapping]"]
      110  
      111      _MERGE_STRATEGIES: ClassVar[Mapping[str, _MergeHandler]] = {
      112          _c_cqrs.MergeStrategy.REPLACE: _merge_replace,
      113          _c_cqrs.MergeStrategy.OVERRIDE: _merge_replace,
```

**Decisão**: 

### 119 · ⚪ MINOR · CODE_SMELL · `python:S5685`
**Local**: `src/flext_core/_utilities/discovery.py:49` · **Effort**: 10min

> Move this assignment out of the argument list; ":=" operator is confusing in this context.

```python
       45                  (name, config)
       46                  for name in dir(module)
       47                  if not name.startswith("_")
       48                  and (
>>>    49                      config := FlextUtilitiesDiscovery._factory_config_for(module, name)
       50                  )
       51                  is not None
       52              ],
       53              key=operator.itemgetter(0),
```

**Decisão**: 

### 120 · ⚪ MINOR · CODE_SMELL · `python:S5713`
**Local**: `src/flext_core/_utilities/dispatcher_execute.py:93` · **Effort**: 1min

> Remove this redundant Exception class; it derives from another which is already caught.

```python
       89      except (
       90          TypeError,
       91          ValueError,
       92          RuntimeError,
>>>    93          KeyError,
       94          AttributeError,
       95          OSError,
       96          LookupError,
       97          ArithmeticError,
```

**Decisão**: 

### 121 · ⚪ MINOR · CODE_SMELL · `python:S5713`
**Local**: `src/flext_core/_utilities/project_metadata.py:101` · **Effort**: 1min

> Remove this redundant Exception class; it derives from another which is already caught.

```python
       97                  FlextUtilitiesProjectMetadata.build_project_metadata(
       98                      project_root, document
       99                  )
      100              )
>>>   101          except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
      102              msg = f"cannot read project metadata from {root}: {exc}"
      103              return _Result[ppm.ProjectMetadata].fail(msg, exception=exc)
      104  
      105      @staticmethod
```

**Decisão**: 

### 122 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:48` · **Effort**: 2min

> Rename this field "Field" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       44      # Why: a bare ``Field = mp.Field`` binds ``_field``'s generic first parameter
       45      # (``default: DefaultT``) to ``self`` = ``FlextUtilities`` when accessed via the
       46      # facade, so ``u.Field(default_factory=...)`` was mis-inferred as returning
       47      # ``FlextUtilities`` (bogus reportAssignmentType). ``mp.Field`` already does this.
>>>    48      Field = staticmethod(mp.Field)
       49      PrivateAttr = staticmethod(PrivateAttr)
       50      SkipValidation = SkipValidation
       51  
       52      # Same unwrapped-class-attribute problem as Field/PrivateAttr above:
```

**Decisão**: 

### 123 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:49` · **Effort**: 2min

> Rename this field "PrivateAttr" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       45      # (``default: DefaultT``) to ``self`` = ``FlextUtilities`` when accessed via the
       46      # facade, so ``u.Field(default_factory=...)`` was mis-inferred as returning
       47      # ``FlextUtilities`` (bogus reportAssignmentType). ``mp.Field`` already does this.
       48      Field = staticmethod(mp.Field)
>>>    49      PrivateAttr = staticmethod(PrivateAttr)
       50      SkipValidation = SkipValidation
       51  
       52      # Same unwrapped-class-attribute problem as Field/PrivateAttr above:
       53      # pyright binds the bare decorator through the facade and infers the
```

**Decisão**: 

### 124 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:50` · **Effort**: 2min

> Rename this field "SkipValidation" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       46      # facade, so ``u.Field(default_factory=...)`` was mis-inferred as returning
       47      # ``FlextUtilities`` (bogus reportAssignmentType). ``mp.Field`` already does this.
       48      Field = staticmethod(mp.Field)
       49      PrivateAttr = staticmethod(PrivateAttr)
>>>    50      SkipValidation = SkipValidation
       51  
       52      # Same unwrapped-class-attribute problem as Field/PrivateAttr above:
       53      # pyright binds the bare decorator through the facade and infers the
       54      # facade type for every decorated property (reportIndexIssue downstream).
```

**Decisão**: 

### 125 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:61` · **Effort**: 2min

> Rename this field "AfterValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       57      field_serializer = field_serializer
       58      model_validator = model_validator
       59      model_serializer = model_serializer
       60  
>>>    61      AfterValidator = AfterValidator
       62      BeforeValidator = mp.BeforeValidator
       63      PlainValidator = PlainValidator
       64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
```

**Decisão**: 

### 126 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:62` · **Effort**: 2min

> Rename this field "BeforeValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       58      model_validator = model_validator
       59      model_serializer = model_serializer
       60  
       61      AfterValidator = AfterValidator
>>>    62      BeforeValidator = mp.BeforeValidator
       63      PlainValidator = PlainValidator
       64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
```

**Decisão**: 

### 127 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:63` · **Effort**: 2min

> Rename this field "PlainValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       59      model_serializer = model_serializer
       60  
       61      AfterValidator = AfterValidator
       62      BeforeValidator = mp.BeforeValidator
>>>    63      PlainValidator = PlainValidator
       64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
       67  
```

**Decisão**: 

### 128 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:64` · **Effort**: 2min

> Rename this field "WrapValidator" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       60  
       61      AfterValidator = AfterValidator
       62      BeforeValidator = mp.BeforeValidator
       63      PlainValidator = PlainValidator
>>>    64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
       67  
       68      ConfigDict = mp.ConfigDict
```

**Decisão**: 

### 129 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:65` · **Effort**: 2min

> Rename this field "PlainSerializer" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       61      AfterValidator = AfterValidator
       62      BeforeValidator = mp.BeforeValidator
       63      PlainValidator = PlainValidator
       64      WrapValidator = WrapValidator
>>>    65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
       67  
       68      ConfigDict = mp.ConfigDict
       69      FieldSerializationInfo = mp.FieldSerializationInfo
```

**Decisão**: 

### 130 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:66` · **Effort**: 2min

> Rename this field "WrapSerializer" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       62      BeforeValidator = mp.BeforeValidator
       63      PlainValidator = PlainValidator
       64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
>>>    66      WrapSerializer = WrapSerializer
       67  
       68      ConfigDict = mp.ConfigDict
       69      FieldSerializationInfo = mp.FieldSerializationInfo
       70      TypeAdapter = mp.TypeAdapter
```

**Decisão**: 

### 131 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:68` · **Effort**: 2min

> Rename this field "ConfigDict" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       64      WrapValidator = WrapValidator
       65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
       67  
>>>    68      ConfigDict = mp.ConfigDict
       69      FieldSerializationInfo = mp.FieldSerializationInfo
       70      TypeAdapter = mp.TypeAdapter
       71      create_model = create_model
       72      validate_call = validate_call
```

**Decisão**: 

### 132 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:69` · **Effort**: 2min

> Rename this field "FieldSerializationInfo" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       65      PlainSerializer = PlainSerializer
       66      WrapSerializer = WrapSerializer
       67  
       68      ConfigDict = mp.ConfigDict
>>>    69      FieldSerializationInfo = mp.FieldSerializationInfo
       70      TypeAdapter = mp.TypeAdapter
       71      create_model = create_model
       72      validate_call = validate_call
       73      with_config = with_config
```

**Decisão**: 

### 133 · ⚪ MINOR · CODE_SMELL · `python:S116`
**Local**: `src/flext_core/_utilities/pydantic.py:70` · **Effort**: 2min

> Rename this field "TypeAdapter" to match the regular expression ^[_a-z][_a-z0-9]*$.

```python
       66      WrapSerializer = WrapSerializer
       67  
       68      ConfigDict = mp.ConfigDict
       69      FieldSerializationInfo = mp.FieldSerializationInfo
>>>    70      TypeAdapter = mp.TypeAdapter
       71      create_model = create_model
       72      validate_call = validate_call
       73      with_config = with_config
       74  
```

**Decisão**: 

