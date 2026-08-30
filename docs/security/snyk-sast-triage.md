# Triagem Snyk Code (SAST) — flext-sh/flext-core

Gerado do scan Snyk (dump 2026-08-06). Bead: `mro-flgu`

## Resumo

**9 achados** — critical 0, high 0, medium 1, low 8

| categoria | achados |
|---|---|
| Use of Hardcoded Passwords | 5 |
| Use of Hardcoded Credentials | 4 |

## Como usar este documento

Cada achado traz o **código real** extraído da worktree (linha `>>>` = sink reportado), a regra completa e o CWE.
Preencha **Decisão**: `corrigir` / `falso-positivo` (registrar em `.snyk`) / `risco-aceito` (com prazo).

## Achados

### 1 · 🟡 MEDIUM · Use of Hardcoded Passwords

**Local**: `examples/_models/output.py:41` · **CWE**: -

```python
       37          LABEL_VALUE_SEPARATOR: ClassVar[str] = ": "
       38          RESULT_LINE_PATTERN: ClassVar[t.RegexPattern] = c.PATTERN_EXAMPLE_RESULT_LINE_RE
       39          TEMPLATE_BY_KIND: ClassVar[Mapping[OutputKind, OutputTemplate]] = (
       40              MappingProxyType({
>>>    41                  OutputKind.SUCCESS: OutputTemplate.SUCCESS,
       42                  OutputKind.FAIL: OutputTemplate.FAIL,
       43                  OutputKind.GENERATED: OutputTemplate.GENERATED,
       44              })
       45          )
```

**Decisão**:

### 2 · ⚪ LOW · Use of Hardcoded Credentials

**Local**: `tests/_models/_mixins/test_data_identity.py:43` · **CWE**: -

```python
       39          """Test identifiers and IDs."""
       40  
       41          model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
       42  
>>>    43          user_id: Annotated[str, m.Field(description="Default test user identifier")] = (
       44              "test_user_123"
       45          )
       46          session_id: Annotated[
       47              str, m.Field(description="Default test session identifier")
```

**Decisão**:

### 3 · ⚪ LOW · Use of Hardcoded Credentials

**Local**: `tests/_models/_mixins/test_data_values.py:45` · **CWE**: -

```python
       41          )
       42          config_key: Annotated[str, m.Field(description="Default test settings key")] = (
       43              "test_key"
       44          )
>>>    45          username: Annotated[str, m.Field(description="Default test username")] = (
       46              "test_user"
       47          )
       48          email: Annotated[str, m.Field(description="Default test email")] = (
       49              "test@example.com"
```

**Decisão**:

### 4 · ⚪ LOW · Use of Hardcoded Credentials

**Local**: `tests/integration/test_service.py:73` · **CWE**: -

```python
       69  
       70      def test_fetch_user_returns_applied_custom_entity(self) -> None:
       71          """fetch_user() returns previously applied custom user data verbatim."""
       72          service = self.UserQueryService()
>>>    73          user_id = "custom_user"
       74          custom = self.UserServiceEntity(
       75              unique_id=user_id,
       76              name="Custom User",
       77              email="custom@example.com",
```

**Decisão**:

### 5 · ⚪ LOW · Use of Hardcoded Credentials

**Local**: `tests/integration/test_service.py:147` · **CWE**: -

```python
      143      ) -> None:
      144          """Bound services resolve back and remain fully functional."""
      145          user_service = self.UserQueryService()
      146          notification_service = self.NotificationService()
>>>   147          user_id = "test_user_123"
      148          user_service.apply_user_data(
      149              user_id,
      150              self.UserServiceEntity(
      151                  unique_id=user_id,
```

**Decisão**:

### 6 · ⚪ LOW · Use of Hardcoded Passwords

**Local**: `tests/unit/test_result_factory_dip.py:169` · **CWE**: -

```python
      165  
      166      def test_fail_from_exception_redacts_sensitive_error_data_keys(self) -> None:
      167          exc = e.OperationError(
      168              "denied",
>>>   169              context={"password": "s3cret", "host": "db.example", "token": "t0k"},
      170          )
      171          result: p.Result[int] = r[int].fail(None, exception=exc)
      172          tm.fail(result, has="")
      173          assert result.error_data is not None
```

**Decisão**:

### 7 · ⚪ LOW · Use of Hardcoded Passwords

**Local**: `tests/unit/test_result_factory_dip.py:215` · **CWE**: -

```python
      211  
      212      def test_fail_explicit_error_data_redacts_sensitive_keys(self) -> None:
      213          result: p.Result[int] = r[int].fail(
      214              "denied",
>>>   215              error_data={"password": "s3cret", "host": "db.example", "token": "t0k"},
      216          )
      217          tm.fail(result, has="denied")
      218          assert result.error_data is not None
      219          tm.that(result.error_data.get("host"), eq="db.example")
```

**Decisão**:

### 8 · ⚪ LOW · Use of Hardcoded Passwords

**Local**: `tests/unit/test_result_factory_dip.py:226` · **CWE**: -

```python
      222  
      223      def test_fail_explicit_error_data_wins_but_still_redacts_with_exception(
      224          self,
      225      ) -> None:
>>>   226          exc = e.OperationError("x", context={"password": "from-exc", "host": "h"})
      227          result: p.Result[int] = r[int].fail(
      228              "denied",
      229              error_data={"password": "explicit", "host": "kept", "api_key": "k"},
      230              exception=exc,
```

**Decisão**:

### 9 · ⚪ LOW · Use of Hardcoded Passwords

**Local**: `tests/unit/test_result_factory_dip.py:229` · **CWE**: -

```python
      225      ) -> None:
      226          exc = e.OperationError("x", context={"password": "from-exc", "host": "h"})
      227          result: p.Result[int] = r[int].fail(
      228              "denied",
>>>   229              error_data={"password": "explicit", "host": "kept", "api_key": "k"},
      230              exception=exc,
      231          )
      232          tm.fail(result, has="denied")
      233          assert result.error_data is not None
```

**Decisão**:
