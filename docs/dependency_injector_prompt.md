# Prompt de aplicação do padrão dependency-injector

Use este prompt quando for evoluir o **flext-core** ou projetos que o consomem para garantir que o padrão de integração com `dependency-injector` (bridge L1 + container L2 + handlers L3) seja aplicado de forma coerente, sem quebrar a ABI e sem reintroduzir dicionários/registries paralelos.

```
Você é responsável por aplicar e reforçar o padrão de injeção de dependências baseado em dependency-injector em todo o código. Siga estas regras:

Contexto arquitetural
- L0.5 (runtime bridge): `flext_core.runtime.FlextRuntime` é a superfície única para acessar providers/containers/wiring (Provide, inject) e utilidades de configuração. Expanda capacidades apenas expondo novos helpers aqui.
- L1 (integração DI): `FlextRuntime.DependencyIntegration` mantém o container declarativo, providers tipados (Singleton/Factory/Resource) e providers.Configuration para defaults/overrides. Ajustes devem preservar a API pública exportada em `flext_core.__init__`.
- L2 (container): `FlextContainer` usa os providers do bridge para registrar serviços, fábricas e recursos com tipos genéricos, clonando-os para scopes sem copiar dicionários manuais. A API pública continua em `register`, `register_factory`, `register_resource`, `resolve`, `create_scope`, `configure`, etc.
- L3 (handlers/dispatcher): handlers são wired via `wire_modules` e decorators @inject/Provide reexportados pelo runtime. Não importe dependency-injector diretamente em L3.

Regras de implementação
- Prefira `providers.Resource` para clientes externos (DB/HTTP/queues), garantindo teardown/close, e remova boilerplate de lifecycle manual.
- Use `providers.Configuration` para sincronizar defaults/overrides validados pelo `FlextConfig`; evite merges manuais e preserve precedência de overrides.
- Registros devem usar providers tipados (Generic[T]) para manter type-safety; evite casts extras.
- Ao alterar caching/lifecycles, preserve configuração existente e a ABI pública (parâmetros e retornos expostos aos consumidores).
- Toda nova superfície DI deve ser exposta via o bridge/runtime ou fachadas existentes, nunca por imports diretos de dependency-injector em camadas superiores.
- Ao wiring, use `wire_modules` do container/bridge e respeite módulos/classes/handlers já registrados pelo dispatcher.

Aplicação em projetos dependentes
- Em projetos que usam flext-core, injete serviços/recursos via as mesmas fachadas (`FlextContainer`, `wire_modules`, decorators reexportados). Não criar containers alternativos nem acessar dependency-injector direto.
- Respeite os contratos de configuração herdados: defaults via `providers.Configuration` e overrides de usuário aplicados por `configure(...)`.
- Se precisar de novos recursos (ex.: novo provider), adicione primeiro ao bridge do flext-core e reexporte; só então consuma no projeto dependente.

Checklist final
- Garantir que testes `tests/unit/test_container.py` e `tests/unit/test_runtime.py` continuam passando.
- Não quebrar a ABI pública (imports documentados em `flext_core/__init__.py`).
- Evitar regressão de wiring automático em dispatcher/handlers.
```
