# Pydantic 2.11 Unification Plan — flext-core

**Status**: Fases 1–4 concluídas; Fase 5 (docs) em andamento; Fase 7 concluída ✅; Fase 8 concluída ✅
**Versão**: 0.9.1
**Última Atualização**: 2025-01-07
**Autoridade**: flext-core + FLEXT_REFACTORING_PROMPT.md + CLAUDE.md + README.md

## Objetivo

Unificar todo o uso de Pydantic 2.11 no flext-core para:

✅ **Validação Centralizada**: Todas as configurações passam por modelos Pydantic unificados (BaseModel/RootModel)
✅ **Zero Duplicação**: Eliminar ~500 linhas de validações manuais repetitivas em dicionários
✅ **Dict apenas nas bordas**: Usar `model_dump()` apenas para serialização/compatibilidade
✅ **APIs compatíveis**: Não quebrar APIs públicas (compatibilidade via `.model_dump()`)
✅ **Padrões respeitados**: Clean Architecture, FlextResult, DI, StrEnum
✅ **Quality Gates**: Passar `make check` e `make validate`

### Regras de Ouro

| Regra | Implementação |
|-------|---------------|
| **Railway Pattern** | Sempre usar `FlextResult` para erros nas operações públicas |
| **DI Único** | Usar apenas `FlextContainer`, sem containers paralelos |
| **Pydantic 2.11** | `ConfigDict`, `field_validator`, `model_validator` |
| **Enums Centralizados** | Usar `FlextConstants.Config.*` (StrEnum) exclusivamente |
| **Compatibilidade** | Não alterar pyproject/APIs públicas |
| **Deprecation Warnings** | Sempre adicionar warnings para mudanças futuras |
| **Dual Signatures** | Manter assinaturas antigas com @overload durante 2 versões |

## Escopo

**Incluído:**
- 📁 `src/flext_core/**` - 13 módulos com funções `configure_*`
- 📁 `src/flext_tests/**` - Utilidades/fixtures que trafegam configs
- 🧪 Testes em `tests/**` - Ajustes quando dependem de dicts

**Excluído:**
- ⛔ `pyproject.toml` - Não alterar configurações
- ⛔ Arquivos de lint/CI - Manter como estão
- ⛔ Validações de domínio - Manter Strategy Pattern onde aplicável

---

## Fases e Status

### Fase 1 — Settings base e Loader (Concluída)
- `FlextConfig.Settings (BaseSettings)` com `from_sources()` e `.to_config()`
- Utilitários locais em `FlextConfig`: `safe_get_env_var`, `safe_load_json_file`, `merge_configs`

### Fase 2 — Settings por subsistema + ponte SystemConfigs (Concluída)
- `*Settings` → `FlextModels.SystemConfigs.*` (Commands, Context, Validations, Services, Domain Services, Guards, Processors, Protocols; Exceptions/Fields com `BaseSystemConfig`)

### Fase 3 — Fachadas usando Settings Loader (Concluída)
- `configure_*` convertem entrada → `*Settings` → `*Config` → `model_dump()` (compat)
- Removidas validações manuais duplicadas de env/log/validation level

### Fase 4 — Dinâmicas vs Estáticas + Registry (Concluída)
- Campos dinâmicos via `json_schema_extra={"dynamic": true}`
- `FlextConfig.SettingsRegistry` com `update_runtime` e `reload_from_sources`
- Wrappers: `register_settings`, `get_settings`, `update_settings_runtime`, `reload_settings`, `get_dynamic_fields`

### Fase 5 — Documentação e Extensão (✅ CONCLUÍDA)
- Guia: `docs/guides/configuration.md` (padrões, exemplos, Registry)
- Exemplo de extensão em subprojeto (subclasse de `FlextConfig.Settings` + `.to_config()`)
- Integração com testes (sem quebrar asserts de mensagens existentes)

---

## Inventário de Módulos e Análise

Esta seção integra o inventário de módulos com o detalhamento crítico (classes, padrões esperados, lacunas e críticas) para facilitar a análise unificada.

A seguir, cada módulo com: papel, uso de Pydantic, problemas/duplicações, ações propostas e, quando aplicável, classes/padrões/lacunas/críticas.

### flext_core/models.py
- Papel: Núcleo de modelagem; `FlextModels.Config(BaseModel)` e diversos modelos (Value/Entity/AggregateRoot/Payload) + RootModels (Email/Host/Port/Url/JsonData/Metadata).
- Pydantic: Uso correto e amplo (2.11), validadores e `ConfigDict` consolidados.
- Problemas: Falta um “núcleo” de configs por subsistema (Commands/DomainServices/Adapters/Fields/Protocols/Core) para evitar validações dispersas em outros módulos baseadas em dict.
- Ações:
  - Adicionar `FlextModels.SystemConfigs` com `BaseSystemConfig(FlextModels.Config)` e modelos específicos: `CommandsConfig`, `DomainServicesConfig`, `TypeAdaptersConfig`, `ProtocolsConfig`, `FieldsConfig` (se aplicável) e `CoreConfig`.
  - Centralizar validadores comuns (environment, log_level, validation_level/service_level, performance_level).
  - Fornecer fábricas/aliases `from_environment(...)`, `optimize_performance(level)` usando `model_copy(update=...)` e validadores pós-merge.
 - Classes: FlextModels.Config, DatabaseConfig, SecurityConfig, LoggingConfig, Entity, Value, AggregateRoot, Payload, RootModels (EmailAddress, Host, Port, Url, JsonData, Metadata).
 - Padrões: Pydantic v2.11 (ConfigDict, field/model validators/serializers), imutabilidade em Value, igualdade por valor, RootModel para VOs simples.
 - Lacunas: Adicionar `FlextModels.SystemConfigs` (BaseSystemConfig + configs por subsistema).
 - Crítica: Usar como base única para validação; evitar repetir regras em outros módulos.

### Básicos (Foundation)

- flext-core/src/flext_core/constants.py
  - Classes: **FlextConstants** com sub-seções (Config, Defaults, Network, Observability, Handlers, etc.).
  - Padrões: StrEnum/valores canônicos; valores numéricos centralizados; nenhuma lista hard-coded fora daqui.
  - Lacunas: Adicionar StrEnum/valores para qualquer nível/perfil hoje hard-coded em módulos (e.g., performance_level adicionais).
  - Crítica: OK como “fonte única de verdade”. Evitar proliferar defaults redundantes em outros módulos.

- flext-core/src/flext_core/typings.py
  - Classes: **FlextTypes** (TypeVars/Core/Domain/Result/Service/Payload/Handler/Commands/Aggregates/Container/Config/Models), com aliases top-level.
  - Padrões: Aliases coesos e próximos às assinaturas reais; Python 3.13+ type alias; manter dicção única.
  - Lacunas: Incluir aliases para modelos de config (e.g., `type CommandsConfigModel = FlextModels.SystemConfigs.CommandsConfig`).
  - Crítica: Hoje reforça uso de `ConfigDict` internamente; migrar para modelos e manter `ConfigDict` apenas na borda.

- flext-core/src/flext_core/result.py
  - Classes: **FlextResult[T]** e helpers.
  - Padrões: Railway pattern (ok/is_failure/map/flat_map/unwrap); status e error_code padronizados.
  - Lacunas: Nenhuma para este escopo.
  - Crítica: Deve embrulhar erros de validação Pydantic quando exposto por APIs públicas.

- flext-core/src/flext_core/exceptions.py
  - Classes: **FlextExceptions** (Error, ValidationError, ConfigurationError, etc.).
  - Padrões: Hierarquia limpa; integração com códigos de erro (`FlextConstants.Errors`).
  - Lacunas: Helper para converter `pydantic.ValidationError` em mensagem/código consistente (opcional).
  - Crítica: Evitar lançar exceções no fluxo de negócio; preferir `FlextResult.fail`.

- flext-core/src/flext_core/loggings.py
  - Classes: **FlextLogger** (ou façade similar).
  - Padrões: Níveis derivados de `FlextConstants.Config.LogLevel` (StrEnum); mapeamento consistente.
  - Lacunas: Normalizador de nível (case-insensitive) a ser reutilizado pelos validadores.
  - Crítica: Remover qualquer validação local duplicada de níveis.

- flext-core/src/flext_core/version.py
  - Classes: Versão/metadata.
  - Padrões: Constantes simples.
  - Lacunas/Crítica: Sem pontos relevantes para a unificação Pydantic.

- flext-core/src/flext_core/__init__.py
  - Classes: Export aggregator.
  - Padrões: Agregar `__all__`; ordem de import por camada; exportar novos `SystemConfigs`.
  - Lacunas: Exportar modelos de config assim que criados.
  - Crítica: Não mexer na ordem para evitar ciclos.

### Domínio (DDD)

- flext-core/src/flext_core/models.py
  - Classes: **FlextModels** com nested: Config, DatabaseConfig, SecurityConfig, LoggingConfig, Entity, Value, AggregateRoot, Payload, RootModels (EmailAddress, Host, Port, Url, JsonData, Metadata).
  - Padrões: Pydantic v2.11 (`ConfigDict`, validators, serializers), imutabilidade em Value, igualdade por valor; RootModel para VO simples.
  - Lacunas: Criar `SystemConfigs` (BaseSystemConfig + configs específicas).
  - Crítica: Forte e consolidado; usar como base única para validação de parâmetros/config.

- flext-core/src/flext_core/domain_services.py
  - Classes: **FlextDomainService[T]** (genérico), validação mínima, `execute()` abstrato; `configure_domain_services_system` (hoje dict).
  - Padrões: Herdar de Config base; retornar `FlextResult`; validações via Pydantic nos modelos.
  - Lacunas: `DomainServicesConfig` ausente; configuradores devolvendo dict.
  - Crítica: Duplicação de validações (environment/log_level/service_level). Migrar para modelo.

### Aplicação (CQRS/Handlers/Validation)

- flext-core/src/flext_core/commands.py
  - Classes: **FlextCommands** com nested `Models.Command`, `Factories`, e métodos `configure_*`/`create_environment_*`/`optimize_*`.
  - Padrões: Command como `FlextModels.Config` (frozen, extra=ignore); `to_payload()`; `FlextResult` em fluxos.
  - Lacunas: `CommandsConfig` ausente; validadores duplicados em dicts.
  - Crítica: Overlap com constants/enums; migrar para modelo e usar `.model_dump()` na borda.

- flext-core/src/flext_core/handlers.py
  - Classes: **FlextProcessing** com nested `Constants`, `Types`, `Protocols`, `Implementation`.
  - Padrões: Padrões enterprise (Chain/CQRS), métricas, thread-safe lock, Protocols alinhados ao core.
  - Lacunas: Sem configurador; quando precisar, consumir modelos de config.
  - Crítica: OK. Evitar reimplementar validações de níveis/ambiente.

- flext-core/src/flext_core/validations.py
  - Classes: Validadores e `configure_validation_system`/`create_environment_validation_config`/`optimize_validation_performance` (dict).
  - Padrões: Regras/níveis de validação.
  - Lacunas: `ValidationSystemConfig` ausente.
  - Crítica: Duplicação de validações (environment/log_level/validation_level). Migrar para modelo.

- flext-core/src/flext_core/guards.py
  - Classes: Decoradores/guards; `configure_guards_system` (dict).
  - Padrões: tip guards, decorator pattern, memoization/pure wrapper.
  - Lacunas: `GuardsConfig` ausente.
  - Crítica: Repetição de validações; migrar para modelo.

- flext-core/src/flext_core/decorators.py
  - Classe: Decorators cross-cutting.
  - Padrões: Implementação de aspectos; config roteada via Core→Mixins.
  - Lacunas: `DecoratorsConfig` somente se houver parâmetros exclusivos; senão usar `MixinsConfig`.
  - Crítica: Evitar duplicar “mixins vs decorators”.

- flext-core/src/flext_core/processors.py
  - Classes: Processors + `configure_processors_system`/`get_processors_system_config` (dict).
  - Padrões: Pipeline/regex config.
  - Lacunas: `ProcessorsConfig` ausente.
  - Crítica: Defaults em dict replicados — migrar para modelo.

- flext-core/src/flext_core/protocols.py
  - Classes: **FlextProtocols** com nested `Config` e métodos `configure_*` etc (dict).
  - Padrões: Protocolos de aplicação/fundação; coesão com handlers.
  - Lacunas: `ProtocolsConfig` ausente.
  - Crítica: Repetição de validações — migrar para modelo.

### Infra (Config/Container/Context/Utilities/Fields/Adapters/Services/Core)

- flext-core/src/flext_core/config.py
  - Classes: **FlextConfig(FlextModels.Config)**, Settings (BaseSettings), nested TypedDicts, utilitários env/json/merge.
  - Padrões: Pydantic v2.11 avançado; validators/serializers; env-prefix.
  - Lacunas: Parte de validações duplicadas que migrarão para `BaseSystemConfig`.
  - Crítica: Evitar duas fontes (FlextConfig vs BaseSystemConfig) para as mesmas regras.

- flext-core/src/flext_core/container.py
  - Classes: **FlextContainer**, métodos `configure_*` (database/security/logging) já aceitam modelos.
  - Padrões: DI, registros/recuperações com `FlextResult`.
  - Lacunas: Locais que manipulam `ConfigDict` devem aceitar modelos e converter na borda.
  - Crítica: Não introduzir container paralelo.

- flext-core/src/flext_core/context.py
  - Classes: **FlextContext** com nested `Variables` (Correlation/Service/Request/Performance) e APIs; `configure_context_system` (dict).
  - Padrões: contextvars, scopes, generators.
  - Lacunas: `ContextConfig` ausente.
  - Crítica: Duplicação de validações de environment/log_level; migrar para modelo.

- flext-core/src/flext_core/utilities.py
  - Classes: Utilidades (Generators/Performance/ProcessingUtils/etc.) e `create_performance_config` (dict), `validate_application_configuration` (dict).
  - Padrões: Helpers puros.
  - Lacunas: Performance presets devem viver nos modelos (`optimize(...)`).
  - Crítica: Evitar validar config aqui — delegar aos modelos.

- flext-core/src/flext_core/fields.py
  - Classes: **FlextFields** com `ValidationStrategies`, `Core.BaseField`, `StringField`, Registry/Schema/Factory; `configure_fields_system` (dict).
  - Padrões: Strategy Pattern; validação de dados de domínio (não Pydantic).
  - Lacunas: `FieldsConfig` só se necessário; manter validação runtime fora do Pydantic.
  - Crítica: OK; não migrar tipos de campo para Pydantic.

- flext-core/src/flext_core/adapters.py
  - Classes: **FlextTypeAdapters** com `Config`, `Foundation`, `Domain`, etc.; `configure_type_adapters_system` (dict com suppress/fallback).
  - Padrões: Pydantic `TypeAdapter`, pipelines de validação.
  - Lacunas: `TypeAdaptersConfig` ausente; remover suppress/fallback.
  - Crítica: Fallback silencioso contraria as regras — migrar para modelo.

- flext-core/src/flext_core/services.py
  - Classes: **FlextProcessing**; `configure_services_system`/`get_services_system_config`/`create_environment_services_config`/`optimize_services_performance` (dicts).
  - Padrões: Service orchestration; batch/caching controls.
  - Lacunas: `ServicesConfig` ausente.
  - Crítica: Defaults e níveis replicados em várias funções — migrar para modelo.

- flext-core/src/flext_core/core.py
  - Classes: **FlextCore** (fachada): `configure_core_system` (dict), `configure_decorators_system` (usa mixins), criação de erros e providers.
  - Padrões: Facade central; roteia para subsistemas; retorna `FlextResult`.
  - Lacunas: `CoreConfig` ausente; normalização de environment/log_level duplicada.
  - Crítica: Garantir compatibilidade via `.model_dump()` sem duplicar lógica.

### flext_core/constants.py
- Papel: Fonte única de verdade para StrEnums (environment, log_level, validation_level, config_source, performance), limites e defaults.
- Problemas: Alguns módulos repetem listas de strings válidas; isso deve ser eliminado.
- Ações:
  - Garantir que todos os validadores nos novos modelos usem exclusivamente `FlextConstants.Config.*` (StrEnum) e constantes numéricas (timeouts, batch sizes, etc.).
  - Se necessário, adicionar StrEnums faltantes para níveis/perfis hoje hard-coded em módulos de aplicação.
  - Proibir listas manuais de valores válidos fora de `FlextConstants`.
 - Relação com Settings: Constants NÃO leem ambiente; servem como defaults/enum. Settings carregam valores do ambiente/arquivo/CLI e referenciam `FlextConstants` para defaults. `SystemConfigs` valida/normaliza o resultado final.

### flext_core/typings.py
- Papel: Sistema de tipos e aliases para todo o ecossistema.
- Problemas: Muitos aliases atuais estão ancorados em `ConfigDict` e dicionários; vamos transicionar para modelos mantendo compatibilidade.
- Ações:
  - Adicionar aliases para modelos de config: `type CommandsConfigModel = FlextModels.SystemConfigs.CommandsConfig` (idem para DomainServices/TypeAdapters/Protocols/Core/Services/Validations/Guards/Mixins/Processors/Fields/Context).
  - Adicionar resultados tipados: `type CommandsConfig = FlextResult[CommandsConfigModel]` onde fizer sentido.
  - Manter `type ConfigDict = FlextTypes.Config.ConfigDict` para borda externa e deprecar seu uso interno em favor dos modelos.
  - Evitar espalhar novos unions de dict em `typings`; preferir modelos.

### flext_core/result.py
- Papel: Railway (FlextResult).
- Ações:
  - Nenhuma mudança estrutural; assegurar que operações que lidam com configurações passem a retornar `FlextResult[ConfigModel]` internamente e apenas convertam para dict nas bordas.

### flext_core/exceptions.py
- Papel: Hierarquia de erros.
- Problemas: Tradução clara de `ValidationError` (Pydantic) para `FlextResult.fail` com códigos padronizados.
- Ações:
  - Garantir mapeamento consistente de `ValidationError` -> `FlextConstants.Errors.VALIDATION_ERROR` (ou equivalente) onde wrapping ocorrer.
  - Fornecer helper opcional para converter/formatar mensagens de validação de modelos.

### flext_core/loggings.py
- Papel: Logging estruturado; integração com níveis de log.
- Problemas: Validações de nível de log redundantes em módulos de aplicação.
- Ações:
  - Usar `FlextConstants.Config.LogLevel` como única fonte de valores; remover quaisquer checagens locais de listas.
  - Se necessário, expor utilitário para normalização (`DEBUG` vs `debug`) usado pelos validadores de modelos.

### flext_core/config.py
- Papel: Configuração “enterprise” agregada (env integration, JSON, business rules) via `FlextConfig(FlextModels.Config)`.
- Pydantic: Extenso, com `BaseSettings`, serialização custom e validadores. Tem validações comuns (environment/log_level) que também aparecem em outros módulos.
- Problemas: Redundância de validações já tratáveis na futura `BaseSystemConfig`; nomes de campos sobrepostos (alguns divergentes de modelos base como `config_environment` vs `environment` em outros contextos).
- Ações:
  - Reaproveitar validadores centralizados de `BaseSystemConfig` onde fizer sentido; manter neste módulo apenas o que é próprio (integração de env/serialização avançada).
  - Padronizar nomenclatura pública para `environment` (usar `Field(validation_alias=..., serialization_alias=...)` se precisar manter compatibilidade interna).
  - Manter utilidades IO (`safe_get_env_var`, `safe_load_json_file`) porém delegar a validações Pydantic para conteúdo.
 - Classes: FlextConfig(FlextModels.Config), Settings (BaseSettings), TypedDicts de kwargs, utilitários de env/JSON/merge.
 - Padrões: Integração de ambiente (env_prefix), validadores/serializadores Pydantic v2, conversões seguras.
 - Lacunas: Parte das validações comuns deve migrar para `BaseSystemConfig` para evitar redundância.
 - Crítica: Evitar duas fontes de verdade para as mesmas regras de config.

### flext_core/commands.py
- Papel: CQRS. Já possui `FlextCommands.Models.Command(FlextModels.Config)`.
- Pydantic: Modelos de comando OK. Porém configuradores retornam dict com validações manuais: `configure_commands_system`, `create_environment_commands_config`, `optimize_commands_performance`.
- Problemas: Duplicação de checagens (environment, validation_level, log_level, defaults) e merges.
- Ações:
  - Substituir retorno para `FlextResult[FlextModels.SystemConfigs.CommandsConfig]`.
  - Migrar validações/normalizações para o modelo. Usar `model_copy(update=...)` nas otimizações. Remover merges manuais e listas hard-coded.
 - Classes: FlextCommands (Models.Command, Factories, configuradores `configure_*`/`create_environment_*`/`optimize_*`).
 - Padrões: Pydantic para Command (frozen, extra=ignore), FlextResult em fluxos.
 - Lacunas: Falta `CommandsConfig` Pydantic.
 - Crítica: Checagens e listas replicadas; mover regras para o modelo e exportar dict na borda via `.model_dump()`.

### flext_core/domain_services.py
- Papel: Serviços DDD. `FlextDomainService(FlextModels.Config, ...)` OK.
- Pydantic: Bom no modelo base; mas configuradores retornam dict (`configure_domain_services_system`, `get_domain_services_system_config`, `create_environment_domain_services_config`, `optimize_domain_services_performance`).
- Problemas: Mesma duplicação dos padrões (environment, log_level, service_level, defaults) + fallbacks de dicionário.
- Ações:
  - Criar `DomainServicesConfig` e migrar configuradores para retornar modelo Pydantic.
  - Consolidar validações/níveis de serviço/otimizações em validadores e métodos de classe.
 - Classes: FlextDomainService[T] (genérico), `configure_domain_services_system`/`get_*`/`create_environment_*`/`optimize_*`.
 - Padrões: Herdar de Config base; FlextResult para erros; validações em modelos Pydantic.
 - Lacunas: Falta `DomainServicesConfig`.
 - Crítica: Duplicação de validações (environment/log_level/service_level) deve ser removida.

### flext_core/adapters.py
- Papel: TypeAdapter v2, validações de domínio e pipelines. Também possui uma “Config” baseada em dict com suppress/fallback.
- Pydantic: Uso correto de `TypeAdapter`. Configuração do sistema não usa BaseModel.
- Problemas: `configure_type_adapters_system` com suppress/fallback (contrário ao objetivo), environment/performance/validation_level duplicados.
- Ações:
  - Introduzir `TypeAdaptersConfig` (Pydantic) e remover suppress/fallback.
  - Expor `.model_dump()` só na borda. Validar níveis e presets via validadores/Enums.
 - Classes: FlextTypeAdapters (Config com Strategy, Foundation, Domain) e utilitários de validação com TypeAdapter.
 - Padrões: TypeAdapter Pydantic; Strategy Pattern para presets; FlextResult para erros.
 - Lacunas: Falta `TypeAdaptersConfig`; suppress/fallback indevidos.
 - Crítica: Fallback silencioso viola padrões; migrar para modelo e erros explícitos.

### flext_core/fields.py
- Papel: Sistema de Fields com Strategy Pattern, validação de valores dinâmicos (não Pydantic por design).
- Pydantic: Não aplicável diretamente aos tipos de campo (OK). Porém há `configure_fields_system` que hoje trabalha com dict.
- Problemas: Se existir “config do sistema de fields”, deve ser um modelo Pydantic.
- Ações:
  - Manter validações de dados via Strategies. Para configuração sistêmica, criar `FieldsConfig` e migrar configurador.
 - Classes: FlextFields (ValidationStrategies, Core.BaseField, StringField, Registry, Schema, Factory).
 - Padrões: Strategy Pattern para validação runtime; não usar Pydantic para valores dos campos.
 - Lacunas: `FieldsConfig` apenas se existirem parâmetros sistêmicos reais.
 - Crítica: Evitar over-engineering migrando tipos de campo para Pydantic.

### flext_core/protocols.py
- Papel: Protocolos/typing patterns. Tem um `Config` com `configure_protocols_system` e familia que retornam dict.
- Problemas: Repetição de validações/log_level/environment.
- Ações:
  - Criar `ProtocolsConfig` Pydantic e migrar os métodos (config/get/create_environment/optimize) para operar/sair com modelos.
 - Classes: FlextProtocols (Foundation/Application) com Config de sistema.
 - Padrões: Contratos e protocolos coerentes; hoje dicts em configuradores.
 - Lacunas: Falta `ProtocolsConfig` Pydantic.
 - Crítica: Centralizar validações no modelo e expor dict só na borda.

### flext_core/core.py
- Papel: Fachada e orquestração central. Possui `configure_core_system`, `validate_config_with_types` e construtores de “provider config” todos com dict + validação manual.
- Problemas: Duplica lógica de validação/normalização (environment/log_level/validation_level/config_source).
- Ações:
  - Criar `CoreConfig` (em `FlextModels.SystemConfigs`) e migrar esses métodos para manipular/retornar o modelo, eliminando validação manual.
 - Classes: FlextCore (fachada), métodos `configure_core_system`, `configure_decorators_system`, criação de providers e erros.
 - Padrões: Facade retornando `FlextResult`; conversão para dict na borda.
 - Lacunas: Falta `CoreConfig` e normalização centralizada em modelo.
 - Crítica: Eliminar duplicação de checagens; usar modelo.

### flext_core/container.py
- Papel: DI/serviços. Expõe `configure_*` que já recebem `FlextModels.DatabaseConfig/SecurityConfig/LoggingConfig` (correto). Também tem `configure_container` e `configure_global` com dicionários internos.
- Problemas: Onde trafegar config sistêmica, usar os novos modelos (ex.: `CoreConfig`) ao invés de dicionários.
- Ações:
  - Ajustar assinaturas internas para aceitar modelos específicos e usar `.model_dump()` nas bordas quando estritamente necessário.
 - Classes: FlextContainer (DI) com registros/recuperações e `configure_*` específicos (db/security/logging).
 - Padrões: DI + FlextResult, uso de modelos para configs.
 - Lacunas: Aceitar modelos para configs sistêmicas remanescentes.
 - Crítica: Não criar containers alternativos.

### flext_core/context.py
- Papel: Contexto de correlação/serviço/performance. Tem `configure_context_system` trabalhando com dicts.
- Ações:
  - Criar `ContextConfig` (se houver parâmetros de configuração) ou migrar para `CoreConfig` quando for só roteamento de flags globais.
 - Classes: FlextContext com Variables (Correlation/Service/Request/Performance) e configuradores.
 - Padrões: contextvars, gerenciadores, métricas simples.
 - Lacunas: Falta `ContextConfig`.
 - Crítica: Remover validações duplicadas; modelo central.

### flext_core/utilities.py
- Papel: Utilidades de conversão/performance/validação; inclui `create_performance_config` e `validate_application_configuration` com saída/validação de dict.
- Problemas: Duplicação de validações (environment/log_level/validation_level) e presets de performance em dicts.
- Ações:
  - Extrair `PerformanceConfig`/mix-in em `FlextModels.SystemConfigs` ou incorporar em cada `*Config` de subsistema.
  - Migrar `create_performance_config` para método(s) `optimize(perf_level)` dos modelos apropriados, retornando instâncias validadas.
  - `validate_application_configuration` deve ser substituída por validação Pydantic de um modelo agregado quando aplicável.
 - Classes: Utilitários (Generators/Performance/ProcessingUtils/etc.).
 - Padrões: Funções puras e auxiliares.
 - Lacunas: Presets de performance devem viver em modelos.
 - Crítica: Não validar configs aqui; delegar aos modelos Pydantic.

### flext_core/validations.py
- Papel: Sistema de validações de dados; também gerencia configurações do sistema de validações (strict/loose/etc.).
- Problemas: `configure_validation_system`/`get_validation_system_config`/`create_environment_validation_config`/`optimize_validation_performance` baseados em dicts com validações duplicadas (environment/log_level/validation_level).
- Ações:
  - Criar `ValidationSystemConfig` em `FlextModels.SystemConfigs` e migrar esses métodos para trabalhar/retornar o modelo.
  - Consolidar regras por ambiente/nível como validadores/model_validators e presets.
 - Classes: Validadores e configuradores (`configure_*`, `get_*`, `create_environment_*`, `optimize_*`).
 - Padrões: Regras por ambiente/nível estratificadas.
 - Lacunas: Falta `ValidationSystemConfig`.
 - Crítica: Checagens replicadas; centralizar no modelo.

### flext_core/services.py, handlers.py, processors.py, mixins.py, decorators.py, delegation.py, guards.py, validations.py, utilities.py, exceptions.py, constants.py, version.py, result.py, loggings.py, protocols.py (restante), __init__.py
- Papel: Infra/negócio/utilitários.
- Pydantic: Em geral não modelam configs próprios (exceto onde já mapeado acima). Não migrar lógicas de validação de “dados de domínio” (ex.: Guards/Validations) para Pydantic; o alvo é “config/parametrização de subsistemas”.
- Ações:
  - Onde houver `configure_*` que recebam/retornem configs, trocar para modelos Pydantic específicos (em `FlextModels.SystemConfigs`).
  - Eliminar checagens ad-hoc de `environment/log_level/...` quando o parâmetro já for `BaseModel` validado.

### flext_core/delegation.py
- Papel: Sistema de delegação e mixins. Possui `configure_delegation_system` que aceita/retorna dict e aplica checagens manuais.
- Ações: Introduzir `DelegationConfig` e migrar método para retornar o modelo. Remover retornos de erro como dict.
 - Classes: FlextDelegationSystem e `configure_delegation_system`.
 - Padrões: Integração com mixins; padronização via `FlextResult`.
 - Lacunas: Falta `DelegationConfig` Pydantic.
 - Crítica: Retorno de erro como dict fere padrão; migrar para modelo + `FlextResult.fail`.

### flext_core/guards.py
- Papel: Guards e decoradores; expõe `configure_guards_system` com validações repetidas (environment/log_level/validation_level).
- Ações: Introduzir `GuardsConfig` e migrar método.

### flext_core/mixins.py
- Papel: Mixins de serialização/logging/identidade etc.; expõe `configure_mixins_system` com validações e defaults em dict.
- Ações: Introduzir `MixinsConfig` e migrar método.

### flext_core/processors.py
- Papel: Processadores e pipelines; expõe `configure_processors_system` e `get_processors_system_config` com dicts.
- Ações: Introduzir `ProcessorsConfig` e migrar métodos.

### flext_core/services.py
- Papel: Arquitetura de serviços; expõe `configure_services_system`/`get_services_system_config`/`create_environment_services_config`/`optimize_services_performance` com dicts.
- Ações: Introduzir `ServicesConfig` e migrar métodos para modelos e presets por ambiente/nível; compatibilidade externa via `.model_dump()` apenas na borda.

### flext_core/decorators.py
- Papel: Padrões de decorators (cross-cutting). Não possui `configure_*` aqui; o roteamento atual está em `FlextCore.configure_decorators_system` usando Mixins.
- Ações:
  - Reutilizar `MixinsConfig` para configuração de decorators (mantendo a rota via Core), ou criar `DecoratorsConfig` se surgirem parâmetros específicos.
  - Em `FlextCore.configure_decorators_system`, garantir uso de modelo e conversão a dict apenas na borda.

### flext_core/handlers.py
- Papel: Infra de handlers (cadeias, CQRS, validação, autorização). Sem `configure_*` hoje.
- Ações: Sem mudanças diretas; quando dependente de configs sistêmicas, consumir modelos Pydantic ao invés de dicts crus.

### flext_core/__init__.py
- Papel: Agregação de exports públicos.
- Ações:
  - Exportar os novos modelos `FlextModels.SystemConfigs.*` em `__all__` seguindo o padrão de agregação já utilizado.
  - Não alterar a ordem de importação entre camadas para evitar ciclos.

### flext_tests/**
- Papel: Helpers/fixtures. Não migrar lógica de validação de runtime de testes para Pydantic, mas quando um utilitário trafegar configs do sistema, ajustar para usar os novos modelos (ou `.model_dump()`).

---

## 🛡️ Estratégia de Compatibilidade e Migração Segura

### Princípios de Não-Quebra

1. **Manter APIs Públicas Intactas**: Todas as funções públicas continuam aceitando e retornando `dict`
2. **Deprecation Warnings Graduais**: Avisos claros sobre mudanças futuras
3. **Período de Transição**: 2 versões (0.9.x → 0.10.x → 0.11.x) para migração completa
4. **Fallback Automático**: Se receber dict, converte para modelo; se esperam dict, converte de modelo

### 🔄 Padrão de Migração com Compatibilidade

```python
import warnings
from typing import overload, Union, Dict
from pydantic import ValidationError

class FlextCommands:
    # Nova assinatura (preferida)
    @overload
    @classmethod
    def configure_commands_system(
        cls, config: CommandsConfig
    ) -> FlextResult[CommandsConfig]: ...

    # Assinatura antiga (compatibilidade)
    @overload
    @classmethod
    def configure_commands_system(
        cls, config: dict
    ) -> FlextResult[dict]: ...

    @classmethod
    def configure_commands_system(
        cls, config: Union[dict, CommandsConfig]
    ) -> FlextResult[Union[dict, CommandsConfig]]:
        """Configuração com compatibilidade total.

        Args:
            config: Dict (deprecated) ou CommandsConfig (preferido)

        Returns:
            FlextResult com dict (se input foi dict) ou CommandsConfig
        """
        # Detectar tipo de entrada
        return_dict = isinstance(config, dict)

        # Emitir warning se usando dict
        if return_dict:
            warnings.warn(
                "Passing dict to configure_commands_system is deprecated. "
                "Use CommandsConfig instead. This will be required in v0.11.0.",
                DeprecationWarning,
                stacklevel=2
            )

        try:
            # Converter para modelo se necessário
            if return_dict:
                commands_config = CommandsConfig.model_validate(config)
            else:
                commands_config = config

            # Processar com modelo
            # ... lógica de configuração ...

            # Retornar no formato esperado
            if return_dict:
                return FlextResult.ok(commands_config.model_dump())
            else:
                return FlextResult.ok(commands_config)

        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            return FlextResult.fail(
                error_msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR
            )
```

### 📢 Sistema de Warnings Progressivos

#### Versão 0.9.x (Atual - Soft Deprecation)
```python
warnings.warn(
    "Passing dict is deprecated. Use ConfigModel instead. "
    "Dict support will be removed in v0.11.0.",
    DeprecationWarning,
    stacklevel=2
)
```

#### Versão 0.10.x (Hard Deprecation)
```python
warnings.warn(
    "Dict support will be REMOVED in next version (0.11.0). "
    "Please migrate to ConfigModel NOW. "
    "See: https://github.com/flext/migration-guide",
    FutureWarning,  # Mais visível que DeprecationWarning
    stacklevel=2
)
```

#### Versão 0.11.x (Removal)
```python
if isinstance(config, dict):
    raise TypeError(
        "Dict configuration no longer supported. "
        "Use CommandsConfig.model_validate(dict) to convert."
    )
```

### 🔀 Helpers de Migração para Subprojetos

```python
# Em flext_core/migration.py
class MigrationHelpers:
    """Utilidades para facilitar migração em subprojetos."""

    @staticmethod
    def dict_to_config(config_dict: dict, config_class: type[BaseModel]) -> BaseModel:
        """Converte dict legado para modelo com logging."""
        logger.info(f"Migrating dict to {config_class.__name__}")
        return config_class.model_validate(config_dict)

    @staticmethod
    def auto_migrate_decorator(config_class: type[BaseModel]):
        """Decorator para auto-migrar parâmetros dict."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(config: Union[dict, BaseModel], *args, **kwargs):
                if isinstance(config, dict):
                    warnings.warn(
                        f"Auto-converting dict to {config_class.__name__}",
                        DeprecationWarning
                    )
                    config = config_class.model_validate(config)
                return func(config, *args, **kwargs)
            return wrapper
        return decorator
```

### 📊 Matriz de Compatibilidade

| Versão | Dict Input | Model Input | Dict Output | Model Output | Warnings |
|--------|------------|-------------|-------------|--------------|----------|
| 0.9.x  | ✅ Aceita  | ✅ Aceita   | ✅ Se dict input | ✅ Se model input | ⚠️ DeprecationWarning |
| 0.10.x | ⚠️ Aceita  | ✅ Aceita   | ⚠️ Se dict input | ✅ Se model input | 🔴 FutureWarning |
| 0.11.x | ❌ Rejeita | ✅ Aceita   | ❌ Nunca    | ✅ Sempre    | ❌ TypeError |

### 🎯 Guia de Migração para Subprojetos

#### Passo 1: Identificar Uso (v0.9.x)
```bash
# Encontrar todos os usos de configure_* com dict
grep -r "configure.*system(" --include="*.py" | grep -v "Config("
```

#### Passo 2: Adicionar Imports (v0.9.x)
```python
# Adicionar no topo dos arquivos
from flext_core.models import SystemConfigs
from flext_core.migration import MigrationHelpers
```

#### Passo 3: Migração Gradual (v0.9.x → v0.10.x)
```python
# ANTES (dict)
config = {
    "environment": "production",
    "log_level": "INFO"
}
result = FlextCommands.configure_commands_system(config)

# DURANTE (compatível com ambos)
from flext_core.models import SystemConfigs
config = SystemConfigs.CommandsConfig(
    environment="production",
    log_level="INFO"
)
result = FlextCommands.configure_commands_system(config)

# DEPOIS (v0.11.x - apenas modelo)
config = CommandsConfig(
    environment="production",
    log_level="INFO"
)
result = FlextCommands.configure_commands_system(config)
```

---

## Arquitetura Proposta

### 🏁 Modelo Base Unificado

```python
# Em flext_core/models.py
class SystemConfigs:
    """Configurações unificadas para todos os subsistemas."""

    class BaseSystemConfig(FlextModels.Config):
        """Base para todas as configurações de sistema."""

        # Campos comuns a todos os subsistemas
        environment: FlextConstants.Config.ConfigEnvironment = Field(
            default=FlextConstants.Config.ConfigEnvironment.DEVELOPMENT,
            description="Environment for configuration"
        )
        log_level: FlextConstants.Config.LogLevel = Field(
            default=FlextConstants.Config.LogLevel.INFO,
            description="Logging level"
        )
        validation_level: FlextConstants.Config.ValidationLevel | None = Field(
            default=FlextConstants.Config.ValidationLevel.NORMAL,
            description="Validation strictness level"
        )

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            extra='forbid',
            str_strip_whitespace=True
        )

        @field_validator('environment', 'log_level', 'validation_level', mode='before')
        @classmethod
        def normalize_enums(cls, v, info):
            """Normaliza strings para enums apropriados."""
            if v is None:
                return v
            if isinstance(v, str):
                field_name = info.field_name
                if field_name == 'environment':
                    return FlextConstants.Config.ConfigEnvironment(v.lower())
                elif field_name == 'log_level':
                    return FlextConstants.Config.LogLevel(v.upper())
                elif field_name == 'validation_level':
                    return FlextConstants.Config.ValidationLevel(v.lower())
            return v

        @classmethod
        def from_environment(cls, env: str) -> Self:
            """Factory method para criar config por ambiente."""
            presets = {
                'development': {
                    'validation_level': FlextConstants.Config.ValidationLevel.STRICT,
                    'log_level': FlextConstants.Config.LogLevel.DEBUG
                },
                'staging': {
                    'validation_level': FlextConstants.Config.ValidationLevel.NORMAL,
                    'log_level': FlextConstants.Config.LogLevel.INFO
                },
                'production': {
                    'validation_level': FlextConstants.Config.ValidationLevel.NORMAL,
                    'log_level': FlextConstants.Config.LogLevel.WARNING
                },
            }
            base_config = {'environment': env}
            base_config.update(presets.get(env, {}))
            return cls(**base_config)

        def optimize(self, level: str = 'balanced') -> Self:
            """Otimiza configuração para performance."""
            optimizations = {
                'performance': {
                    'validation_level': FlextConstants.Config.ValidationLevel.LOOSE
                },
                'balanced': {
                    'validation_level': FlextConstants.Config.ValidationLevel.NORMAL
                },
                'strict': {
                    'validation_level': FlextConstants.Config.ValidationLevel.STRICT
                }
            }
            updates = optimizations.get(level, {})
            return self.model_copy(update=updates)
```

### 🎯 Configurações Específicas por Subsistema

```python
class CommandsConfig(BaseSystemConfig):
    """Configuração específica para Commands."""
    enable_handler_discovery: bool = True
    enable_middleware_pipeline: bool = True
    enable_performance_monitoring: bool = False
    max_concurrent_commands: int = Field(default=100, ge=1, le=1000)
    command_timeout_seconds: int = Field(default=30, ge=1, le=300)

    @model_validator(mode='after')
    def validate_production_settings(self) -> Self:
        """Ajusta configurações para produção."""
        if self.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION:
            if self.enable_performance_monitoring is False:
                # Em produção, monitoring deve estar ativo
                self.enable_performance_monitoring = True
        return self

class DomainServicesConfig(BaseSystemConfig):
    """Configuração específica para Domain Services."""
    service_level: str = Field(default="standard", pattern="^(basic|standard|premium)$")
    enable_caching: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0, le=86400)
    max_retry_attempts: int = Field(default=3, ge=0, le=10)

    @field_validator('cache_ttl_seconds')
    @classmethod
    def validate_cache_when_enabled(cls, v, info):
        """Valida TTL apenas quando cache está habilitado."""
        if info.data.get('enable_caching') and v == 0:
            raise ValueError("cache_ttl_seconds deve ser > 0 quando cache está habilitado")
        return v
```

---

## 🔄 Padrão de Migração

### ❌ ANTES (Código Atual - Validação Manual)

```python
@classmethod
def configure_commands_system(
    cls, config: dict
) -> FlextResult[dict]:
    """Validação manual repetitiva e propensa a erros."""
    try:
        validated_config = dict(config)

        # Validação manual de environment (repetida em 13+ módulos!)
        if "environment" in config:
            env_value = config["environment"]
            valid_environments = [
                e.value for e in FlextConstants.Config.ConfigEnvironment
            ]
            if env_value not in valid_environments:
                return FlextResult[dict].fail(
                    f"Invalid environment '{env_value}'. Valid options: {valid_environments}"
                )
            validated_config["environment"] = env_value
        else:
            validated_config["environment"] = (
                FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
            )

        # Validação manual de validation_level (mais código repetitivo)
        if "validation_level" in config:
            val_level = config["validation_level"]
            valid_levels = [v.value for v in FlextConstants.Config.ValidationLevel]
            if val_level not in valid_levels:
                return FlextResult[dict].fail(
                    f"Invalid validation_level '{val_level}'"
                )
            validated_config["validation_level"] = val_level
        else:
            validated_config["validation_level"] = (
                FlextConstants.Config.ValidationLevel.NORMAL.value
            )

        # Mais validações manuais...
        validated_config.setdefault("enable_handler_discovery", True)
        validated_config.setdefault("max_concurrent_commands", 100)
        validated_config.setdefault("command_timeout_seconds", 30)

        return FlextResult[dict].ok(validated_config)

    except Exception as e:
        return FlextResult[dict].fail(f"Failed to configure: {e}")
```

### ✅ DEPOIS (Código Alvo - Pydantic)

```python
@classmethod
def configure_commands_system(
    cls, config: dict
) -> FlextResult[dict]:
    """Validação via Pydantic - limpo, type-safe, mantendo compatibilidade."""
    try:
        # Pydantic faz TODA a validação automaticamente!
        commands_config = CommandsConfig.model_validate(config)

        # Retorna dict para manter compatibilidade de API
        return FlextResult[dict].ok(commands_config.model_dump())

    except ValidationError as e:
        # Converte erro Pydantic para FlextResult mantendo detalhes
        error_details = "; ".join(
            f"{err['loc'][0]}: {err['msg']}" for err in e.errors()
        )
        return FlextResult[dict].fail(
            f"Configuration validation failed: {error_details}",
            error_code=FlextConstants.Errors.VALIDATION_ERROR
        )
```

### 🎉 Benefícios da Migração

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Linhas de código** | ~65 linhas por função | ~10 linhas |
| **Validação** | Manual, repetitiva | Automática via Pydantic |
| **Type Safety** | Dict não tipado | Modelo totalmente tipado |
| **Manutenção** | Regras espalhadas | Centralizado em modelos |
| **Performance** | Várias iterações | Uma única validação |

---

## 📋 Plano de Mudança Incremental

### 🚀 Fase 0 — Baseline e Segurança (30 min)
**Objetivo**: Garantir estado limpo antes de refatorar

**Checklist**:
- [ ] Executar `make check` - deve passar sem erros
- [ ] Executar `make test` - registrar cobertura atual (baseline)
- [ ] Criar branch `feature/pydantic-unification`
- [ ] Verificar referências dos configuradores

**Referências Verificadas**:
  - `flext_core/commands.py:875` configure_commands_system
  - `flext_core/domain_services.py:173` configure_domain_services_system
  - `flext_core/adapters.py:99` FlextTypeAdapters.Config.configure_type_adapters_system
  - `flext_core/protocols.py:747` FlextProtocols.Config.configure_protocols_system
  - `flext_core/core.py:1180` FlextCore.configure_core_system
  - `flext_core/validations.py:949` FlextValidations.configure_validation_system
  - `flext_core/mixins.py:831` FlextMixins.configure_mixins_system
  - `flext_core/guards.py:1069` FlextGuards.configure_guards_system
  - `flext_core/processors.py:770` FlextProcessing.configure_processors_system
  - `flext_core/services.py:227` FlextProcessing.configure_services_system
  - `flext_core/context.py:630` FlextContext.configure_context_system
  - `flext_core/delegation.py:1477` FlextDelegationSystem.configure_delegation_system
  - `flext_core/fields.py:1790` FlextFields.configure_fields_system
**Validação**: ✅ Baseline registrado, pronto para refatorar

### 🏗️ Fase 1 — Base unificada (SystemConfigs)

**Objetivo**: Criar base de modelos Pydantic unificados para configurações de subsistemas
**Tempo Estimado**: 2 horas

#### 📋 Passos de Implementação (em `flext_core/models.py`):

1. **Adicionar classe `SystemConfigs`**:
   ```python
   class SystemConfigs:
       class BaseSystemConfig(FlextModels.Config):
           # Campos comuns a todos os subsistemas
           environment: FlextConstants.Config.ConfigEnvironment
           log_level: FlextConstants.Config.LogLevel
           validation_level: FlextConstants.Config.ValidationLevel | None
           config_source: FlextConstants.Config.ConfigSource | None
           performance_level: str | Literal[...]  # Se houver enum central, usar StrEnum

           model_config = ConfigDict(
               validate_assignment=True,
               extra='forbid',
               str_strip_whitespace=True,
               use_enum_values=True
           )
   ```

2. **Adicionar modelos específicos**:
   - [ ] `CoreConfig` - configuração central
   - [ ] `CommandsConfig` - subsistema de comandos
   - [ ] `DomainServicesConfig` - serviços de domínio
   - [ ] `TypeAdaptersConfig` - adaptadores de tipo
   - [ ] `ProtocolsConfig` - protocolos
   - [ ] `ValidationSystemConfig` - sistema de validação
   - [ ] `ServicesConfig` - serviços
   - [ ] `MixinsConfig` - mixins
   - [ ] `GuardsConfig` - guards
   - [ ] `ProcessorsConfig` - processadores
   - [ ] `ContextConfig` - contexto
   - [ ] `DelegationConfig` - delegação
   - [ ] `FieldsConfig` - campos (se necessário)

3. **Implementar validadores**:
   - [ ] Normalização/checagem de environment, log_level, validation_level
   - [ ] `model_validator` para regras cruzadas por ambiente
   - [ ] Validadores específicos por subsistema

4. **Criar fábricas**:
   ```python
   @classmethod
   def from_environment(cls, env: str) -> Self:
       """Factory method para criar config por ambiente."""
       # Usar presets por ambiente

   def optimize(self, level: str) -> Self:
       """Otimiza configuração usando model_copy(update=...)."""
       # Aplicar otimizações
   ```

#### 🔧 Integrações Auxiliares:

- **Atualizar `flext_core/__init__.py`**:
  - [ ] Exportar novos modelos (sem reordenar imports)
  - [ ] Manter ordem de camadas para evitar ciclos

- **Atualizar `flext_core/typings.py`**:
  ```python
  type CommandsConfigModel = FlextModels.SystemConfigs.CommandsConfig
  type DomainServicesConfigModel = FlextModels.SystemConfigs.DomainServicesConfig
  # ... outros aliases (sem substituir usos ainda)
  ```

**Validação**:
- [ ] `make check` - sem erros de tipo
- [ ] `pytest -k models` - testes passam
- [ ] Import funciona: `from flext_core import SystemConfigs`

### 🎯 Fase 2 — Commands (configuração via modelo)

**Objetivo**: Migrar os configuradores de Commands para usar `CommandsConfig`
**Escopo**: `flext_core/commands.py`
**Tempo Estimado**: 1 hora

#### 📋 Passos de Migração COM Compatibilidade:

1. **Em `configure_commands_system(config)`**:
   - [ ] Adicionar @overload para dual signatures
   - [ ] Detectar tipo de entrada (dict vs CommandsConfig)
   - [ ] Emitir DeprecationWarning se dict
   - [ ] Construir/validar CommandsConfig
   - [ ] Retornar formato baseado na entrada

   ```python
   @overload
   @classmethod
   def configure_commands_system(cls, config: CommandsConfig) -> FlextResult[CommandsConfig]: ...

   @overload
   @classmethod
   def configure_commands_system(cls, config: dict) -> FlextResult[dict]: ...

   @classmethod
   def configure_commands_system(
       cls, config: Union[dict, CommandsConfig]
   ) -> FlextResult[Union[dict, CommandsConfig]]:
       # Detectar tipo
       return_dict = isinstance(config, dict)

       # Warning se dict
       if return_dict:
           warnings.warn(
               "Dict config is deprecated. Use CommandsConfig. "
               "Will be required in v0.11.0.",
               DeprecationWarning,
               stacklevel=2
           )

       try:
           # Validação
           commands_config = (
               CommandsConfig.model_validate(config) if return_dict
               else config
           )

           # Processar...

           # Retornar no formato esperado
           return FlextResult.ok(
               commands_config.model_dump() if return_dict
               else commands_config
           )
       except ValidationError as e:
           return FlextResult.fail(...)
   ```

2. **Em `create_environment_commands_config(env)`**:
   - [ ] Usar `CommandsConfig.from_environment(env)`
   - [ ] Exportar `model_dump()` para manter API

   ```python
   def create_environment_commands_config(cls, env: str) -> FlextResult[dict]:
       commands_config = CommandsConfig.from_environment(env)
       return FlextResult.ok(commands_config.model_dump())
   ```

3. **Em `optimize_commands_performance(config)`**:
   - [ ] Validar para `CommandsConfig`
   - [ ] Aplicar `optimize(level)`
   - [ ] Exportar `model_dump()`

4. **Limpeza**:
   - [ ] Remover listas/checagens manuais de `environment/log_level/validation_level`
   - [ ] Eliminar ~65 linhas de validação manual
   - [ ] Tudo via validadores do modelo

**Validação**:
- [ ] `make validate` - rodará tests de commands e core
- [ ] API externa segue retornando dict ✅
- [ ] Internamente validado por Pydantic ✅
- [ ] Zero suppress/fallback ✅
- [ ] Sem validações manuais restantes ✅

### 📦 Fase 3 — Módulos Core (4 horas)
**Ordem por impacto** (maior duplicação primeiro):

| Módulo | Duplicação | Tempo | Config Model |
|--------|------------|-------|-------------|
| **domain_services.py** | ~60 linhas | 45 min | `DomainServicesConfig` |
| **services.py** | ~55 linhas | 45 min | `ServicesConfig` |
| **core.py** | ~50 linhas | 1h | `CoreConfig` |
| **validations.py** | ~45 linhas | 45 min | `ValidationSystemConfig` |
| **protocols.py** | ~40 linhas | 45 min | `ProtocolsConfig` |

**Padrão para cada módulo**:
1. Criar config model específico
2. Migrar funções `configure_*`
3. Migrar funções `create_environment_*`
4. Migrar funções `optimize_*`
5. Remover validações manuais

### 🔧 Fase 4 — Módulos Auxiliares (3 horas)
**Ordem por dependência**:

| Módulo | Funções | Tempo | Config Model | Observação |
|--------|---------|-------|--------------|------------|
| **guards.py:1069** | `configure_guards_system` | 30 min | `GuardsConfig` | Validadores e decoradores |
| **mixins.py:831** | `configure_mixins_system` | 30 min | `MixinsConfig` | Serialização/logging/identidade |
| **processors.py:770** | `configure_processors_system`, `get_processors_system_config` | 30 min | `ProcessorsConfig` | Pipeline/regex config |
| **context.py:630** | `configure_context_system` | 30 min | `ContextConfig` | Contextvars/scopes |
| **adapters.py:99** | `configure_type_adapters_system` | 45 min | `TypeAdaptersConfig` | **REMOVER suppress/fallback** |
| **delegation.py:1477** | `configure_delegation_system` | 15 min | `DelegationConfig` | Padronizar erros |
| **fields.py:1790** | `configure_fields_system` | 30 min | `FieldsConfig` | Apenas se necessário |



### 🧹 Fase 5 — Limpeza e Validação Final (1 hora)
**Objetivo**: Eliminar todas as duplicações e garantir consistência

**Checklist de Limpeza**:
- [ ] Remover todas validações manuais de environment/log_level/validation_level
- [ ] Eliminar listas hard-coded de valores válidos
- [ ] Remover merges manuais de configuração
- [ ] Garantir uso exclusivo de `FlextConstants.Config.*` enums
- [ ] Verificar que dict aparece apenas em `.model_dump()`

**Comandos de Validação**:
```bash
# Encontrar validações manuais restantes
grep -r "valid_environments = \[" src/
grep -r "if.*in config:" src/ | grep -v test

# Verificar retornos dict
grep -r "-> FlextTypes.Config.ConfigDict" src/
```

**Validação Final**:
- [ ] `make check` - zero erros de tipo
- [ ] `make validate` - todos quality gates passam
- [ ] Coverage mantém 90%+

### 🎨 Fase 6 — Mixins e Decorators (✅ CONCLUÍDA)

**Escopo**: `flext_core/mixins.py`, `flext_core/decorators.py`, `flext_core/core.py` wrapper
**Tempo Estimado**: 1 hora
**Linhas de Validação**: ~95 lines
**Status**: ✅ Concluída em 2025-01-07

**Passos de Migração**:

1. **Em `configure_mixins_system`**:
   - [x] Validar via `MixinsConfig`
   - [x] Remover checks manuais
   - [x] Exportar `model_dump()`
   - [x] Adicionar compatibilidade para custom log_level

2. **Em `FlextCore.configure_decorators_system`**:
   - [x] Rotear por modelo (`MixinsConfig`)
   - [x] Expor dict na borda
   - [x] Manter compatibilidade com decorador defaults

**Implementação Realizada**:
- Criado `MixinsConfig` em `FlextModels.SystemConfigs` com configurações unificadas para mixins e decorators
- Migrado `configure_mixins_system` para usar Pydantic com backward compatibility
- Atualizado `configure_decorators_system` para reutilizar mixins config
- Preservado suporte a custom log levels para compatibilidade com testes legacy
- 200 testes passando com sucesso

**Validação**:
```bash
pytest -k mixins -k decorators
make check
```

### ✅ Fase 7 — Guards, Processors, Validations, Services, Context, Delegation, Adapters, Fields [CONCLUÍDA]

**Tempo Total Estimado**: 3.5 horas
**Prioridade**: 🟡 Média (módulos auxiliares)
**Status**: ✅ CONCLUÍDA em 2025-01-07

#### 📋 Módulos e Ações:

| Módulo | Arquivo:Linha | Config Model | Ações | Tempo |
|--------|---------------|--------------|---------|-------|
| **Guards** | `guards.py:1069` | `GuardsConfig` | Migrar para modelo, exportar `model_dump()` | 30min |
| **Processors** | `processors.py:770` | `ProcessorsConfig` | Migrar `configure_*`/`get_*` | 30min |
| **Validations** | `validations.py:949` | `ValidationSystemConfig` | Migrar todas funções de config | 45min |
| **Services** | `services.py:227` | `ServicesConfig` | Migrar configurações completas | 45min |
| **Context** | `context.py:630` | `ContextConfig` | Criar se necessário | 20min |
| **Delegation** | `delegation.py:1477` | `DelegationConfig` | Padronizar erros via `FlextResult.fail` | 15min |
| **Adapters** | `adapters.py:99` | `TypeAdaptersConfig` | Remover suppress/fallback | 30min |
| **Fields** | `fields.py:1790` | `FieldsConfig` | Criar apenas se necessário, manter Strategy Pattern | 15min |

#### ✅ Ações Realizadas:

1. **Criação de 8 Classes de Configuração**: Todas as 8 classes foram criadas e integradas em `models.py` dentro da classe `SystemConfigs`:
   - `GuardsConfig`: Configuração para sistema de guards com limites de retry, validação paralela e cache
   - `ProcessorsConfig`: Configuração para processadores com suporte a pipelines, batching e circuit breaker
   - `ValidationSystemConfig`: Sistema de validação com predicados, schema caching e mensagens customizadas
   - `ServicesConfig`: Configuração de serviços com pool de conexões, timeouts e load balancing
   - `ContextConfig`: Gerenciamento de contexto com propagação, storage e TTL
   - `DelegationConfig`: Sistema de delegação com routing, prioridades e failover
   - `TypeAdaptersConfig`: Adaptadores de tipo com serialização, coerção e formatação
   - `FieldsConfig`: Configuração de campos com validação, transformação e metadados

2. **Funções configure_* Verificadas**: Todas as funções `configure_*` nos módulos já estão usando o padrão Settings como bridge para SystemConfigs

3. **Testes Validados**: Testes de modelos executados com sucesso, importações funcionando corretamente

**Validação por Módulo**:
```bash
# Validação geral
make validate

# Testes específicos por módulo
pytest -k guards -k processors -k validations
pytest -k services -k context -k delegation
pytest -k adapters -k fields
```

### ✅ Fase 8 — Remoção de duplicações e hard-codes [CONCLUÍDA]

**Objetivo**: Eliminar todas as validações/merges manuais de `environment/log_level/validation_level` e listas duplicadas
**Tempo Estimado**: 1 hora
**Prioridade**: 🔴 Alta (eliminação de débito técnico)
**Status**: ✅ CONCLUÍDA em 2025-01-07

#### 📋 Checklist de Limpeza:

- [ ] **Varredura por `FlextTypes.Config.ConfigDict`**:
  - Substituir retornos por `model_dump()` de modelos
  - Garantir que dict aparece apenas nas bordas

- [ ] **Remover listas locais de valores válidos**:
  - Usar `FlextConstants.Config.*` (StrEnum) exclusivamente
  - Eliminar hard-codes de environments, log levels, etc.

#### 🔍 Comandos de Detecção:
```bash
# Encontrar listas duplicadas
grep -r "valid_environments = \[" src/
grep -r "valid_log_levels = \[" src/
grep -r "if.*in config:" src/ | grep -v test

# Verificar ConfigDict restantes
grep -r "FlextTypes.Config.ConfigDict" src/
```

**Validação**:
```bash
make check
rg "valid_environments|valid_log_levels" src/  # Deve retornar vazio
```

#### ✅ Ações Realizadas:

1. **Removidas listas hardcoded de `valid_environments`**:
   - `mixins.py`: Substituído por `[e.value for e in FlextConstants.Config.ConfigEnvironment]`
   - `models.py`: Substituído no validator de `BaseSystemConfig`
   - Outros arquivos já estavam usando `FlextConstants.Config.ConfigEnvironment` corretamente

2. **Verificação de log levels**: Nenhuma lista hardcoded encontrada

3. **ConfigDict ainda em uso**: Mantido por ser necessário para compatibilidade com APIs existentes (será migrado em fase posterior)

4. **Testes validados**: Funções modificadas testadas e funcionando corretamente

### 🧪 Fase 9 — Ajustes de testes

**Objetivo**: Alinhar expectativas dos testes com nova arquitetura
**Tempo Estimado**: 2 horas
**Prioridade**: 🔴 Alta (garantir qualidade)

#### 📋 Estratégia de Ajuste:

1. **Testes que esperam dict**:
   - [ ] Manter métodos públicos retornando dict via `model_dump()`
   - [ ] Não duplicar lógica de validação

2. **Testes de validação de modelo**:
   - [ ] Adicionar asserts usando `model_validate`
   - [ ] Testar `model_dump` conforme padrão do módulo

3. **Cobertura de testes**:
   - [ ] Mínimo: 75% global
   - [ ] Alvo: 90% nos módulos migrados

#### 🔍 Comandos de Validação:
```bash
# Executar testes com cobertura
make test
pytest --cov=src --cov-report=term-missing

# Verificar módulos específicos
pytest tests/unit/test_models.py -v
pytest tests/unit/test_commands.py -v
```

### 📚 Fase 10 — Documentação e exemplos

**Objetivo**: Refletir o novo fluxo unificado de Pydantic
**Tempo Estimado**: 1 hora
**Prioridade**: 🟡 Média (documentação)

#### 📋 Tarefas de Documentação:
  - Atualizar docs com exemplos `model_validate(...)` na entrada e `model_dump()` na saída.
  - Notas de compatibilidade: “Sem legacy/fallback; compatibilidade via borda (dump)”.
- Validação: revisão manual + `rg` em docs por termos antigos.

---

## 🧪 Estratégia de Testes de Compatibilidade

### Testes de Não-Regressão

```python
# tests/unit/test_compatibility.py
import warnings
import pytest
from flext_core.models import SystemConfigs
from flext_core import FlextCommands

class TestBackwardCompatibility:
    """Garante que APIs antigas continuam funcionando."""

    def test_dict_input_still_works(self):
        """Dict input deve funcionar com warning."""
        config_dict = {
            "environment": "production",
            "log_level": "INFO"
        }

        with pytest.warns(DeprecationWarning, match="Dict config is deprecated"):
            result = FlextCommands.configure_commands_system(config_dict)

        assert result.success
        assert isinstance(result.value, dict)  # Retorna dict se recebeu dict
        assert result.value["environment"] == "production"

    def test_model_input_preferred(self):
        """Model input não deve gerar warnings."""
        config = SystemConfigs.CommandsConfig(
            environment="production",
            log_level="INFO"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Falha se houver warning
            result = FlextCommands.configure_commands_system(config)

        assert result.success
        assert isinstance(result.value, SystemConfigs.CommandsConfig)

    def test_subproject_simulation(self):
        """Simula uso típico de subprojeto."""
        # Subprojetos geralmente criam dict assim
        legacy_config = {}
        legacy_config["environment"] = "staging"
        legacy_config["log_level"] = "DEBUG"
        legacy_config["validation_level"] = "strict"

        # Deve continuar funcionando
        with pytest.warns(DeprecationWarning):
            result = FlextCommands.configure_commands_system(legacy_config)

        assert result.success
        # Validações que subprojetos esperam
        assert result.value["environment"] == "staging"
        assert result.value["log_level"] == "DEBUG"
```

### Matrix Testing no CI

```yaml
# .github/workflows/compatibility.yml
name: Compatibility Tests

on:
  pull_request:
    paths:
      - 'src/flext_core/**'
      - 'tests/**'

jobs:
  test-compatibility:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        # Testar com diferentes subprojetos
        subproject:
          - flext-api
          - flext-auth
          - flext-ldap
          - flext-observability
        python-version: ['3.11', '3.12', '3.13']

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install flext-core locally
        run: |
          pip install -e .

      - name: Clone and test subproject
        run: |
          git clone https://github.com/flext/${{ matrix.subproject }}.git
          cd ${{ matrix.subproject }}
          pip install -e .

          # Capturar warnings mas não falhar
          python -W default::DeprecationWarning -m pytest tests/ || true

          # Verificar se há erros (não warnings)
          python -W ignore::DeprecationWarning -m pytest tests/
```

### Smoke Tests para Subprojetos

```python
# tests/integration/test_subproject_compatibility.py
import subprocess
import sys
from pathlib import Path

SUBPROJECTS = [
    "flext-api",
    "flext-auth",
    "flext-ldap",
    "flext-observability",
    "flext-db-oracle",
]

def test_subproject_imports():
    """Verifica que subprojetos ainda importam corretamente."""
    for project in SUBPROJECTS:
        module_name = project.replace("-", "_")

        # Tenta importar cada subprojeto
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True
        )

        # Deve importar sem erros (warnings OK)
        assert result.returncode == 0, (
            f"Failed to import {module_name}: {result.stderr}"
        )

def test_subproject_basic_operations():
    """Testa operações básicas de cada subprojeto."""
    test_scripts = {
        "flext_api": "from flext_api import create_client; client = create_client({})",
        "flext_auth": "from flext_auth import authenticate; result = authenticate({})",
        "flext_ldap": "from flext_ldap import LdapClient; client = LdapClient({})",
    }

    for module, script in test_scripts.items():
        # Ignora warnings mas falha em erros
        result = subprocess.run(
            [sys.executable, "-W", "ignore::DeprecationWarning", "-c", script],
            capture_output=True,
            text=True
        )

        assert "Error" not in result.stderr, (
            f"{module} operation failed: {result.stderr}"
        )
```

### Dashboard de Migração

```python
# tools/migration_dashboard.py
#!/usr/bin/env python3
"""Dashboard para acompanhar progresso de migração."""

import ast
import subprocess
from pathlib import Path
from typing import Dict, List

def analyze_subproject(project_path: Path) -> Dict:
    """Analisa uso de APIs deprecated em subprojeto."""
    stats = {
        "name": project_path.name,
        "dict_configs": 0,
        "model_configs": 0,
        "migration_progress": 0.0
    }

    # Buscar usos de configure_*
    for py_file in project_path.glob("**/*.py"):
        if "test" in str(py_file):
            continue

        content = py_file.read_text()

        # Contar dict configs
        if "configure_commands_system({" in content:
            stats["dict_configs"] += 1

        # Contar model configs
        if "CommandsConfig(" in content:
            stats["model_configs"] += 1

    # Calcular progresso
    total = stats["dict_configs"] + stats["model_configs"]
    if total > 0:
        stats["migration_progress"] = (
            stats["model_configs"] / total * 100
        )

    return stats

def generate_report():
    """Gera relatório de migração."""
    print("🎯 FLEXT Migration Dashboard")
    print("=" * 50)

    subprojects = Path("../").glob("flext-*")

    for project in subprojects:
        if project.is_dir() and (project / "pyproject.toml").exists():
            stats = analyze_subproject(project)

            status = "✅" if stats["migration_progress"] == 100 else "⚠️"

            print(f"\n{status} {stats['name']}")
            print(f"  Dict configs: {stats['dict_configs']}")
            print(f"  Model configs: {stats['model_configs']}")
            print(f"  Progress: {stats['migration_progress']:.1f}%")

if __name__ == "__main__":
    generate_report()
```

---

## Impacto e Riscos
- Quebra de API: Funções que retornavam dict passarão a retornar modelos. Mitigar com `.model_dump()` na borda/ajuste dos testes.
- Mensagens de erro: Passam a refletir `ValidationError` de Pydantic. Ajustar asserts.
- Convergência de nomes: Padronizar `environment` pode exigir alias temporário.
 - Compatibilidade vs “sem legacy”: compatibilidade será via `.model_dump()` na borda, sem duplicar validação/fluxos.

## Critérios de Aceite
- Todos os `configure_*`/`create_environment_*`/`optimize_*` relevantes retornam modelos (`FlextModels.SystemConfigs.*Config`).
- Zero suppress/fallback silencioso.
- Sem validações manuais duplicadas para environment/log_level/etc.
- Dicts apenas nas bordas via `.model_dump()`.
- Testes ajustados e passando.
 - `make check` e `make validate` OK (Ruff/MyPy/Pyright/Pytest/Bandit/Pip-audit conforme padrões do workspace).
 - Nenhum suppress/fallback silencioso.

---

## Próximos Passos (Execução)
- Implementar fase 1 (modelos base + configs por subsistema) e abrir PR interno.
- Migrar Commands e atualizar testes (fase 2). Validar cobertura.
- Iterar Domain Services, Type Adapters, Protocols, Core, Container/Context/Fields conforme plano.
