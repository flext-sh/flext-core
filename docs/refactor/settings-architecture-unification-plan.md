# Settings Architecture Unification Plan — flext-core

Status: Draft (incremental, sob padrões flext-core)
Autoridade: flext-core 0.9.0 + FLEXT_REFACTORING_PROMPT.md + CLAUDE.md + README.md

## Objetivo

- Unificar a arquitetura de Settings usando Pydantic v2.11 BaseSettings (SettingsConfigDict) com camadas (constants → env → arquivo → CLI) de forma padronizada.
- Separar clara e tecnicamente “Config” (modelos Pydantic de sistema) de “Settings” (fonte de verdade vinda de ambiente/arquivo), com ponte entre eles.
- Definir padrões para Settings estáticas (requerem restart) e dinâmicas (atualizáveis em runtime) e o fluxo de recarga seguro.
- Manter compatibilidade de APIs (dicts na borda via `model_dump()`), sem fallback/legacy.
- Tornar o design extensível para outras bibliotecas do ecossistema (subclasses e loaders compartilhados).

## Escopo

- flext-core/src/flext_core/config.py (atual BaseSettings/Config avançado)
- flext-core/src/flext_core/models.py (SystemConfigs Pydantic de domínio)
- Módulos com `configure_*` atuais (commands, domain_services, protocols, core, validations, services, mixins, guards, processors, context, delegation, adapters, fields) que hoje montam dicts.

## Regras de Ouro (flext-core)

- Clean Architecture: Foundation → Domain → Application → Infrastructure → Support.
- FlextResult em todas as operações públicas de negócio; sem fallback silencioso.
- StrEnum/valores válidos somente de `FlextConstants.Config.*`.
- Pydantic v2.11: BaseSettings/SettingsConfigDict para entrada de ambiente + BaseModel/ConfigDict para validação de sistema.
- Compatibilidade: dicts apenas na borda via `model_dump()`. Internamente trafegar modelos.

### Constants x Settings

- Constants (FlextConstants): defaults imutáveis e StrEnums canônicos. Nunca ler ambiente, nem conter lógica de merge.
- Settings (BaseSettings): entrada de configuração (env/arquivo/CLI). Podem referenciar `FlextConstants` para defaults. Não aplicar regras de negócio — apenas coleta.
- SystemConfigs (BaseModel): validação e normalização final (regras por subsistema), usando StrEnums de `FlextConstants` e validadores Pydantic.
- Precedência: `FlextConstants` < `Settings` < overrides < `SystemConfigs` → borda.

## Inventário e Pontos de Ajuste

- config.py: Já usa BaseSettings em `FlextConfig` com validações/serializadores. Precisa separar papeis: Settings (fonte) vs Config (modelo de sistema). Reaproveitar validações comuns via base.
- models.py: Terá `FlextModels.SystemConfigs` (ver plano Pydantic) como destino final de validação de parâmetros. Settings devem carregar valores e alimentar esses modelos.
- Módulos `configure_*`: hoje montam/validam dicts. Passar a validar com `SystemConfigs` e usar Settings somente para entrada/env.

### Auditoria de Subclasses Cross-Domain (migração para subprojetos)

- Problema: Classes/Settings/Configs com semântica de domínio específico podem estar em flext-core mas pertencer a subprojetos (ex.: DBT/LDIF/GRPC específicos).
- Ação de auditoria:
  - Mapear classes e constants potencialmente cross-domain (ex.: referências a `FlextConstants.DBT`, `FlextConstants.LDIF`, `FlextConstants.GRPC`).
  - Comandos de apoio:
    - `rg -n "FlextConstants\.(DBT|LDIF|GRPC|OIC|WMS)" -S` no workspace
    - `rg -n "class\s+\w+Settings\(|class\s+\w+Config\(" flext-core/src -S`
  - Para cada ocorrência:
    - Verificar uso fora de flext-core (ex.: flext-dbt-*, flext-grpc, flext-ldif). Se exclusivo de um subprojeto → migrar para o subprojeto.
    - Se compartilhado por múltiplos projetos → manter em flext-core, mas reduzir escopo ao mínimo necessário e expor pontos de extensão.
- Migração:
  - Criar a classe no subprojeto (ex.: `flext_dbt.settings.DbtSettings`, `flext_dbt.configs.DbtConfig`).
  - Em flext-core, manter façade leve (alias/deprecation) por 1 ciclo de release: documentar caminho novo e prazo de remoção.
  - Atualizar imports nos subprojetos para o novo local.
- Critérios:
  - Nenhuma classe de domínio específico permanece em flext-core sem uso cross-projeto.
  - Façades em flext-core sem lógica: apenas alias e deprecation warnings.

## Arquitetura Proposta

### Camadas

1) Defaults (FlextConstants) → 2) Settings (BaseSettings) → 3) SystemConfigs (BaseModel) → 4) Borda (dict via `model_dump()`).

- Defaults: todos os valores canônicos e StrEnums em `FlextConstants`.
- Settings: classes BaseSettings com `SettingsConfigDict(env_prefix="FLEXT_", env_file=".env")` carregando de env/arquivo/CLI.
- SystemConfigs: modelos Pydantic (não-Settings) com validadores e regras por subsistema, herdando de `BaseSystemConfig`.
- Borda: exporta dict apenas para consumidores externos/legados.

### Unificação Settings

- Nova hierarquia em `flext_core/config.py` ou `flext_core/settings.py` (manter o arquivo atual por compatibilidade):
  - `class FlextSettingsBase(BaseSettings)`
    - `model_config = SettingsConfigDict(env_prefix="FLEXT_", env_file=".env", env_file_encoding="utf-8", case_sensitive=False)`
    - Métodos utilitários: `from_sources(...)` para combinar constants/env/arquivo/CLI.
  - `class FlextStaticSettings(FlextSettingsBase)`: `model_config = SettingsConfigDict(frozen=True, extra="forbid")`
  - `class FlextDynamicSettings(FlextSettingsBase)`: `model_config = SettingsConfigDict(frozen=False, extra="forbid")`
  - Mecanismo de marcação de dinamismo: `Field(..., json_schema_extra={"dynamic": True})` para campos atualizáveis; omisso/False para estáticos.

- Subclasses por subsistema (ex.: `CommandsSettings`, `DomainServicesSettings`, `ProtocolsSettings`, etc.) com mesmos campos dos respectivos `*Config`, mas orientadas a entrada/env.

### Ponte Settings → SystemConfigs

- Para cada subsistema:
  - `CommandsSettings.to_config() -> CommandsConfig`
    - Executa `CommandsConfig.model_validate(self.model_dump())`
  - Factory composta: `CommandsSettings.from_sources(**kwargs).to_config()`
  - Para uso direto sem Settings: ainda suportado: `CommandsConfig.model_validate(dict)`.

### Dinâmicas vs Estáticas

- Campos dinâmicos: `json_schema_extra={"dynamic": True}` e `FlextDynamicSettings`.
- Campos estáticos: padrão (sem marcação) em `FlextStaticSettings` ou `frozen=True`.
- Helper:
  - `is_dynamic(field_name) -> bool`
  - `diff_requires_restart(old: SystemConfig, new: SystemConfig) -> list[str]` (lista de campos estáticos alterados).
  - `apply_runtime_update(current: SystemConfig, patch: dict) -> FlextResult[SystemConfig]` (falha se patch altera campo estático; caso contrário, `model_copy(update=...)`).

### Loader com Prioridade (Layered)

- Precedência (do menor → maior): Constants → Settings (env/arquivo) → CLI/Override explícito.
- `FlextSettingsLoader` (composição estática):
  - `load_defaults()` de `FlextConstants`.
  - `load_env_file()` do BaseSettings.
  - `apply_overrides(dict)` (ex.: CLI/arquivo JSON/TOML validado fora).
  - Retorna `*Settings` e `*Config` via `.to_config()`.

### Registry de Settings

- `FlextSettingsRegistry`
  - Guarda instâncias atuais por subsistema (`commands`, `protocols`, etc.) em memória.
  - APIs: `get_config(name) -> FlextResult[SystemConfig]`, `update_runtime(name, patch) -> FlextResult[SystemConfig]`, `reload(name) -> FlextResult[SystemConfig]`.
  - Internamente:
    - Verifica dinamismo por campo e usa `apply_runtime_update`.
    - Em caso de estático modificado: retorna lista de campos que exigem restart.

### Extensão por Outras Bibliotecas

- Outras libs criam `MyLibSettings(FlextSettingsBase)` e `MyLibConfig(BaseSystemConfig)`; implementam `.to_config()`.
- Reaproveitam `FlextSettingsLoader` e `FlextSettingsRegistry`.
- Padrão de env prefix customizável: `env_prefix` override (ex.: `MYLIB_`), mantendo compatibilidade do workspace.

## Plano de Mudança (Incremental)

Fase 0 — Baseline
- make check; make test (registrar baseline).

Fase 1 — Introduzir Settings base e Loader
- Em `flext_core/config.py`: adicionar `FlextSettingsBase`, `FlextStaticSettings`, `FlextDynamicSettings`, `FlextSettingsLoader`, helpers `is_dynamic`, `apply_runtime_update`, `diff_requires_restart`.
- Exportar via `flext_core/__init__.py`.
- Validação: make check; pytest -k config.

Fase 2 — Subclasses de Settings por subsistema e ponte com SystemConfigs
- Para Commands/DomainServices/Protocols/Core/Validations/Services/Mixins/Guards/Processors/Context/Delegation/Adapters/Fields:
  - Adicionar `*Settings` (dinâmicas/estáticas conforme o caso) com `to_config()` apontando para `*Config` em `FlextModels.SystemConfigs`.
- Validação: make check; pytest -k models -k config -k commands.

Fase 3 — Adotar Settings Loader nas fachadas
- Nos `configure_*` de cada módulo: aceitar `dict` (borda); converter para `*Settings` via `from_sources(...)`; gerar `*Config` com `.to_config()`; retornar `model_dump()` para compatibilidade.
- Remover validações manuais; remover listas hard-coded.
- Validação: make validate.

Fase 4 — Dinâmicas vs Estáticas e Registry
- Marcar campos dinamicamente atualizáveis (`json_schema_extra={"dynamic": True}`) nos Settings; mapear para `*Config`.
- Implementar `FlextSettingsRegistry` e métodos `update_runtime`/`reload` por subsistema.
- Testes de mutação dinâmica com `apply_runtime_update` e rejeição de alterações estáticas.
- Validação: pytest -k runtime -k reload.

Fase 5 — Documentação e Extensão
- Documentar padrões em docs/refactor e docs/guides/configuration.
- Exemplo de extensão em outra biblioteca (subclass de `FlextSettingsBase` + `to_config()`).
- Validação: revisão + exemplos rodando em tests de integração.

## Critérios de Aceite

- Nenhum `configure_*` mantém validação manual de `environment/log_level/validation_level`.
- Padrão de camadas aplicado: constants → settings → configs → borda.
- Atualização dinâmica respeita marcação de campos; alterações estáticas reportam necessidade de restart.
- make check e make validate OK; testes cobrindo carregamento, validação, atualização dinâmica, compatibilidade de dicts.

## Notas de Compatibilidade

- APIs públicas que retornam dict continuam retornando dict (via `model_dump()`), sem caminhos de validação duplicados.
- Para consumo interno, preferir `*Settings/*.Config` em vez de dicts crus.
