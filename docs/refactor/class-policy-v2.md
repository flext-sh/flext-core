# Class Policy v2

## Declaracao de Regras v2

- Arquivos: `class-policy-v2.yml` e `class-policy-v2.schema.json`
- Campos principais: `project_kind`, `facade_family`, `module_family`, `source_symbol`, `target_facade_class`, `target_namespace_path`, `expected_base_chain`, `forbidden_targets`, `confidence`, `pre_checks`, `post_checks`, `rewrite_scope`
- `confidence`: escala `0.0` a `1.0`
- `pre_checks` e `post_checks`: lista de objetos com `type` e `params`

## Matriz por Familia

- `models`: composicao sob `m`, bloqueado para `u/d/dispatcher`
- `_utilities`: consolidacao sob `u`, mudanca de assinatura exige validacao
- `_dispatcher`: consolidacao sob `FlextDispatcher`, propagacao cross-project obrigatoria
- `_decorators`: consolidacao em namespace de decorators, contrato callable preservado
- `_runtime`: whitelist obrigatoria, default deny

## Exemplos por Familia

- Cada regra define origem, destino, bloqueios e escopo de reescrita
- Estrutura minima por regra:
  - `project_kind`
  - `facade_family`
  - `module_family`
  - `source_symbol`
  - `target_facade_class`
  - `confidence`
  - `pre_checks`
  - `post_checks`

## Casos de Excecao

- `_runtime` exige entrada explicita em `runtime_whitelist`
- Regras fora da whitelist em `_runtime` devem falhar no pre-check
- Excecoes operacionais devem ser rastreadas em artefatos de rollout

## Nao Simplificar `_private`

- `models`, `_utilities`, `_dispatcher`, `_decorators`, `_runtime` possuem politicas distintas
- Misturar familias quebra contratos de namespace e MRO
- O pipeline deve bloquear transformacoes fora da politica declarada

## MRO por Projeto

- `expected_base_chain` deriva do classificador de projeto + classes reais
- Exemplo de cadeia valida:
  - `AlgarOudMigModels(FlextLdapModels, FlextCliModels)`
- Validacao ocorre no post-check e deve bloquear mismatch de ordem
