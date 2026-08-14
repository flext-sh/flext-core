# Auditoria Pydantic v2 way (2026-03-30)

## Objetivo

Identificar padrões ruins relacionados à migração/uso de Pydantic v2 e mapear aplicações de boas práticas já presentes no projeto.

## Escopo analisado

- `src/flext_core`
- `docs/guides`
- `tests`

Comando base usado na varredura:

```bash
rg -n "\.dict\(|\.json\(|parse_obj\(|parse_raw\(|@validator\b|@root_validator\b|class Config:" src docs tests
```

## Padrões ruins identificados

### 1) Exemplos com API legada (`.dict()`) na documentação

Foram encontrados exemplos de documentação usando `.dict()`, que é padrão de Pydantic v1, em guias de troubleshooting e configuração.

**Ação aplicada neste PR**:
- Substituição por `model_dump()` nos trechos de exemplo para alinhar com Pydantic v2.

## Boas práticas já aplicadas no projeto

### 1) Uso consistente de `ConfigDict`

Modelos usam `model_config: ClassVar[ConfigDict] = ConfigDict(...)` para controle explícito de validação/imutabilidade/serialização.

### 2) Validação com `@field_validator` e `@model_validator`

O projeto aplica validadores de v2 com modo explícito (`before`/`after`) quando necessário.

### 3) Serialização centralizada com `model_dump`

Há utilitário específico para serialização consistente de modelos Pydantic, reduzindo uso ad-hoc e divergências.

### 4) Uso de `BaseSettings` com padrões atuais

Estruturas de configuração em `settings` e modelos de config estão no padrão v2, com foco em validação estrita e previsibilidade.

## Recomendações práticas (curto prazo)

1. Manter check automático em CI com busca por APIs v1 proibidas em `src/` e `docs/`.
2. Criar lint/doc-check para bloquear novos exemplos com `.dict()`, `.parse_obj()`, `@validator`, `class Config:`.
3. Consolidar uma seção "migração v1 -> v2" nos guias mais acessados para reduzir regressão documental.

## Resultado

- Código de runtime segue majoritariamente aderente ao Pydantic v2.
- Principal risco atual era de **drift de documentação** (exemplos legados), corrigido nos pontos encontrados nesta entrega.
