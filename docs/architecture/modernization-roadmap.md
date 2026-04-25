# Roadmap Objetivo de Modernização (Python 3.13, OO/MRO, Pydantic v2)


<!-- TOC START -->
- [Escopo](#escopo)
- [Resultado esperado (DoD do programa)](#resultado-esperado-dod-do-programa)
- [Princípios de execução](#princpios-de-execuo)
- [Backlog objetivo por trilha](#backlog-objetivo-por-trilha)
- [Trilha A — Tipagem e hierarquia OO/MRO](#trilha-a-tipagem-e-hierarquia-oomro)
  - [Entregáveis](#entregveis)
  - [Critérios de aceite](#critrios-de-aceite)
- [Trilha B — Pydantic v2 avançado](#trilha-b-pydantic-v2-avanado)
  - [Entregáveis](#entregveis)
  - [Critérios de aceite](#critrios-de-aceite)
- [Trilha C — Simplificação estrutural (SOLID + YAGNI)](#trilha-c-simplificao-estrutural-solid-yagni)
  - [Entregáveis](#entregveis)
  - [Critérios de aceite](#critrios-de-aceite)
- [Plano de execução por ondas (foco em risco)](#plano-de-execuo-por-ondas-foco-em-risco)
  - [Onda 1 — Baixo risco](#onda-1-baixo-risco)
  - [Onda 2 — Médio risco](#onda-2-mdio-risco)
  - [Onda 3 — Alto impacto](#onda-3-alto-impacto)
- [Métricas objetivas de acompanhamento](#mtricas-objetivas-de-acompanhamento)
- [Checklist de fechamento](#checklist-de-fechamento)
- [Nota de contexto](#nota-de-contexto)
<!-- TOC END -->

## Escopo

Padronizar o restante do código para reduzir duplicação e lógica ad-hoc, sem compromisso de retrocompatibilidade, alinhando com a arquitetura em camadas (L0–L3), CQRS e DI já documentadas no projeto.

## Resultado esperado (DoD do programa)

1. **Checks de tipo/hierarquia** centralizados em uma API interna única (sem novos usos diretos de `__mro__`).
1. **Validação Pydantic v2** consolidada (adapters reutilizáveis + menos validators duplicados).
1. **Módulos críticos com menor complexidade** (`runtime`, `checker`, `handlers`, `container`, `models/*`).
1. **Testes de regressão** cobrindo casos de compatibilidade e serialização.

______________________________________________________________________

## Princípios de execução

- **YAGNI:** remover fallback e abstração não exercitados por testes.
- **DRY:** uma única implementação para cada regra de compatibilidade.
- **SOLID:** separar extração de tipo, decisão de compatibilidade e tratamento de erro.
- **Arquitetura atual:** preservar fronteiras L0–L3 (contratos → runtime bridge → domínio/infra → orquestração).

______________________________________________________________________

## Backlog objetivo por trilha

## Trilha A — Tipagem e hierarquia OO/MRO

### Entregáveis

- [ ] Criar módulo interno único de compatibilidade de tipos em `_utilities`:
  - [ ] `is_subclass_safe(candidate, parent)`
  - [ ] `is_instance_or_subclass(value_or_type, expected)`
  - [ ] `is_mapping_type(candidate)` e `is_sequence_type(candidate)`
- [ ] Migrar chamadas duplicadas para o módulo único em:
  - [ ] `runtime.py`
  - [ ] `_utilities/checker.py`
  - [ ] `handlers.py`
  - [ ] `container.py`
- [ ] Proibir novos checks manuais por `__mro__` (exceto introspecção explícita documentada).

### Critérios de aceite

- [ ] 100% dos novos checks usam API central.
- [ ] Zero uso novo de `__mro__` nos arquivos migrados.
- [ ] Testes unitários dedicados para classe, instância, `origin`, `Union` e coleções.

______________________________________________________________________

## Trilha B — Pydantic v2 avançado

### Entregáveis

- [ ] Criar registry/cache de `TypeAdapter` para tipos recorrentes.
- [ ] Extrair normalizações repetidas para funções puras reutilizáveis:
  - [ ] metadata
  - [ ] tags
  - [ ] payload/settings map
- [ ] Unificar validators redundantes em `models/container.py`, `models/settings.py`, `models/cqrs.py`.
- [ ] Introduzir `Annotated[...]` com constraints nativas para remover validação manual onde aplicável.

### Critérios de aceite

- [ ] Redução de `TypeAdapter(...)` inline nos módulos alvo.
- [ ] Redução de validators duplicados com cobertura equivalente.
- [ ] Sem mudança de comportamento observável nos testes existentes.

______________________________________________________________________

## Trilha C — Simplificação estrutural (SOLID + YAGNI)

### Entregáveis

- [ ] Quebrar métodos extensos em responsabilidades pequenas:
  - [ ] extração de tipo
  - [ ] política de compatibilidade
  - [ ] fallback/erro
- [ ] Converter regras ad-hoc em políticas testáveis.
- [ ] Remover branches não cobertos e `try/except` amplo sem ação observável.

### Critérios de aceite

- [ ] Menor complexidade ciclomática nos módulos críticos.
- [ ] Cada política com testes diretos (não apenas integração por fluxo final).

______________________________________________________________________

## Plano de execução por ondas (foco em risco)

### Onda 1 — Baixo risco

- [ ] API central de tipagem + suíte de testes dedicada.
- [ ] Migração inicial de `_utilities/checker.py` e `handlers.py`.

### Onda 2 — Médio risco

- [ ] Migração de `runtime.py` e `container.py`.
- [ ] Introdução do registry de `TypeAdapter` compartilhado.

### Onda 3 — Alto impacto

- [ ] Refactor de `models/*` para reduzir validators redundantes.
- [ ] Limpeza de fallbacks/YAGNI e endurecimento de contratos.

______________________________________________________________________

## Métricas objetivas de acompanhamento

- Contagem de usos de `__mro__` fora de introspecção documentada.
- Contagem de `TypeAdapter(...)` inline por arquivo alvo.
- Contagem de validators com lógica duplicada.
- Branch coverage nos módulos críticos.
- Complexidade ciclomática de `runtime`, `checker`, `handlers`.

______________________________________________________________________

## Checklist de fechamento

- [ ] API de compatibilidade adotada pelos módulos críticos.
- [ ] Validação Pydantic consolidada sem regressão funcional.
- [ ] Testes de regressão e compile pass nos arquivos alterados.
- [ ] Documentação arquitetural atualizada com padrão final.

## Nota de contexto

Para incorporar padrões de `flext-sh/flext`, é necessário acesso local ao repositório (arquivos `AGENTS.md`/docs de arquitetura). Neste ambiente atual, somente `flext-core` está disponível.
