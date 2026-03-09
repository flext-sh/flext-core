# Plano objetivo: padronização e desduplicação de modelos (OO + MRO)

## Contexto aplicado (flext-sh/flext)
Este plano foi ajustado considerando o contexto do ecossistema FLEXT (AGENTS/CLAUDE):
- padronização forte e previsível;
- composição por aliases/namespaces;
- disciplina arquitetural para evitar drift;
- foco em regras automáveis.

Premissa desta versão: **não manter compatibilidade legada**.

---

## 1) Problema atual (direto ao ponto)
No estado atual de `flext-core`, a fachada `FlextModels` cresce com muitas subclasses de reexport sem comportamento próprio, gerando duplicidade de nomes e alto custo cognitivo. Em paralelo, já existe uma foundation sólida (`StrictBoundaryModel`, `FlexibleInternalModel`, `ImmutableValueModel`, `ArbitraryTypesModel`) que pode ser usada como padrão único de intenção.

### Evidências principais no código
- Reexports flat e wrappers em `models.py` (ex.: snapshots/progress/handler/config). 
- Duplicidade semântica explícita (`ProcessingRequest`/`ProcessingConfig`, `CollectionsCategories`/`Categories`, versões flat e aninhadas de handler).
- Base comum clara em `_models/base.py` para consolidar comportamento de validação.
- Containers com API compartilhada em `_models/containers.py`, com espaço para redução de wrappers sem semântica real.

---

## 2) Princípios mandatórios (YAGNI, DRY, SOLID)

1. **YAGNI:** não criar/manter classe pública sem necessidade funcional atual.
2. **DRY:** um conceito público -> um nome canônico.
3. **SOLID (aplicação prática):**
   - S: modelo com uma intenção só;
   - O: extensão por novo modelo de domínio, não por wrapper vazio;
   - L: subclasse não pode afrouxar contrato do pai;
   - I: contratos separados por contexto (CQRS, handler, container, config);
   - D: camadas altas dependem de abstrações da foundation.
4. **MRO curto:** máximo de 3 níveis públicos (Foundation -> Domain -> Facade opcional).
5. **Deletion-first:** primeiro tentar remover; só depois adicionar.

---

## 3) Alvo arquitetural

## 3.1 Camadas
- **Foundation (`_models/base.py`)**: políticas de validação e mutabilidade.
- **Domain (`_models/*.py`)**: modelos reais por contexto.
- **Facade (`models.py`)**: apenas o mínimo canônico para API pública.

## 3.2 Regras de aceitação para novo modelo público
Entrar apenas se cumprir todos:
- adiciona comportamento/validação real;
- não existe equivalente canônico;
- usa base de intenção correta;
- não aumenta MRO desnecessariamente.

Se falhar em 1 item -> rejeitar.

---

## 4) Plano de execução em 4 fases

## Fase A — Inventário objetivo (2 dias)
Entregáveis:
- tabela `public_class | base_chain | has_own_fields | has_own_methods | canonical_candidate`;
- lista de duplicidades por conceito;
- baseline de métricas (contagem de símbolos públicos, profundidade MRO, pares duplicados).

## Fase B — Canonicalização da API pública (3–4 dias)
Ações:
- remover subclasses vazias na fachada;
- remover nomes paralelos para o mesmo conceito;
- manter somente rota canônica por família (handler, collections, config, generic).

Critério de aceite:
- zero “one concept, many names” na API pública.

## Fase C — Normalização por intenção (4–5 dias)
Ações:
- alinhar todos os modelos de `_models/*` às 4 bases de intenção da foundation;
- eliminar `model_config` duplicado quando a base já cobre;
- consolidar validações comuns em pontos únicos.

Critério de aceite:
- 100% dos modelos classificados em uma base de intenção.

## Fase D — Guardrails permanentes (2 dias)
Ações:
- adicionar testes arquiteturais simples (falham ao detectar subclasses públicas vazias novas);
- adicionar regra de review: sem justificativa funcional, sem nova classe pública;
- monitorar métricas em CI.

Critério de aceite:
- regressão de duplicidade detectada automaticamente.

---

## 5) Backlog inicial (ordem recomendada)
1. **Handler:** unificar flat vs aninhado.
2. **Processing:** colapsar `ProcessingRequest`/`ProcessingConfig`.
3. **Collections:** escolher convenção única (`Collections*` ou nomes curtos) e remover duplicatas.
4. **Config errors:** reduzir explosão de classes nominais quando sem ganho funcional.
5. **Generic snapshots/progress/value:** manter apenas reexports canônicos de domínio.

---

## 6) Métricas de sucesso
- **M1:** -40% de subclasses vazias públicas em `FlextModels`.
- **M2:** >=95% dos modelos públicos com MRO <= 3.
- **M3:** 0 duplicidades semânticas ativas na fachada.
- **M4:** 100% dos modelos mapeados para base de intenção.
- **M5:** 1 caminho canônico de import por conceito principal.

---

## 7) Anti-padrões proibidos
- “Wrapper por conveniência” sem comportamento.
- Dois nomes públicos para o mesmo conceito.
- `dict[str, Any]` em fronteira pública quando há modelo.
- `model_config` copiado sem necessidade.
- expansão da fachada como espelho completo de `_models`.

---

## 8) Resultado esperado em 30 dias
Uma API de modelos menor, mais previsível e mais fácil de evoluir:
- menos símbolos públicos;
- herança mais curta e explícita;
- menos ambiguidade de uso;
- governança automática contra regressão.
