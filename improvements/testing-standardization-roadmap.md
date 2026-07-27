# Roadmap objetivo de padronização de testes (restante do projeto)

<!-- TOC START -->
- [Contexto consolidado (flext)](#contexto-consolidado-flext)
- [Escopo desta fase](#escopo-desta-fase)
- [Entregáveis obrigatórios (curto prazo)](#entregveis-obrigatrios-curto-prazo)
- [Plano de execução em 4 PRs (objetivo e incremental)](#plano-de-execuo-em-4-prs-objetivo-e-incremental)
- [Regras de design (enxutas)](#regras-de-design-enxutas)
  - [DRY](#dry)
  - [SOLID](#solid)
  - [YAGNI](#yagni)
- [Métricas objetivas de sucesso](#mtricas-objetivas-de-sucesso)
- [Validação padrão por PR](#validao-padro-por-pr)
- [Riscos e contenção](#riscos-e-conteno)
- [Próxima ação imediata](#prxima-ao-imediata)
<!-- TOC END -->

**Meta única**: reduzir duplicação e aumentar confiabilidade dos testes sem overengineering, com entregas pequenas e verificáveis.

## Contexto consolidado (flext)

- O `AGENTS.md` do repositório `flext-sh/flext` define política de **ponteiro único**: regras canônicas ficam em `CLAUDE.md` (sem duplicar governança em múltiplos arquivos).
- O `CLAUDE.md` reforça: **DRY/SOLID obrigatório**, proibição de duplicação, sem atalhos e com validação factual.
- Este plano segue esse mesmo modelo: poucas regras, execução objetiva, evidência por comando.

## Escopo desta fase

- **Dentro**: testes unitários de utilidades e contratos reutilizáveis.
- **Fora (agora)**: integração/e2e e reestruturação ampla de domínio.

## Entregáveis obrigatórios (curto prazo)

1. `tests/unit/contracts/cache_contract.py`
1. `tests/unit/contracts/generators_contract.py`
1. `tests/unit/contracts/validation_contract.py`
1. `docs/standards/testing-refactor-checklist.md`

## Plano de execução em 4 PRs (objetivo e incremental)

| PR   | Objetivo                    | Mudanças mínimas                                                                                     | Critério de pronto (DoD)                                                             |
| ---- | --------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| PR-1 | Consolidar cache tests      | Criar `cache_contract.py` + migrar duplicações de `test_utilities.py` e `test_coverage_utilities.py` | `pytest -q tests/unit/test_utilities.py tests/unit/test_coverage_utilities.py` verde |
| PR-2 | Consolidar generators tests | Criar `generators_contract.py` + unificar prefixo/tamanho/unicidade                                  | mesmos testes verdes + sem duplicação de cenários em 2+ arquivos                     |
| PR-3 | Consolidar validation tests | Criar `validation_contract.py` + migrar pipeline/type-check cases                                    | testes unitários afetados verdes                                                     |
| PR-4 | Governança de revisão       | Criar checklist de PR e regra de adoção de contratos                                                 | checklist publicado + exemplos de uso em 1 PR                                        |

## Regras de design (enxutas)

### DRY

- Cada família de comportamento terá **1 fonte de verdade** de cenários (contrato).
- Testes concretos só estendem casos locais quando houver necessidade real.

### SOLID

- Contratos pequenos por domínio (cache/generators/validation).
- Herança por MRO apenas para composição de invariantes, sem “mega classe base”.

### YAGNI

- Não criar framework genérico de testes.
- Só extrair contrato quando houver duplicação real em mais de um arquivo.

## Métricas objetivas de sucesso

- Redução de duplicação textual nos testes-alvo: **>= 40%**.
- Novos contratos reutilizáveis além de texto: **>= 3**.
- Módulos com property-based tests (Hypothesis) além de texto: **>= 2**.
- Tempo de execução da suíte afetada: sem regressão superior a **15%**.

## Validação padrão por PR

Rodar sempre (escopo mínimo):

```bash
PYTHONPATH=src pytest -q tests/unit/test_utilities.py tests/unit/test_coverage_utilities.py tests/unit/test_utilities_text_full_coverage.py
```

Quando houver novos contratos, incluir também:

```bash
python -m py_compile tests/unit/contracts/*.py
```

## Riscos e contenção

- **Risco**: abstração demais.
  - **Ação**: limitar PR a um domínio por vez, com diff pequeno.
- **Risco**: flakiness de property-based.
  - **Ação**: configurar `@settings` com limites explícitos quando necessário.
- **Risco**: divergência entre contrato e comportamento real.
  - **Ação**: alinhar expectativa ao código-fonte antes de endurecer invariantes.

## Próxima ação imediata

Executar **PR-1 (cache_contract)** com foco exclusivo em deduplicação real e evidência de ganho no diff.
