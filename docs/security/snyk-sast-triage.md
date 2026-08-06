# Triagem Snyk Code (SAST) — flext-sh/flext-core

Gerado do scan Snyk da org Datacosmos (dump 2026-08-06).

**9 achados** — critical 0, high 0, medium 1, low 8

| categoria | achados |
|---|---|
| Use of Hardcoded Passwords | 5 |
| Use of Hardcoded Credentials | 4 |

## Achados

Coluna **Decisão**: `corrigir` / `falso-positivo` / `risco-aceito`.

| # | sev | categoria | arquivo | linha | CWE | Decisão |
|---|---|---|---|---|---|---|
| 1 | medium | Use of Hardcoded Passwords | `examples/_models/output.py` | 41 | - | |
| 2 | low | Use of Hardcoded Credentials | `tests/_models/_mixins/test_data_identity.py` | 43 | - | |
| 3 | low | Use of Hardcoded Credentials | `tests/_models/_mixins/test_data_values.py` | 45 | - | |
| 4 | low | Use of Hardcoded Credentials | `tests/integration/test_service.py` | 73 | - | |
| 5 | low | Use of Hardcoded Credentials | `tests/integration/test_service.py` | 147 | - | |
| 6 | low | Use of Hardcoded Passwords | `tests/unit/test_result_factory_dip.py` | 169 | - | |
| 7 | low | Use of Hardcoded Passwords | `tests/unit/test_result_factory_dip.py` | 215 | - | |
| 8 | low | Use of Hardcoded Passwords | `tests/unit/test_result_factory_dip.py` | 226 | - | |
| 9 | low | Use of Hardcoded Passwords | `tests/unit/test_result_factory_dip.py` | 229 | - | |

## Como triar

1. Abrir `arquivo:linha` e seguir o fluxo de dados até o sink.
2. Classificar: **corrigir** (entrada externa alcança o sink sem sanitização), **falso-positivo** (credencial de fixture, path de constante — registrar em `.snyk` com justificativa), **risco-aceito** (com prazo de revisão).

Dados brutos: `~/snyk-violations/sast/flext-sh__flext-core.sast.json`

