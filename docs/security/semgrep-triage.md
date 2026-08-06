# Triagem Semgrep — flext-sh/flext-core

Gerado do dump da plataforma Semgrep (deployment `datacosmos`, 2026-08-06).

Bead de rastreio: `mro-p57t.5`

## Resumo

**6 findings** — high 1, medium 5, low 0
Confiança: high 4, medium 0, low 2

| regra | achados |
|---|---|
| `package_managers.dependabot.dependabot-missing-cooldown.dependabot-missing-cooldown` | 3 |
| `python.lang.compatibility.python37.python37-compatibility-importlib2` | 1 |
| `package_managers.uv.uv-missing-dependency-cooldown.uv-missing-dependency-cooldown` | 1 |
| `python.lang.security.audit.non-literal-import.non-literal-import` | 1 |

## Findings

Coluna **Decisão** a preencher: `corrigir` / `falso-positivo` / `risco-aceito`.

| # | sev | conf | regra | arquivo | linha | Decisão |
|---|---|---|---|---|---|---|
| 1 | high | low | `python37-compatibility-importlib2` | `src/flext_core/_constants/_enforcement_data/__init__.py` | 9 | |
| 2 | medium | high | `dependabot-missing-cooldown` | `.github/dependabot.yml` | 4 | |
| 3 | medium | high | `dependabot-missing-cooldown` | `.github/dependabot.yml` | 11 | |
| 4 | medium | high | `dependabot-missing-cooldown` | `.github/dependabot.yml` | 18 | |
| 5 | medium | high | `uv-missing-dependency-cooldown` | `pyproject.toml` | 632 | |
| 6 | medium | low | `non-literal-import` | `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_01.py` | 31 | |

## Como triar

1. Abrir `arquivo:linha` e seguir o fluxo até o sink.
2. Classificar: **corrigir** (entrada externa alcança o sink), **falso-positivo** (registrar via `nosemgrep` ou `.semgrepignore` com justificativa), **risco-aceito** (com prazo de revisão).
3. Priorizar findings high com confidence=high.

Dados brutos: `~/semgrep-violations/by-repo/flext-sh__flext-core.json`

