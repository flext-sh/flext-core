# Triagem SonarCloud — flext-sh/flext-core

Gerado do dump da plataforma SonarCloud (2026-08-06).

Bead de rastreio: `mro-2wjm.2`

## Resumo

**133 issues** — BLOCKER 5, CRITICAL 24, MAJOR 27, MINOR 77
Tipos: VULNERABILITY 4, BUG 6, CODE_SMELL 123

| regra | issues |
|---|---|
| `python:S116` | 66 |
| `python:S3776` | 15 |
| `python:S1192` | 8 |
| `python:S108` | 6 |
| `python:S930` | 5 |
| `python:S5713` | 4 |
| `python:S3358` | 3 |
| `python:S6796` | 3 |

## Issues

Coluna **Decisão**: `corrigir` / `falso-positivo` / `risco-aceito`.

| # | sev | tipo | regra | componente | linha | Decisão |
|---|---|---|---|---|---|---|
| 1 | BLOCKER | BUG | `python:S930` | `src/flext_core/__version__.py` | 35 | |
| 2 | BLOCKER | BUG | `python:S930` | `src/flext_core/_result/base.py` | 87 | |
| 3 | BLOCKER | BUG | `python:S930` | `src/flext_core/_result/base.py` | 88 | |
| 4 | BLOCKER | BUG | `python:S930` | `src/flext_core/_result/base.py` | 89 | |
| 5 | BLOCKER | BUG | `python:S930` | `src/flext_core/_result/base.py` | 90 | |
| 6 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_01.py` | 24 | |
| 7 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_03.py` | 33 | |
| 8 | CRITICAL | CODE_SMELL | `python:S1186` | `src/flext_core/_result/behavior.py` | 32 | |
| 9 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_result/composition.py` | 64 | |
| 10 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_result/construction.py` | 119 | |
| 11 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_result/transforms.py` | 39 | |
| 12 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_result/transforms.py` | 76 | |
| 13 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_settings.py` | 95 | |
| 14 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_beartype/attr_visitor.py` | 106 | |
| 15 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_beartype/deprecated_visitor.py` | 39 | |
| 16 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_beartype/field_visitor.py` | 45 | |
| 17 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_beartype/import_visitor.py` | 61 | |
| 18 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_beartype/module_visitor.py` | 52 | |
| 19 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_checker_parts/checker_part_03.py` | 112 | |
| 20 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_enforcement_collect_parts/enforcement_collect_part_01.py` | 98 | |
| 21 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/_enforcement_parts/enforcement_part_02.py` | 54 | |
| 22 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_utilities/_mapper_access_parts/mapper_access_part_02.py` | 48 | |
| 23 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/_utilities/dispatcher_execute.py` | 27 | |
| 24 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/domain.py` | 66 | |
| 25 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/guards.py` | 98 | |
| 26 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/_utilities/mapper.py` | 102 | |
| 27 | CRITICAL | CODE_SMELL | `python:S1192` | `src/flext_core/dispatcher.py` | 188 | |
| 28 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/registry.py` | 80 | |
| 29 | CRITICAL | CODE_SMELL | `python:S3776` | `src/flext_core/registry.py` | 431 | |
| 30 | MAJOR | VULNERABILITY | `githubactions:S8264` | `.github/workflows/docs.yml` | 18 | |
| 31 | MAJOR | VULNERABILITY | `githubactions:S8233` | `.github/workflows/docs.yml` | 19 | |
| 32 | MAJOR | VULNERABILITY | `githubactions:S8233` | `.github/workflows/docs.yml` | 20 | |
| 33 | MAJOR | VULNERABILITY | `text:S8565` | `pyproject.toml` | - | |
| 34 | MAJOR | CODE_SMELL | `python:S8786` | `src/flext_core/_constants/regex.py` | 62 | |
| 35 | MAJOR | CODE_SMELL | `python:S3358` | `src/flext_core/_decorators/_base.py` | 70 | |
| 36 | MAJOR | CODE_SMELL | `python:S6796` | `src/flext_core/_exceptions/_factories_parts/flextexceptionsfactories_part_01.py` | 72 | |
| 37 | MAJOR | CODE_SMELL | `python:S5890` | `src/flext_core/_lazy_parts/flextlazy_part_01.py` | 76 | |
| 38 | MAJOR | CODE_SMELL | `python:S8963` | `src/flext_core/_models/_base_parts/flextmodelsbase_part_03.py` | 115 | |
| 39 | MAJOR | CODE_SMELL | `python:S8963` | `src/flext_core/_models/domain_event.py` | 29 | |
| 40 | MAJOR | CODE_SMELL | `python:S8963` | `src/flext_core/_models/entity.py` | 37 | |
| 41 | MAJOR | CODE_SMELL | `python:S6794` | `src/flext_core/_models/pydantic.py` | 115 | |
| 42 | MAJOR | CODE_SMELL | `python:S6794` | `src/flext_core/_models/pydantic.py` | 116 | |
| 43 | MAJOR | CODE_SMELL | `python:S6796` | `src/flext_core/_protocols/result.py` | 59 | |
| 44 | MAJOR | CODE_SMELL | `python:S6796` | `src/flext_core/_protocols/result.py` | 72 | |
| 45 | MAJOR | CODE_SMELL | `python:S5890` | `src/flext_core/_result/base.py` | 39 | |
| 46 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/_beartype/_alias_visitor.py` | 101 | |
| 47 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_01.py` | 74 | |
| 48 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_01.py` | 107 | |
| 49 | MAJOR | CODE_SMELL | `python:S3358` | `src/flext_core/_utilities/_beartype/_class_visitor_parts/class_visitor_part_03.py` | 69 | |
| 50 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/_beartype/deprecated_visitor.py` | 189 | |
| 51 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/_beartype/method_visitor.py` | 109 | |
| 52 | MAJOR | BUG | `python:S3699` | `src/flext_core/_utilities/_logging_config_parts/logging_config_part_01.py` | 130 | |
| 53 | MAJOR | CODE_SMELL | `python:S5864` | `src/flext_core/_utilities/_parser_targets_parts/parser_targets_part_01.py` | 92 | |
| 54 | MAJOR | CODE_SMELL | `python:S5864` | `src/flext_core/_utilities/_parser_targets_parts/parser_targets_part_01.py` | 108 | |
| 55 | MAJOR | CODE_SMELL | `python:S108` | `src/flext_core/_utilities/guards.py` | 139 | |
| 56 | MAJOR | CODE_SMELL | `python:S3358` | `src/flext_core/context.py` | 188 | |
| 57 | MINOR | CODE_SMELL | `python:S7504` | `conftest.py` | 33 | |
| 58 | MINOR | CODE_SMELL | `python:S117` | `conftest.py` | 111 | |
| 59 | MINOR | CODE_SMELL | `python:S7500` | `examples/ex_01_flext_result_helpers.py` | 126 | |
| 60 | MINOR | CODE_SMELL | `python:S6353` | `src/flext_core/_constants/regex.py` | 32 | |
| 61 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_exceptions/_base_parts/flextexceptionsbase_part_03.py` | 160 | |
| 62 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/cqrs.py` | 110 | |
| 63 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/domain_event.py` | 58 | |
| 64 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 115 | |
| 65 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 116 | |
| 66 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 118 | |
| 67 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 122 | |
| 68 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 123 | |
| 69 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 131 | |
| 70 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 132 | |
| 71 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 133 | |
| 72 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 134 | |
| 73 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 137 | |
| 74 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 138 | |
| 75 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 141 | |
| 76 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 142 | |
| 77 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 143 | |
| 78 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 146 | |
| 79 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 149 | |
| 80 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 150 | |
| 81 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 151 | |
| 82 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 154 | |
| 83 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 157 | |
| 84 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 160 | |
| 85 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_models/pydantic.py` | 161 | |
| 86 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_runtime/_base.py` | 25 | |
| 87 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 126 | |
| 88 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 127 | |
| 89 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 128 | |
| 90 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 129 | |
| 91 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 130 | |
| 92 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 131 | |
| 93 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 140 | |
| 94 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 141 | |
| 95 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 142 | |
| 96 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 143 | |
| 97 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 144 | |
| 98 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 145 | |
| 99 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 148 | |
| 100 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 149 | |
| 101 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 150 | |
| 102 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 151 | |
| 103 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 152 | |
| 104 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 153 | |
| 105 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 154 | |
| 106 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 155 | |
| 107 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 156 | |
| 108 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 157 | |
| 109 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 158 | |
| 110 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 159 | |
| 111 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/pydantic.py` | 160 | |
| 112 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/services.py` | 138 | |
| 113 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_typings/services.py` | 139 | |
| 114 | MINOR | CODE_SMELL | `python:S5713` | `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_01.py` | 32 | |
| 115 | MINOR | CODE_SMELL | `python:S7500` | `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_02.py` | 132 | |
| 116 | MINOR | CODE_SMELL | `python:S5713` | `src/flext_core/_utilities/_beartype/_helpers_parts/helpers_part_03.py` | 92 | |
| 117 | MINOR | CODE_SMELL | `python:S5685` | `src/flext_core/_utilities/_beartype/import_visitor.py` | 114 | |
| 118 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/collection_merge.py` | 109 | |
| 119 | MINOR | CODE_SMELL | `python:S5685` | `src/flext_core/_utilities/discovery.py` | 49 | |
| 120 | MINOR | CODE_SMELL | `python:S5713` | `src/flext_core/_utilities/dispatcher_execute.py` | 93 | |
| 121 | MINOR | CODE_SMELL | `python:S5713` | `src/flext_core/_utilities/project_metadata.py` | 101 | |
| 122 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 48 | |
| 123 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 49 | |
| 124 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 50 | |
| 125 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 61 | |
| 126 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 62 | |
| 127 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 63 | |
| 128 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 64 | |
| 129 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 65 | |
| 130 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 66 | |
| 131 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 68 | |
| 132 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 69 | |
| 133 | MINOR | CODE_SMELL | `python:S116` | `src/flext_core/_utilities/pydantic.py` | 70 | |

## Como triar

1. **BLOCKER e CRITICAL primeiro**, e todo VULNERABILITY independente de severidade.
2. Classificar: **corrigir**, **falso-positivo** (marcar na plataforma SonarCloud com justificativa), **risco-aceito** (com prazo).
3. CODE_SMELL em volume alto sugere padrão — corrigir a causa raiz, não issue a issue.

Dados brutos: `~/sonarqube-violations/by-repo/flext-sh__flext-core.json`

