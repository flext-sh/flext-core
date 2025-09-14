# Cross-Domain Audit — Checklist

Status: Draft

Este checklist orienta a auditoria de classes/constantes do flext-core que podem pertencer a subprojetos (DBT/LDIF/GRPC/OIC/WMS/etc.). O objetivo é reduzir acoplamento e manter flext-core agnóstico de domínio específico.

## Como executar

1) Mapear referências a domínios específicos:

```
rg -n "FlextConstants\.(DBT|LDIF|GRPC|OIC|WMS)" -S
rg -n "class\s+\w+Settings\(|class\s+\w+Config\(" flext-core/src -S
```

2) Classificar cada ocorrência:
- Uso exclusivo em um subprojeto (ex.: apenas flext-dbt-*) → migrar para o subprojeto
- Uso em múltiplos projetos → manter em flext-core, mas com escopo mínimo e pontos de extensão documentados

3) Migração
- Criar classes Settings/Config no subprojeto (ex.: `flext_dbt.settings.DbtSettings`, `flext_dbt.configs.DbtConfig`)
- Em flext-core, manter façade leve (alias / DeprecationWarning) por 1 release
- Atualizar imports nos subprojetos

4) Critérios de aceite
- Nenhuma classe específica de domínio permanece em flext-core sem uso cross-projeto
- Facades em flext-core não contêm lógica além de alias e deprecação

5) PRs sugeridos
- [ ] flext-dbt: mover DbtSettings/DbtConfig
- [ ] flext-ldap: migrar settings/configs específicas de LDAP (se aplicável)
- [ ] flext-grpc: migrar configs de GRPC (se aplicável)
- [ ] flext-web: alinhar com Settings/Registry (se usar configs específicas)

Notas:
- Preserve compatibilidade de saída (dict via `model_dump()`)
- Preserve mensagens de erro dependentes dos testes
- Evite side-effects em import; use bootstrap no entrypoint
