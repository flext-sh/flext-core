<!-- AUTO-GENERATED FILE — regenerate through `make gen APPLY=Y` from the workspace root. -->
<!-- Source of truth: `docs/guides/security.md`; adjust that source, never this projection. -->

# flext-core - Security Guide

> Project profile: `flext-core`

<!-- TOC START -->
- [Dependabot vulnerability governance](#dependabot-vulnerability-governance)
<!-- TOC END -->

Security practices are governed by project-specific policies and central architecture ADRs.

Primary references:

- `docs/architecture/adr/README.md`
- `.agents/skills/scripts-security/SKILL.md`
- `flext-core/docs/architecture/clean-architecture.md`

## Dependabot vulnerability governance

- O inventário oficial de alertas de segurança está em:
  - `docs/reports/dependabot-alerts-2026-06-24.md`
- O plano atual cobre três frentes:
  - inventariar alertas por gravidade e pacote,
  - agrupar remediações em ondas (critical/high first),
  - ampliar Dependabot para rastrear os módulos Python com `pyproject.toml` no monorepo.
- A execução de segurança deve registrar evidência por ação (alerta, commit de correção e status de fechamento) no `bd`,
  sem "close" sem trilha.
