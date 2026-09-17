<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/security.md`; adjust that source, never this projection. -->

# flext-core - Security Guide

> Project profile: `flext-core`

<!-- TOC START -->
- [Dependabot vulnerability governance](#dependabot-vulnerability-governance)
<!-- TOC END -->

Security practices are governed by project-specific policies and central architecture ADRs.

Primary references:

- `docs/architecture/adr/README.md`
- `docs/architecture/baseline-v0.13.0.md`
- `docs/reports/dependabot-alerts-2026-06-24.md`

## Dependabot vulnerability governance

- The official security alert inventory is at:
  - `docs/reports/dependabot-alerts-2026-06-24.md`
- The current plan covers three fronts:
  - inventory alerts by severity and package,
  - group remediations into waves (critical/high first),
  - expand Dependabot to track Python modules with `pyproject.toml` in the monorepo.
- Security execution must record evidence per action (alert, fix commit, and closure status) in the bead tracker, without closing without a trail.
