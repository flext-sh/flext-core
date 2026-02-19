# Scripts

<!-- TOC START -->

- No sections found

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

Utility scripts that support the FLEXT Core 1.0.0 modernization programme.

- Prefer short, single-purpose commands that are safe to re-run.
- Use `poetry run` (or `make`) so dependencies resolve through the project environment.
- Keep dispatcher/context/config migrations scripted when possible (e.g., export checkers, doc sync helpers).
- Document inputs/outputs in the script header so downstream teams can reuse them during ecosystem migrations.
