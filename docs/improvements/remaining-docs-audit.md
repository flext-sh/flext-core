# Remaining Docs Audit

**Scope**: Secondary docs not covered by main audits (standards, guides, improvers)
**Date**: 2025-11-15
**Status**: ⚠️ Cleanup required for legacy bus references

---

## Key Findings
- Several secondary guides still mention `FlextBus` in verification snippets and import blocks even though the application layer now exposes `FlextDispatcher` only.
- `FlextProcessors` appears in a few checklists but the module is not present in `src/flext_core`.
- Import blocks are otherwise lean and map to current symbols.

## Recommended Remediations
1. Replace `FlextBus` with `FlextDispatcher` (or remove entirely) in:
   - verification commands (`python -c "from flext_core import FlextDispatcher ..."`)
   - example import blocks in secondary guides and standards files.
2. Drop `FlextProcessors` from standards/checklists where it was previously listed as an application-layer component.
3. Re-run a quick `rg "FlextBus" docs/` check after edits to ensure no residual mentions remain.
4. Keep existing minimal import style; avoid reintroducing broad import lists.
