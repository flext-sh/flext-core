# Entry Docs Audit

<!-- TOC START -->

- [Highlights](#highlights)
- [Issues to Address](#issues-to-address)
- [Recommended Updates](#recommended-updates)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

**Documents**: README.md, INDEX.md, and navigation landing pages
**Date**: 2025-11-15
**Status**: ⚠️ Minor alignment needed (layer listing + dispatcher wording)

______________________________________________________________________

## Highlights

- Navigation links are intact and point to updated architecture and API sections.
- Layered overview is still accurate for the core modules after the bus removal.

## Issues to Address

1. **Layer 3 listing**

   - Update the Layer 3 bullet to reflect the current set: `h`, `FlextDispatcher`, `FlextRegistry`, and `FlextDecorators`.
   - Remove `FlextBus` and `FlextProcessors`, which no longer exist in the package surface.

1. **Landing copy**

   - Any lingering references to "bus" wording should be replaced with "dispatcher" to match the application layer API.

1. **Quick-start wording**

   - Ensure the quick-start examples highlight dispatcher-based CQRS alongside DI and railway patterns.

## Recommended Updates

- Refresh the Layer Architecture section accordingly.
- Do a pass over README headings and summaries to ensure terminology is dispatcher-first.
- Keep import blocks minimal and scoped to the symbols being demonstrated.
