# Domain-Driven Design Guide - Audit Report

<!-- TOC START -->

- [Findings](#findings)
- [Recommended Actions](#recommended-actions)

<!-- TOC END -->

**Reviewed**: 2026-02-17 | **Scope**: Canonical rules alignment and link consistency

**Document**: `docs/guides/domain-driven-design.md`
**Date**: 2025-11-15
**Status**: ✅ Examples compile; adjust integration notes

______________________________________________________________________

## Findings

- Entity, Value, and AggregateRoot samples use `m.Value` (no stale `Value` references remain).
- Service examples return `FlextResult` and align with `service.py` signatures.
- Layer references still mention `FlextBus` integration in historical notes.

## Recommended Actions

1. Update integration callouts to reference `FlextDispatcher` (dispatcher registration + dispatch) instead of the removed bus layer.
1. Add a brief note on how aggregates can publish events handled through dispatcher-managed subscribers.
1. Keep imports scoped to the symbols used in each example.
