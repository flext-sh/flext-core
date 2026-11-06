<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/engine.md`; adjust that workspace source, never this member projection. -->

# flext-core - FLEXT Engine Law — Rope-Only Python Interpretation

> Project profile: `flext-core`

<!-- TOC START -->

- [What stays canonical](#what-stays-canonical)
- [Isolation by repository](#isolation-by-repository)
- [Deprecation window](#deprecation-window)
- [Correction authority](#correction-authority)
- [Blocked capabilities](#blocked-capabilities)

<!-- TOC END -->

The active rewrite's 2026-10-30 handoff
records owners, source changes, unresolved bootstrap/candidate failures and exact
runtime limits. Bead `flext-u9h2i` remains open. Existing templates also form the
base for refactoring existing modules: the existing Infra Rope catalog assembles
typed slot content, preserving authored code and native identity. See
ADR-018.

| Workspace declaration                                                              | Accepted decision                         |
| ---------------------------------------------------------------------------------- | ----------------------------------------- |
| This page; the living census and wave status (beads `flext-pipol`, `flext-u9h2i`). | ADR-028 Decision 17, amendment 2026-10-28 |

Operator law, 2026-10-28: **rope is the only accepted Python interpreter in
FLEXT's own code.** The flext-infra engine never parses, analyzes or rewrites
Python source with the `ast` module, `libcst`, or `re` applied to code — every
such site is deprecated and migrates to rope's PyModules from the single
indexed project.

The operator reiterated the complete cutover on 2026-10-28: "vc tem que
exterminar ast e libcst , e todos fallback" (tracker memory
`operator-order-2026-10-28-exterminate-ast-libcst-fallbacks`). Accessing syntax
nodes through Rope is also a superseded route. Use native DSL queries, source
bindings, scopes and refactorings; a missing capability is repaired at its native
owner, never replaced by a second parser or fallback implementation.

Native module election preserves the first import-resolution exception. A
failed stdlib spec lookup is not converted to an absent spec followed by a
second resolver attempt. Source and extension-module selection retain their
declared native boundaries.

Facade component ancestry consumes the superclass objects exposed by Rope for
both source and builtin classes. A builtin has no source header or AST. Do not
reconstruct this hierarchy from import text, lazy-map literals or alternate
lexical binding paths. An invalid native superclass remains an error; the
planner must not skip it or substitute another resolver.

The core exception facade composes the violation declarations from their single
defining module. Renaming a declaration must not make bootstrap depend on an
outdated generated package re-export. Keep the defining module and public facade
consistent; the normal generator still owns package projections, which are never
patched manually to repair startup.

Lint-identity authentication consumes that same governed source inventory and
native bindings. It must not recursively scan report trees, interpret source
with AST, infer a settings owner from a class-name suffix, ignore decoding
failures or discard a finding merely because a probe path cannot be read.
Preserve the diagnostic and authenticate only the exact declared contract.

An interrupted source cutover may need the immutable declaration that preceded
its removal. Read it through the existing Git blob owner with an explicit
revision; its default remains `HEAD`. Historical bytes authenticate the analysis
candidate only. Keep the live physical witness and current publication source
separate, and never publish historical providers to make unresolved consumers
appear valid.

Removal admission consumes the existing census and its explicit `consumer_roots`
from the typed operation request and semantic source plan. Pass those validated
inputs through composition by reference. An empty local census does not prove
that external consumers have migrated; do not discover neighboring checkouts or
introduce another inventory, receipt protocol or retirement ledger.

Lazy projection consumes those same native modules, source bindings and import
owners. A second Python syntax model, class-ancestry cache or export/member fact
cache is not a separate authority. Remove the duplicate parser and derived
transport mirrors while rewiring the existing planner and renderer; do not
replace them with another wrapper model or another source scan. Pydantic remains
the owner of validated input and publication contracts.

The approved facade-planner cutover is defined by
ADR-018's nested projection contract.
Rewrite `codegen_facades.py` as a few declarative composition lines over the
existing families and add facade planning to the existing workspace DSL. Retire
`CodegenFacadeSyntax` and its authored CST/AST routes through the same consumer
cutover. Do not relocate the monolith, conceal it in a wrapper or reproduce its
visitors over Rope syntax nodes. Reuse native queries and utility operations;
the existing candidate and publisher retain their owners. Preserve functionality
and prove net code reduction over the complete change. This is the accepted
target, not a receipt of implementation or runtime recovery.

The same ADR owns scope-aware error handling: existing typed results/models and
the publisher retain the cause, traceback, context and reached/unreached work.
A failed local unit blocks unsafe dependents while proven independent cohorts
continue; any error makes the final command nonzero. Preserve valid effects and
WIP. Neither silent continuation nor broad exception capture that reports
unexpected failure as success is an error-handling strategy; add no parallel
error framework.

Initializer import rendering uses complete declared module identities. It may
shorten a target relative to the current package only when both identities share
their actual root. An underscore prefix or a repeated private path segment never
authorizes inventing a relative import; incomplete identities are repaired at
the source index, not guessed by the renderer.

Generation progress reports actual boundaries: request planning, native source
capture, repository planning, candidate admission and publication. Repository
rows name the physical root, surface, start or end, elapsed time and observed
file count. A start without an end identifies unfinished work, never success
or an estimated percentage. Distinguish lazy-provider and managed passes when
the same repository appears twice. Keep per-symbol inference at debug verbosity;
normal progress must not repeat full structured completion payloads. Profile
evidence belongs to the exact process and invocation; an older latest-path
profile does not diagnose the current command.

An export query over one candidate file binds that file in the existing source
view and reads its native module. It must not enumerate or materialize the entire
repository as a side effect of obtaining that binding. Reuse the original file
witness and inherited plans; unchanged provider sources remain in the same
native project. Discovery belongs to the inventory owner, once per relevant
source epoch, not once per exported module.

Parse external configuration once into the existing Pydantic contracts and pass
validated values by reference. Runtime-checkable protocols own capability shapes;
native identity checks still establish that the SDK supplied the declared object.
Reuse that single boundary instead of repeating manual validation downstream.
Compose typed result-producing operations through the public `FlextResult`
railway where its exception contract applies; preserve raw exceptions at
boundaries that require their original traceback. No JSON round-trip or model
copy is a substitute for passing a validated input.

Composed documentation models declare shared catalog rows at the leaf that
consumes them; the public documentation holder inherits that declaration by MRO.
A leaf cannot resolve its own package's fields through the upstream core `m`
facade or import its unfinished aggregate facade to complete a schema.

`upg` upgrades the complete Python declaration through one native writer:
generated `make upg` runs `uv lock --upgrade` before any FLEXT Python;
`workspace verify-lock` then verifies that lock and `deps upgrade-mise-lock`
upgrades the Mise half. No package-selected upgrade or selected-override projection
exists; the former `FLEXT_INFRA_UPGRADE_PACKAGES` route was a second UV writer running
inside the environment it had to repair. `setup` activates the resulting locked candidate
through its normal owner.

## What stays canonical

| Engine                 | Domain                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rope                   | Python analysis and refactoring: one loop and native session/index per repository per invocation, shared by DI; small dependency-closed cohorts use the same publisher within the five-second transaction budget (ADR-028 Decision 17; ADR-018 transaction contract). This does not mean one physical transaction for the whole invocation. Native refactorings first (rename, move, restructure, ImportOrganizer); custom transforms operate on the indexed PyModules through rope's public API. |
| ast-grep               | Declarative rules (detection AND fix), one generalized rule per law (ADR-028 Decision 11, Decision 18); the rules catalog is SSOT data.                                                                                                                                                                                                                                                                                                                                                           |
| sed                    | Non-Python text.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Generator (`make gen`) | Projections from declared templates.                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Isolation by repository

The engine never resolves a member module through the interpreter: the
interpreter may elect stdlib modules only (`sys.stdlib_module_names` gate in the
rope native project); member code resolves through the repository's own rope
index. A fresh interpreter — where genuinely required — uses
`u.Infra.runtime_python(root)` and the physical environment elected by checkout
topology: the workspace-owned `.venv` for an attached member, or the checkout's
own `.venv` in standalone mode (ADR-003, Decision 5). It never borrows the engine's
ambient environment. Divergent member libraries never leak into engine analysis.

## Deprecation window

Superseded engine routes follow the workspace's deprecated-signature window:
a typed warning marks the route (`compat: <route> -> <new owner>; fix:
<rewire>`), every use is reported as a finding in the consumer (the census),
rewires land in waves through `make mod`, and the route is deleted when the
census reaches zero. Detection without a fix form is a rule defect.

## Correction authority

Violations in member code are corrected by the engine — declarative rule fixes
for mechanical shapes, rope refactorings for semantic ones — through
`make mod`. Manual member edits are the exception of the exception, justified
in the commit.

## Blocked capabilities

A site with no rope-expressible equivalent (for example, runtime-object
semantics: `is`-identity, `__mro__` of live classes, protocol checks) is
reported as `BLOCKED-ON-ROPE-CAPABILITY` with the rope API evaluated, and
stays untouched until the operator rules; an alternative parser is never
silently kept.
