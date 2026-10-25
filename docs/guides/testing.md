<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/testing.md`; adjust that workspace source, never this member projection. -->

# flext-core - Testing

> Project profile: `flext-core`

<!-- TOC START -->

- [Test design](#test-design)
- [Pydantic and mypy](#pydantic-and-mypy)
- [Canonical execution](#canonical-execution)
- [Generated documentation](#generated-documentation)
- [Related guides](#related-guides)

<!-- TOC END -->

FLEXT tests prove observable runtime behavior through public package facades. The
workspace root `AGENTS.md` and the nearest package scope remain authoritative.

Where Gas City and Beads are selected, resolve their effective state and canonical
store through `~/agents/rules/coordination/gascity.md`. The active Bead owns the
declared scope and evidence. A running supervisor or session is not project runtime
or test acceptance, and orchestration is not a workaround for a missing native
environment. Do not assume another city's activation or connection settings.

## Test design

- Exercise only public `api.py` surfaces and canonical `c`, `t`, `p`, `m`, and `u`
  facades.
- Follow the current source-owned signatures and return contracts. A sequence return
  does not promise a mutable list, and a structural port exposes only its declared
  properties. Update stale callers instead of widening production parameters.
- Construct canonical models directly with their declared fields and collection
  types. Keep serialization round-trips for serialization tests; do not use them,
  unchecked copies, casts, or local conversion helpers to adapt an outdated fixture.
- Exercise malformed external data through the public validation boundary. Keep valid
  service calls typed, and retain their failure scenarios through real providers.
- Project-owned helpers, including decorators, belong to the single `u` surface;
  do not introduce a competing `d`/`de` helper facade or compatibility shim.
- Put shared setup in the unified `conftest.py` and typed fixtures under
  `tests/fixtures/`.
- Use `tm` matchers and shared `flext-tests` builders for assertions and test data.
- Read project-owned values from typed config or settings. Never freeze current defaults
  in tests, examples, or golden files.
- Use real, bounded dependencies. Mocks, fakes, stubs, patching, monkeypatch mutation,
  and assertions about private construction are prohibited.
- Treat warnings, unexpected skips, empty collection, and suppressed failures as red.
  Connectivity prerequisite skips follow root `AGENTS.md`, section "Production
  Readiness And Connectivity Tests"; they never prove the omitted service works.

Private attribute usage is enforced by Pyright's resolved owner and export semantics in
source, tests, examples, and scripts according to the typed path policy. The
`ban-test-private-access` ast-grep rule retains only dynamic private-module imports; it
does not duplicate semantic attribute detection or require deleting test scenarios.

## Pydantic and mypy

Pydantic 2 is mandatory for owned model contracts. Consume it through the canonical
FLEXT facades and model presets; do not introduce `pydantic.v1`, raw consumer bases, or
compatibility models. The [Pydantic mypy plugin](https://docs.pydantic.dev/latest/integrations/mypy/)
adds model-aware static analysis; it does not replace runtime validation.

Upstream Pydantic declaration primitives retain their library/model owner. Moving
project helpers does not authorize importing `u` backward into `m`, `config`, or
`settings`, or changing validation behavior merely to relocate a decorator.

`flext-infra/config/tooling.yaml` owns the mypy policy. Its generated configurations,
runtime profiles, project overlays, and checker arguments must retain `pydantic.mypy`
and the operator-mandated suspension of `prop-decorator` and `call-arg`. Change and
regenerate the producer, never hand-edit a consumer projection. Correct YAML alone
does not prove that the installed checker consumed the policy.

These two suspensions apply to mypy only. They do not authorize disabling unrelated
diagnostics, removing the plugin, weakening another analyzer, or excluding source,
tests, examples, or scripts from their native gates.

Treat code changed to accommodate these two diagnostics as a regression. Use
the original commit, diagnostic, canonical owner, and public behavior to establish
causality; restore only the affected hunks and preserve unrelated changes. Revalidate
the lawful Pydantic 2 declaration through the public consumer before encoding its
behavior in tests. Do not replace a validator or computed field with a shim, unchecked
mapping, cast, or alternate model to satisfy the checker.

## Canonical execution

Run tests only through the dispatcher at the workspace root:

```bash
make tests
```

`make tests` executes the complete applicable suite once and retains the project's
persistent Testmon database. It does not run an incremental suite first. The generated
Make boundary fixes the target to the typed configuration's complete test directory;
caller variables cannot replace it with a file or narrower directory. The runner
accounts for collection, execution and declared exclusions through its existing typed
receipts. Never clear or bypass the database, or invoke the underlying runner directly.

The supported Testmon environment combines installed toolchain provenance with current
`config/*.yaml` content. Declare additional template and resource directories in the
workspace manifest's `test_inputs.templates` and `test_inputs.resources`. Declare
behavior-affecting environment variable names in `test_inputs.environment`; only their
digests enter the environment identifier, never their values. Missing and empty values
remain distinct. Directory creation, removal, and content changes invalidate the
selection conservatively while preserving the same external database. Each downstream
workspace owns its own declarations; the infrastructure template catalog does not stand
in for downstream inputs. Declared directories use the existing authenticated physical
inventory contract: symbolic links and multiply linked files fail explicitly. This is
not an unrestricted filesystem dependency scanner; declare physical source inputs and
repair the inventory owner if a legitimate native consumer requires another shape.

The approved Make consolidation contract supersedes the older incremental/full verb
split. CI and pre-push also call `make tests`; pre-commit runs no tests. No compatibility
aliases or file selectors belong to the common Make interface.

The complete suite retains its declared services, network access and authentication
requirements. Connectivity prerequisite skips follow root `AGENTS.md`; they never
prove the omitted service works. Marker applicability belongs to typed configuration,
and the runner records every excluded item. An exclusion is never a passed test. A
runner that violates this contract is repaired at its owner and rerun through Make.

Each phase retains its mode, database, raw process outcome, collection manifest, and
diagnostics. The latest receipt names the current attempt even when collection fails.
Warning totals include selection, inventory, and suite occurrences, with blocking and
explicitly suspended warnings reported separately. The existing non-strict MRO
enforcement suspension remains visible in those receipts. Unexpected skips block
acceptance; connectivity prerequisite skips retain their explicit unavailable reason
and never establish production readiness.

The persistent database remains available for impact analysis and later runs. A cache
hit cannot replace the complete execution required by `make tests`. Empty collection
or zero execution fails this verb.

A receipt used for acceptance must identify the exact source candidate, local changes,
installed dependencies and checker policy, command, working directory, declared scope,
raw exit, and decisive output. A root gitlink, an installed editable provider, and a
remote integration tip are different identities until their relationship is proven.
Record not-executed cases explicitly; never transfer an older passing count to new
source or newly added tests.

For executable Markdown, use the actual collector's eligibility and item identities.
Reconcile eligible fences with collection and successful setup, call, and teardown;
an enabled Markdown option or accepted file path is not execution evidence. Invalid
eligible Python fails visibly, and skipped, deselected, retried, or missing items are
not passed fences. Do not substitute the formatter's parseable-block inventory or a
second parser for the runtime collector.

Reports have invocation scope. A formatter-only or spelling-only `PASS` is not a full
lint/type/test receipt, and a mutable latest-report path does not preserve the result
of an earlier command. Use the existing typed receipt owner to retain each phase's
identity and outcome. Administrative PR integration does not replace native runtime,
lint, type, coverage, or post-merge proof.

Run the complete verification gate through the same dispatcher:

```bash
make check
```

The checker evaluates the current declared repository. An omitted project selection
never expands to nested members; a root without project metadata fails even if it
contains declared members. Fleet orchestration invokes each repository's own lifecycle.

Conformance can explicitly select multiple repository owners for one generation
transaction. Lazy-init opens a separate Rope index for each selected owner and combines
the authenticated inputs and publication plans; it never widens the parent's implicit
scan to include submodules. A module target must resolve in exactly one selected owner.
Missing or ambiguous targets fail before publication, and repeated generation verifies
the complete selected set.

Run the declared operation without selectors. If a required workflow is missing,
repair its existing owner and rerun the standard verb; do not introduce a parallel
diagnostic command or revive retired aliases.

## Generated documentation

Member copies of this guide are generated projections. Change this root source and
regenerate from the workspace root:

```bash
make gen
```

Do not edit a member projection by hand.

## Related guides

- Development
- Troubleshooting
- Testing standards
