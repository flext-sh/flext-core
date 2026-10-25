<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/make-commands.md`; adjust that workspace source, never this member projection. -->

# flext-core - FLEXT Make Commands

> Project profile: `flext-core`

<!-- TOC START -->

- [Discover commands](#discover-commands)
- [Architecture admission and acceptance](#architecture-admission-and-acceptance)
- [Canonical workflow](#canonical-workflow)
- [Operation ownership](#operation-ownership)
- [Constants migration](#constants-migration)
- [Verb single-pass contract](#verb-single-pass-contract)
- [Markdown quality pipeline](#markdown-quality-pipeline)
- [Test contract](#test-contract)
- [Failure contract](#failure-contract)
- [Scope and generation](#scope-and-generation)
- [Related guides](#related-guides)

<!-- TOC END -->

The typed Make catalog owns command grammar. `make help` at the workspace root
projects that catalog; this guide records the invariants every declared verb keeps.

## Discover commands

```bash
make help
```

Help is available before provisioning. It lists the operations and named inputs
declared for the selected profile. The common lifecycle has one public verb per
operation; profiling and reporting belong to the operation's existing engine.
`OPTIONS=Y` or `HELP=Y` displays the selected operation's catalog contract without
running its prerequisites, hooks or handler, including repository-owned product
verbs. These views do not install tools, prepare a runtime or enter a product
dispatcher. Detailed product subcommand help remains owned by that dispatcher's
declared public interface.
The Make template owns separation of optional release and script sections and
emits one final line feed for every profile, including an empty optional section.

Named inputs describe data consumed by an operation, not an alternative supplier
resolver or installation mode. Their presence in help does not select a partial
operation. Workflow
admission requires the operation in every declared profile; optional input metadata
must not reject a valid bare invocation. Common handlers keep their complete scope.
The violations sweep consumes those same profile declarations and uses `check` for
its read-only rules operation.

Never infer a target, flag, or selector from historical documentation. When a required
verb is missing or broken, repair the root dispatcher owner and rerun that verb.

## Architecture admission and acceptance

Command success does not replace architecture acceptance. Start with
[root AGENTS.md](https://github.com/flext-sh/flext/blob/0.12.0-dev/AGENTS.md), the
[session router](https://github.com/flext-sh/flext/blob/0.12.0-dev/.agents/commands/flext-law.md), and its exact
[routing](https://github.com/flext-sh/flext/blob/0.12.0-dev/.agents/skills/flext-context-routing/SKILL.md) and
[FLEXT law](https://github.com/flext-sh/flext/blob/0.12.0-dev/.agents/skills/flext-law/SKILL.md) owners. The router composes
global `search-first`, `yagni`, `ssot`, `solid`, `simplify` and `dry`; concrete
dependency and data boundaries additionally follow the canonical global
`dep-injection`, `clean-architecture` and `pydantic-development` skills under
`~/.agents/skills/`. Read their procedures, not a copied local substitute.

Before generation or structural modification, record the domain owner, declaration family,
namespace, affected consumers and obsolete paths in the active Bead. FLEXT law's
**Architecture and imports** owns facade MRO `c -> t -> p -> m -> u`, inward
dependencies, data-only declarations and explicit `p`-port/`api` composition.
Root **Architecture Overview** owns config/settings as layer-zero SSOT with no
own-namespace imports. Repair declaration and dependency order at their owner;
do not hide a cycle through local imports, eager exports or compatibility aliases.

Reuse an existing helper only at its correct domain boundary. Parse owned input
once through canonical Pydantic models and preserve typed serialization/results.
The **Sources, generation, and commands** section of FLEXT law owns source-only
projection changes and canonical regeneration. Never hand-edit generated member
guides, facades or Makefiles to obtain a green result.

After each bounded repair, prove complete consumer rewiring, removal of the old
route and net-negative simplification of redundant implementation without reduced
functionality. No duplicate engine, fallback, shim, suppression or weakened
configuration is acceptance. Apply global `make-check` and `verification-loop`:
record command, CWD, exit, scope and original receipt; exercise the public runtime,
run applicable lint/type/test/docs gates and prove repeated gen/fix/fmt are no-ops
on the unchanged candidate. Later edits invalidate overlapping evidence. Root-only
checks are not fleet proof; failures and unexecuted gates remain explicit.

## Canonical workflow

The approved `flext-0029c` native-bootstrap target uses one root uv workspace and
lock in local execution and root CI; CI checks out declared submodules and installs
noneditable. Suppliers use native pyproject sources, not environment selection,
persisted state or synthetic resolvers. Official Mise acquisition and native locked
tool provisioning precede `uv sync --frozen` into the owned physical `.venv`.
No custom Python or AWK/shell lock/installer parser, copied admission resource or
installed Infra check precedes successful provision. The task-only
direct public CLI route
keeps Make integration last. This target contract does not attest current runtime;
the general workflow below remains unchanged outside that task.

Run each required operation explicitly from the workspace root:

```bash
make setup
make gen
make mod
make fix
make fmt
make build
# Exercise the selected public consumer before the final tests.
make check
make tests
```

Generation and structural modification each publish the dependent projections required
by their own change. Repeat an operation only when its inputs changed or to prove
convergence. Run `make upg` explicitly
when dependency resolution needs to change, and then `make setup` to install that
resolution. Mutation and documentation generation finish before the final
candidate's public runtime, checks and complete suite. `make build` packages that
candidate; it does not replace runtime verification.
Each standard verb executes its declared operation directly, without selectors that
change its effect or phase. Domain commands may retain explicitly declared selectors
interpreted by their owning registry; they never select a standard lifecycle's effect.

Setup provisions governed gitlinks before creating the runtime environment. Its shared
owner is `flext-infra/src/flext_infra/templates/project/base/submodule_setup_recipe.j2`;
`config.Infra.codegen.make.submodule_timeout_seconds` bounds native provisioning, and
the root credential contract supplies authentication without logging the token.
The recorded gitlink is the parent index entry, not the origin's moving branch tip.
Setup initializes only governed gitlinks that `git submodule status` reports as
uninitialized, at depth 1 and without recursion; submodules marked
`flext-managed = false` are never initialized. Initialized checkouts retain their
branch, HEAD, index, files, and local work: setup neither fetches, validates, nor
rewrites them.

`make help` is the complete live inventory. The approved common surface is
`setup`, `upg`, `gen`, `mod`, `fix`, `fmt`, `check`, `tests`, `help`, `build` and
`clean`. Repository-owned product commands join it only through their declared
extension owner. No compatibility alias preserves a retired common verb.

The ordinary dispatcher executes each declared `pre-<verb>` hook, handler and
`post-<verb>` hook once in local and CI contexts. CI supplies activation and its
declared scope; it does not bypass these extensions. Approval wrappers still
must delegate to the corresponding builtin. Setup retains its separate
provisioning and activation contract.

Release operations belong to an explicitly declared product protocol.
`publishes_release` elects `release-<phase>` commands from the shared release
engine's phase enumeration. The same declaration produces their project Make
handlers and workflow references; no duplicate `repository.extra_verbs` rows are
required. Each handler invokes the existing release CLI directly.
`PR_TITLE` supplies the optional plan title, `LANE_CHECKPOINT_REF` the optional
versioning checkpoint and `INDEX=Y` the explicit package-index publication request.
Generation, checks and local builds never perform tagging or publication.
The same full-owned catalog declaration authorizes a workflow update or its
retirement when its profile or capability no longer applies. Both operations
capture the original physical destination and current declaration inputs before
publication; a fresh clone needs no machine-local history. Headers never elect
ownership. Authored seeds are excluded from managed plans, and merge ownership
does not authorize deleting an entire file.

Documentation uses the existing documentation engine through each operation:
`gen` produces sources, `fix` repairs, `fmt` formats, `check` validates and audits,
and `build` produces the local site artifact. The typed documentation policy owns
the applicable actions and report location. Sharing its engine does not authorize
another composite public lifecycle. The first failed action retains its original
exit status and emission. Reached, incomplete and not-reached phases remain
explicit in the existing receipts; partial execution is never complete acceptance.

The root `pre-gen` hook invokes the existing typed plan collector using
`config/plan-collection.yaml`. That configuration alone authorizes collection.
When disabled, the collector only plans retirement of its recorded manifest-owned
outputs; the hook does not enable sources or expand ownership.

At the native documentation engine boundary, an explicit project selection must
resolve to its requested governed scope, including for a standalone repository.
Unknown, unsafe or explicitly empty
selection fails before publishing phase outputs; it is never replaced by the root
scope. Omitting selection retains the declared root/workspace semantics. This is
the shared scope owner's contract for existing documentation operations, not a
second selector implementation in a dispatcher or a new public lifecycle.

Under ADR-025's October 9 amendments,
setup provisions current committed locks without resolution or lock writes; check
judges missing, stale or incomplete locks. An absent
Mise self-pin uses the qualified host provisioning path, not an invented release.
Functional installation failures always remain red. `make upg` owns resolution
and lock writes. Distributed JavaScript tools belong to one owned Bun project
with `package.json` and a frozen lock, never a Mise npm backend; native Node
wrappers remain where used. These contracts are not current runtime receipts.

## Operation ownership

Each verb enters its existing domain owner. Generation and conformance share the
same projection planner and transaction publisher; they do not share permission
to repair authored declarations.

| Verb | Owned operation |
| --- | --- |
| `setup` | Install committed locks, synchronize the physical environment and publish its declared runtime bindings |
| `gen` | Render and publish projections from the current declarations |
| `upg` | Resolve declared dependencies and tools and publish their resolution metadata and locks |
| `mod` | Apply the declared semantic and structural consumer rewrites |
| `fix` | Apply conformance repairs and the configured fixable gates |
| `fmt` | Run the configured formatters |
| `check` | Run the configured read-only gates and check conformance |
| `tests` | Execute the complete applicable suite once with the persistent Testmon database |
| `help` | Discover the declared public interface without effects |
| `build` | Produce local distribution and documentation artifacts without publishing |
| `clean` | Remove only declared disposable resources owned by the selected target |

Generation and captured check processes share the validated
`settings.Infra.profile` instrumentation setting, disabled by default.
`FLEXT_INFRA_PROFILE=true` enables profiling through the existing adapter without
changing the operation, scope or deadline. `make.generation_profile_path` owns
the generation artifact's repository-relative destination. The adapter admits
settings before starting cProfile; that admission remains inside the command's
wall-clock deadline and outside its profile capture. Native status, tracebacks and
interrupted-run evidence retain their existing owners. The recipe becomes active
after its generated Makefile is published.
Product release commands expose their optional inputs in `help` and `OPTIONS=Y`;
contract display never executes their handlers.

The operation that initiates a semantic change owns its complete coherent
publication, including dependent facades and package exports. Generation and
modification use the same source authority, typed plans, engines and publisher;
neither recursively invokes another public verb. A generated file is changed
through its generating owner, never by a second repair or formatting writer.

New-project generation consumes the target's own explicit project, integration,
kind and provenance declarations. Workspace membership comes only from
`.gitmodules`. Its Git/worktree is prepared before generation; a missing
`pyproject.toml` in an established repository is a failure, not scaffold intent.
Generation never initializes remotes or creates an artificial commit.

Each implementation slice reconciles its README, guides, AGENTS, ADRs, tracker
and relevant canonical skills before code changes. Prove the public operation in
the real consumer before changing tests. Tests then validate observable behavior
without mocks, implementation-shaped assertions, tautologies, copied
configuration values or fallback execution.

The generated dispatcher invokes conformance checking only through `make check`
and conformance repair only through `make fix`. Installing, generating,
upgrading, rewriting or formatting does not implicitly run a global conformance
check. Each operation still proves the prerequisites and effects it actually
consumes. Native environment verification, input authentication and publication
receipt verification remain required; they are not global conformance scans.

General code audits belong exclusively to `make check`. `make tests` executes
tests; it does not run the general check gates. Generation reads the declarations,
templates and source exports needed for its projections, including lazy
initializers and facade imports. Its publisher guards the consumed file identities,
destination preimages and journaled effects without traversing the generator's
executable dependency graph. Documentation generation uses the same physical
publication boundary. Facade generation preserves authored declarations and imports
outside the selected family's exported bindings. The namespace rule catalog,
consumed by `make check`, owns their architecture findings.

The read-only conformance observer verifies the completed journal, actual inputs,
effects and requested scope through that same publication owner. It does not
repeat the executable dependency audit owned by the `fresh-import` check gate.

Facades and the planned package initializers required by their imports form one
dependent publication group. Generation reuses those initializer plans rather than
publishing a facade against an older export map. Unrelated managed artifacts remain
outside that group.

Completed receipts retain their original input evidence. The advancing publication
frontier contains outputs: a later phase proves those outputs before consuming them,
and authenticates its own inputs before effects. Read-only inputs from a completed
phase do not become outputs that every subsequent phase must revalidate.

Generation uses the provisioned physical runtime without first activating the
environment file it may regenerate. Verbs that consume activation enter the
declared environment before their handlers. Both routes retain the same physical
workspace and interpreter ownership checks.

Setup in the declared CI context synchronizes the environment without authorizing a
local `.envrc`. Native tool installation and version proof remain mandatory. Backend
executable names belong to the typed toolchain entry and its generated Mise options;
the installer does not repair an incorrect name through shell renames or retries.
CI tool caches match the generated Mise configuration as well as the committed
locks and package metadata. A cache from another declaration is not a restore
candidate. Upgrade consumes the current declarations and publishes resolution
metadata and locks. Installation remains an explicit setup operation; an upgrade
does not re-enter setup or invoke generation hooks.

Runtime bindings are generated outputs. `setup` and `gen` render them from the
declared configuration without migrating authored configuration or retiring its
Git index entries. For tracked editor settings, `fix` owns the existing migration:
preserve the portable configuration in its declaration owner before retiring
tracking and publishing the runtime-local binding. A binding refresh cannot
silently authorize that source migration.

A usable physical runtime and a producer that understands the selected declarations
are prerequisites. A missing prerequisite remains a failure until its owning
operation repairs it explicitly. There is no older-producer fallback or hidden
generation, upgrade or installation chain.

## Constants migration

The constants-consumer phase uses Rope identities, class ancestry, occurrence
coordinates and native import plans. Python detection and transformation at this
owner do not use a separate AST, CST or regular-expression parser. Its edits
participate in the existing semantic publication and fixed-point loop.

The census recognizes `typing.Final` by its resolved identity, including aliased
imports and lowercase declaration names. A user-defined type named `Final` does
not establish constant ownership. Parameter positions come from Rope occurrences
in the same immutable module snapshot, including multiline signatures.

Within the declared constants family, a class-level `typing.Final` annotation
becomes `ClassVar` so a composed Pydantic namespace does not create an instance
field. Rope resolves aliases and qualified imports; same-spelled user types,
module bindings and function-local declarations retain their contracts. The
initializer remains unchanged, including `frozenset` expressions that depend on
runtime inputs. Annotation repair does not authorize extracting a calculated
collection or replacing its elements.

Constant imports resolve through their unique reaching declarations and the
published facade's MRO. Local shadowing, import cycles and competing owners reject
the candidate before publication. Consumer tests exercise public tap, target and
dbt providers; their expected values come from the same upstream facades rather
than freezing current configuration.

The same phase can reuse an existing public string or nonnegative decimal
integer constant in a complete function return. Rope proves the literal region,
its exact inferred type and its single assignment owner;
computed expressions, docstrings, defaults and metadata retain their meaning.
An enum member cannot replace a plain string return because its runtime type
differs from `str`.

For singleton `typing.Literal` field annotations, the phase can reuse an
existing public `enum.StrEnum` member with the same wire string. Aliased and
qualified `Literal` imports keep their identity. Defaults and discriminator
metadata remain intact. This rule does not create new enums or rewrite
multi-value Literals or function annotations; those contexts require their own
semantic proof before automatic publication.

Existing singleton `frozenset` constants can replace matching built-in
constructions on the right side of `in` and `not in`. String and decimal integer
elements must be literal declarations. Computed values, mutable inputs and
same-spelled user functions are excluded. Construction and return contexts keep
fresh-object identity; this rule does not extract new collection declarations.

## Verb single-pass contract

The typed tooling and Make policy owns tool membership and capabilities; this guide
does not maintain another gate matrix. `fix` applies declared repairs, `fmt` runs
formatters and `check` performs read-only verification. A formatter does not become a
second lint-repair path, and a repair does not repeat formatting.
`make.fmt_gates` owns the formatter selection, including embedded Python in Markdown.
The repair selection derives from the existing repair-capable catalog minus those
formatters; a gate's reusable mutating implementation does not elect it for both
public operations.
Outside CI, a workspace root's `fmt` and `fix` run one checker process with
`check run --scope all`: the root plus every governed member declared in
`.gitmodules`, never one member Makefile after another (operator ruling
2026-10-25). A gate whose tool natively accepts several project roots
(`union_capable`, today Ruff format) runs once over the union of every project's
targets and its findings are split back per project for the report; the other
gates still run per project inside that single process. CI (`CI=Y`) keeps
`--scope self` and evaluates only its own repository.
`fix` also consumes the existing documentation repair owner for authored links,
table-of-contents projections and fenced-code lint repairs. It prepares one selected
docs bundle only to reuse its captured inputs and exact generated-output ownership;
it does not publish that generation plan. Full generated files remain with their
generator. A declared mixed Markdown file without an existing region compositor
fails before any repair is published. Repairs compose against the captured bytes
and use the existing guarded atomic writer.

Each operation runs once for its selected inputs. Its own admission and publication
checks remain necessary; they do not authorize a hidden global gate pass. Repeat only
after relevant inputs change or to prove a fixed point. Normalizing bytes owned by a
generator remains part of its single publisher, not another formatter over authored
sources.

`make check` runs as one `flext-infra workspace check` process. The read-only
projection verdict (`codegen conform --mode check`) runs first, so projection or lock
skew fails as drift. The captured gates follow, keeping their own attested checker
process, then the lane census (skipped in CI) and the documentation validation and
audit. The first failure stops the verb. Under the commit-hook token the verb runs
only the commit-stage gates. A dirty or changing tree is never a check condition.

The checker owns runtime integrity as a registered gate alongside the other checks.
It invokes the existing dependency, lock, environment-provenance and toolchain
owners once. A repository declares additional Python-module checks in
`make.project_check_gates`; their typed `integrity` flag includes them in the
read-only quick-health selection. The same registry and runner execute either
selection, including repository-specific dependency auditing.

Consumers needing individual verdicts request an invocation-owned report destination
through the declared report-output environment key. The checker publishes its
`CheckReportSummary` there using the existing guarded atomic writer after each
completed gate, retaining the requested inventory and actual executions. An incomplete
inventory keeps the complete check red. Component consumers require every requested
component and its native outcome; they cannot infer success for an unreached component.
An overall Make exit never substitutes for a component verdict. Full health
executes `make check` once and uses that execution's integrity results. Quick health
selects only integrity through the existing checker CLI, preserving the original
required/slow classification even if a later advisory check times out. Missing evidence
is failure. Provisioned tools must have their runtime manifest declaration; an absent
declaration fails at the existing Mise owner without selecting a host executable.
That same owner resolves the executable using the version already recorded in the
lock. Moving requests such as `latest` never trigger resolution during generation
or checking; `upg` owns changes to the resolved release.

## Markdown quality pipeline

The markdown standard lives once in `flext-infra/src/flext_infra/config/tooling.yaml`
(`Infra.tooling.tools.markdown`) and is projected to every repository by `make gen`:

- `rumdl` is the linter (markdownlint-compatible `MD*` rules through the generated
  `.markdownlint.json` / `.markdownlintignore`); syntax findings inside embedded code
  belong to the flext-tests markdown validator, not to a second linter.
- `rumdl fmt` is the formatter and reads the same `.markdownlint.json`:
  `rumdl fmt --check` in `make check`, `rumdl fmt` in `make fmt`. rumdl is locked in
  `uv.lock`; the Mise npm backend remains retired (ADR-025). The amended Bun
  project contract for distributed JavaScript tools does not change this formatter.
- `markdown-code` holds parseable embedded Python and doctest examples to the
  ruff-format contract; unparseable documentation fragments are prose and stay with the
  validator. Generated and provider-projected trees (`.agents`, `.claude`, `.gemini`,
  `AGENTS.md`, `target/`, and friends) are excluded by the same SSOT list.

## Test contract

`make tests` is the single public suite operation. It executes the complete applicable
inventory once, retaining the same persistent Testmon database. It does not prepend an
incremental run, clear the cache, accept a file selector or preserve an alias for a
retired test verb. CI and Git hooks consume this same operation.

The execution receipt preserves the collected inventory, applicability decisions,
executed counts and original subprocess outcomes. Zero collection, an absent manifest
or a cache-only observation does not prove the complete suite. The native inventory
and manifest must agree; neither may silently expand into another target. Warnings
remain visible in the native result.

Connectivity tests follow the root environment and reachability prerequisites;
availability skips never prove connectivity or production readiness. Once a service
is available, authentication, protocol and assertion errors remain failures. Pure unit
and integration tests stay active. Establish the implemented behavior through the real
public consumer before changing tests, then validate it without mocks, tautologies,
private-method assertions or copied configuration expectations.

## Failure contract

- The first exception, traceback, and non-zero exit propagate unchanged.
- Warnings, empty output, and missing tools are failures. Connectivity availability
  skips follow [root prerequisite law](https://github.com/flext-sh/flext/blob/0.12.0-dev/AGENTS.md#production-readiness-and-connectivity-tests)
  and never count as successful connectivity or production validation.
- No retry, fallback, suppression, normalization, partial run, or alternate raw tool
  path can replace the canonical verb.

The recovery contract under implementation also covers a journal-listed
`TEMPORARY` directory intent whose original preimage was absent and whose creation
receipt is still `None`. Reconciliation must remain strictly inside the original
authenticated transaction root. An observed empty directory is a new input to
that reconciliation; it cannot stand in for the missing historical creation
receipt.

Removal requires an exact identity guard and applies only to the proven empty
directory. The original authenticated manifest must agree before and after the
removal. Unknown nonempty content, a different directory or parent identity, or
divergent aliases remain failures and are preserved. Already published independent
effects remain published. Native regression tests and runtime proof for this
recovery case are pending; this contract is not a successful recovery receipt.

## Scope and generation

The parent `.gitmodules` owns membership and each repository owns its typed kind and
local policy. Local workspace generation includes the root and eligible managed
members; `CI=Y` limits generation to the selected project. An independent checkout
owns only itself. Non-FLEXT, unmanaged, read-only and content-only boundaries remain
authoritative; a directory name or missing file never elects another profile or
scaffold mode.

The checker resolves every source's owning project through that topology and uses
its working directory, tool configuration, imports and codemod policy. It never
validates a member as the root or introduces another public file-check verb.

Each `flext-infra check run` invocation exclusively creates one UUID4 directory under
the typed request's report base. `--reports-dir` selects that base.
Relative bases resolve under `--repository-root`, not
the caller's working directory. All project native artifacts, full raw receipts,
Markdown and SARIF reports stay under that same invocation directory. The CLI prints
the actual report paths. Existing fixed-path reports remain historical files and are
not overwritten; no shared latest writer is published. Consumers of
`u.Infra.check_report_findings` pass the explicit invocation directory printed by
the CLI, never the shared base or an implicitly selected latest run.

Root `make check` currently checks only the root project; it is not a fleet-wide
receipt. Run a member's standard verb from the workspace root with
`make -f <member>/Makefile <verb>`, for example:

```bash
make -f flext-meltano/Makefile check
make -f flext-meltano/Makefile tests
make -f flext-meltano/Makefile build
```

The source template
`flext-infra/src/flext_infra/templates/project/base/Makefile.j2`, in its derived-paths
and repository-root sections, binds `PROJECT_ROOT` to the selected Makefile's
checkout rather than CWD. Attached members share the containing workspace's runtime
environment. Standalone checkouts and member worktrees without a containing
superproject own their runtime environment and run `make <verb>` from their own
root. Selecting the member Makefile adds no project selector or alternate dispatcher.
Manifest exclusions remain authoritative. Generated Make surfaces and documentation
are changed at their template or configuration owner, then regenerated with `make gen`.

Explicit new-project declarations let `make gen` reuse the existing scaffold service.
The target's project identity, kind, integration and provenance are validated before
effects; its Git/worktree is prepared separately. A missing package manifest in an
established repository is not permission to initialize Git or create a new project.
Generated settings use `FlextSettings.resolve_env_file` with the project's declared
namespace: an explicit `FLEXT_ENV_FILE` takes precedence over the namespace file and
the consumer's working-directory `.env`. Discovery never depends on the installed
package's location. Settings use the framework's per-class singleton; YAML business
configuration keeps its independent frozen lifecycle. Hosted setup selects the
existing non-editable installation path, so runtime acceptance must also verify that
the generated package is loaded from its own environment.
The generated API supplies the ping service's configuration and settings ports
explicitly. Its service base uses the Core kernel, and separate service instances
retain their injected settings without reading module globals. The CLI binds the
already-composed API alias.

`make mod` validates the rule fixtures and applies declared consumer text cutovers,
then invokes the existing Rope census before the structural fixed-point cycle. The census
derives public `c.*` paths from live facade declarations and inheritance, rewrites
qualified import bindings, and preserves the declaration parts composing `c`.
Config and settings bootstrap owners retain their dedicated default declarations.
Publication requires unchanged source identities; affected projects are checked
through the canonical lint, Pyrefly, and fresh-import gates.
The census classifies constants by their declaration contract before the containing
file's family, so constants in utilities, protocols, models, and services remain
visible as placement violations.

The single template catalog, `Infra.codegen.templates.entries`, binds each source,
destination, profile and ownership policy. Full-file outputs are never hand-edited.
Merged outputs preserve authored content outside managed regions; initial scaffold
files without recurring generation ownership remain authored after creation. Change
the declaration or generating source rather than introducing another catalog or writer.

Journal inspection, physical-effect admission, profiling and reporting belong to the
operation's existing engine. A preflight authenticates the actual journal and selected
physical scope without claiming completed generation. The publisher rechecks those
identities before effects. Profiling retains the same operation's outcome and raw
evidence; wall time and function timings remain distinct.

Acceptance exercises each required public operation explicitly against workspace
roots, selected members, independent checkouts and applicable CI contexts. Root
mutation fanout follows declared membership; tests, checks and local builds retain
their selected project scope. Product operations are exercised only through their own
declared commands. A composite wrapper or a sampled member cannot stand in for this
matrix.

Acceptance belongs to the candidate after its last effective structural, formatting,
generation or environment mutation. Read the native generation operation receipts and
their journal-backed artifact batches, including unchanged outputs, actual effect
targets, pending effects and source/output identities. Repeated generation must have no
relevant artifact effects; routine receipt/archive publication is evidence, not source
drift. An exit-zero invocation or a sample of unchanged files is insufficient: bounded
observations do not cover every generated artifact. Repeated `mod`, `fix`
and `fmt` also require zero relevant effects. A later effective change invalidates the
affected runtime and gate receipts and requires their renewed validation, never a hidden
retry or a whole-workspace quiet window.

Staged, unstaged and untracked work are normal shared-workspace inputs, not
admission failures. Check's existing Git observer reports that state without
demanding cleanup or a globally stable tree. It does not authorize discarding
changes or clearing recovery objects. Repair actual validation failures forward
and recapture only the affected inputs when they change.

Clean removes only declared disposable resources admitted by its existing owner.
It preserves authored or foreign WIP, nested repositories, environments, persistent
test state and active transaction journals or locks. A Git ignore rule alone does
not establish that a resource is owned and disposable.

Each command retains a typed receipt recording its selected repository, invoking
working directory, physical environment, applicability and original process outcome.
The declared scope and all planned rows remain in the report, including CI-inapplicable
members and dependencies without authenticated editable sources. Such rows are not
execution or a fleet pass. The first failure stops the operation;
unreached commands remain unexecuted, never successful. Attached members share
their physical superproject runtime, so attached validation does not substitute
for standalone consumer proof. Ordinary root verbs remain repository-local and
cannot, by themselves, establish a fleet-wide result.

The generated local `make check` includes the public
`flext-infra workspace verify-lanes --repo-root <checkout>` gate. A workspace
invocation covers its declared members as well as the root. This is detection only:
it does not fetch, prune, merge, stash, delete refs, remove worktrees, or publish
receipts. The coordinator synchronizes integration outside the guard. The guard
compares the explicit live remote integration OID to the available tracking object
and checks that the current lane consumes it.

`FlextInfraGitService.verify_lanes` resolves to the single `FlextInfraGitLanes`
evaluator. Its utility input, `u.Infra.git_lane_facts`, collects native stash OIDs
and local/configured-remote refs only; it neither elects a base nor judges lanes.
Partial native reads retain explicit errors which keep the evaluator red. The
public request is `GitLaneVerificationRequest` and the response is `GitLaneReport`,
including structured violation classes and complete textual findings. Clean linked
worktrees already contained in the same live integration are reported as residue,
never automatically retired. Checked-out refs are not exempted from that proof.

Failures include the full exact refs, stash OIDs, and registered worktrees. A branch
already contained in integration is residue; integration refs and symbolic remote
HEAD are protected. Unknown ownership is inconclusive, never inferred from commit
age or directory mtime. Open PRs and fresh, explicitly owned Bead lanes are allowed.
Inactive owned work requires correlated tracker, ref activity, and worktree evidence;
the activity window comes from the global governance SSOT configured by
`BranchPolicySpec.lane_governance_file`, never a copied threshold. The public
`--governance-file` option selects an explicit coordinator policy. Without valid
policy evidence, ownership/activity is inconclusive rather than assumed safe.
Optional `--evidence-file` selects a typed,
repository-bound coordinator receipt and requires freshness under that same policy.
Provider, tracker, policy, and Git read errors remain failures, not empty inventories.

Live auxiliary reads require explicit `--read-pull-requests` and/or `--read-beads`
selection. Installation never selects a capability. Dormant sources remain typed
unknowns and appear as `NOT EXECUTED` in the inventory; any artifact needing that
missing ownership proof remains inconclusive. Git facts still run independently.
Selected Beads reads consume `BeadsProjectSpec.ownership_command_prefix` and
`ownership_command_cwd` from the repository's `config/beads.yaml`, through the public
city/rig wrapper declared there. An undeclared or failed selected route fails;
the service appends the immutable read-only list protocol. There is no bare `bd`,
alternate database, runtime startup, or retry fallback.
An ownership receipt and live sources cannot be selected together.

A storage-only primary registry row is bound to a checkout only when native Git
proves the caller's Git directory equals shared storage and its top-level agrees
with GitPython's working-tree identity. The raw porcelain remains available as
evidence. An unprovable storage/checkout mapping fails rather than substituting
the request path.

The same Git service rejects stash entries and stale effective bases before a
worktree ADD creates directories or refs. REMOVE proves that the lane tip is an
ancestor of the live integration tip before delegating to the existing clean/index/
untracked preservation boundary. Neither ancestry nor a green inventory grants
retirement authorization. Direct shell/session/sweep enforcement is the separate
governance consumer of this owner, not a second Git hygiene engine.

## Related guides

- Development
- Testing
- Getting started
