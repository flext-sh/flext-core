<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/codemod-rules.md`; adjust that workspace source, never this member projection. -->

# flext-core - Codemod Rules

> Project profile: `flext-core`

<!-- TOC START -->

- [Catalogs and layers](#catalogs-and-layers)
- [Make verbs](#make-verbs)
- [ast-grep rules](#ast-grep-rules)
  - [Pyrefly strict-typing cures (campaign flext-osru9, census 2026-10-28)](#pyrefly-strict-typing-cures-campaign-flext-osru9-census-2026-10-28)
  - [One relocation law: every declaration lives in its family](#one-relocation-law-every-declaration-lives-in-its-family)
  - [One construction owner: the model, never the protocol](#one-construction-owner-the-model-never-the-protocol)
  - [One rule per law](#one-rule-per-law)
- [Text rules](#text-rules)
- [Tests and snapshots](#tests-and-snapshots)
- [Workflow](#workflow)

<!-- TOC END -->

FLEXT applies structural migrations through the codemod engine that flext-infra owns.
The provider-neutral authoring procedure (one law per rule, generalization, exactness,
consolidation) is the `codemod` skill. This guide covers the FLEXT engine:
where rules live, the keys it reads, and the verbs that run it. Every value named here
is declared by the flext-infra owner cited next to it; that owner wins over this guide.

The exhaustive-enum migration is an accepted target, not a validated rollout.
An exhaustive `Literal` of members from one `StrEnum` may become that enum's direct
Pydantic field annotation only after native Rope identities prove its complete
canonical domain and references prove that it is not a discriminator. Preserve
`Annotated` metadata, defaults, aliases and proper subsets. Unresolved ownership,
mixed domains or unproved safety fail before effects, rather than disappearing
from the accounted migration. The current rule must complete that proof and
public-boundary coverage before application can establish acceptance.

Metadata protection follows the annotation's native declaring scope. A local
binding in a function body must not shadow `Annotated` or `Literal` in that
function's header. Imported aliases and generic declaration contexts retain
their native identities; lexical spelling alone cannot authorize a rewrite.

The latest operator revision elects `use_enum_values=False` for governed presets,
superseding the intermediate primitive-storage contract (ADR-005 and ADR-028
Decision 14).
Validation retains the input domain and the effective preset's strictness;
validated `StrEnum` attributes retain enum members and the enum type in generated
protocols. Writable properties retain their accepted enum input type. Python construction,
default validation, JSON ingress, output values and consumer migration are separate proofs;
the annotation alone does not establish any of them.

## Catalogs and layers

Layer-zero own-helper admission is derived from the same native candidate proof
used by import normalization (ADR-005 §2 and ADR-028 Decision 1). The declarative
producer captures the complete workspace sources and asks that owner to authenticate
each imported helper and its dependency closure, including type-checking imports.

Only proven exact module identities enter the generated constraint; a regex does
not establish ownership or acyclicity. For those layer-zero owners, own root and
family imports remain forbidden. Family declarations composing upstream same-letter
facades follow ADR-028 Decision 10; they are not layer-zero self-imports.
Generation and native-scan agreement require runtime validation after source changes.

| Layer | ast-grep rules | Text rules |
|---|---|---|
| Universal (every project) | `flext-infra/src/flext_infra/config/rules/ast-grep/` (`sgconfig.yml` with `scope: universal`) | `flext-infra/src/flext_infra/config/rules/mod/sed.yaml` |
| Runtime (consumers of a provider) | an installed package's `sgconfig.yml` with `scope: runtime` | none |
| Project overlay | `<repo>/config/rules/ast-grep/sgconfig.yml` | `<repo>/config/rules/mod/sed.yaml` |

The engine composes the providers in that order:
`flext_infra/_utilities/codemod_rules.py`, `codemod_rule_plan`. Rule ids are unique across
every loaded provider. A text rule may also name the projects it targets in
`distributions`; an empty list means every project (`codemod/text_gates.py`,
`_selected_rules`). Put a rule at the highest layer where its law holds.

## Make verbs

Declared signature additions use native Rope `ChangeSignature` through the
existing contract planner and guarded publisher. Their source changes are not
accepted until the native lifecycle verifies the declaration and every caller.

Distribution-name normalization belongs to `u.Infra.distribution_identity`.
The declared Rope migration targets proven one-argument calls and single-iterable
`map` callbacks while retaining the native backend in its implementation owner.
Calls with other arguments, including `validate=True`, retain their contracts and
require their own evidence before migration.

Common verbs are declared in
`flext-infra/src/flext_infra/config/codegen.yaml` (`make.verbs`). Infra declares
its owned snapshot operation in `flext-infra/config/codegen.yaml`
(`repository.extra_verbs`), with the native handler in `flext-infra/custom.mk`.

| Surface | Effect |
|---|---|
| `make mod` | Apply ast-grep, semantic and text rules to a proven fixed point; committed snapshots are verified, never rewritten. |
| `make check` | Run the root's declared read-only admission and retain native findings. |
| `make tests` | Exercise the public transformation and consumer contracts through the persistent incremental test owner. |

Snapshot refresh has an existing Infra extension declaration, `mod-snapshots`,
whose handler invokes the native refresh owner with the repository root and
apply flag. In workspace mode, execution must be routed through the workspace
root Make owner. Its composed public route requires generation and native
verification before it becomes an instruction to execute. Missing routing is a
defect in that owner, to repair using this same operation. Preserve snapshot
acceptance while repairing it; no consumer owns another snapshot engine or
edits the generated Makefile.
Fixture refresh elects the authored catalogs through the existing provider
composition. Every rule is authored and carries fixtures; no rule is rendered
(ADR-028 Decision 10).

The catalog applies through ast-grep's own `scan --update-all`; the pipeline order,
native tool use and validation in the real tree are ADR-028 Decisions 11-13.
Authored fixtures exercise draft rules through the root's native test and
refactoring owners; real consumer behavior and the completed root cycle prove
the transformation.

The pipeline's rule stage runs the authored catalog over one staged candidate
(ADR-028 Decisions 13 and 18): the import rules, then every other rule, repeated
until no byte changes. Rendering emits the final import order; no Ruff stage decides
or repairs imports. Each stage's staged `sgconfig.yml` lives under
`.reports/codemod/native/<stage>`, so `ast-grep scan -c` reproduces the stage by hand.

Machine-only native configurations use JSON-form YAML documents, serialized by
the existing typed JSON owner. Elected scan rules share one multi-document YAML
file; global utilities retain one document per file as ast-grep requires.
Computed fact utilities are serialized into the staged scan by the same typed JSON
owner; they carry facts only (names, paths, regular expressions of published
identities), never a pattern or a law of their own.
Authored catalogs, fixtures and snapshots keep their existing source format.
This avoids runtime-value normalization and one file write per elected scan
rule without changing native rule identities, scopes, bindings or diagnostics.
Parsed authored catalogs are reused only in process memory, keyed by provider,
resource paths and exact current contents. Planning always authenticates the
current files before selecting that entry. Infra does not serialize its parsed
catalog into a separate persistent cache; native tool caches keep their own
contracts.

Native parameterized utilities preserve their formal argument names and qualify
only provider-global callees, including mapping-form `matches` calls. The
`captured-rule` utility evaluates one complete predicate with its bindings and
exports its named captures through ast-grep's native argument mechanism. Relational
context stays inside that native predicate instead of repeating whole enclosing
classes in every diagnostic. Primary match ranges, binding equality, messages and
severity remain owned by the original rule; the complete native output is retained.

The engine computes each governed project's facts (own package, facade packages,
publication, layer rank, elected upstream, runtime aliases, runtime reads and the
import-cycle graph) and binds them into the staged scan as fact utilities that
authored rules reference through `matches:`. `u.Infra.codemod_rule_plan` composes the
authored catalogs with those fact utilities into one staged native provider per
repository. The workspace coordinator visits repositories in dependency order in one
process, and each stage scans only its repository's owned sources. `ast-grep scan -c`
on the staged provider reproduces the run.

| Workspace declaration | Infra implementation |
| --- | --- |
| ADR-028 Decision 17 requires one snapshot/planner per repository. | [Codemod native utilities](https://github.com/flext-sh/flext-infra/blob/0.12.0-dev/src/flext_infra/_utilities/_codemod/native_utilities.py) |
| ADR-028 Decision 18 owns one generalized source rule per law. | The authored catalog is the single rule owner; repository plans bind computed facts to it. |

Every finding is the native match; no Python admission runs after the scan. A
project fact a catalog rule needs is a computed fact utility (`own-package`,
`stdlib-module`, `runtime-package`, `facade-package`, `local-alias`,
`runtime-alias`, `package-published-import`, `compose-module`, ...) that the rule references
through `matches:`, or a computed fact selector bound through `scope`. Catalog
fixtures resolve those identities through
`config/rules/ast-grep/fixture-utils/`, which only the fixture stage reads. A native
rule that still reports an actionable finding after the rule stage fails `make mod`.

The deep-import law has one rewrite rule, `rewire-deep-import-to-package`: a
`from <package>.<module> import ...` moves to `from <package> import ...` only
when the `package-published-import` utility proves that the package's initializer
publishes every imported name and that no sibling import of the package's modules
in the file imports an unpublished name. The header of a package's members moves
together or stays untouched; a package without an initializer publishes nothing
and never rewrites. The detection rules `ban-test-deep-module-import`,
`ban-consumer-submodule-import` and `ban-own-family-internal-import` report what
the rewrite cannot repair.

## ast-grep rules

Protocol field compatibility belongs to the existing canonical Pyrefly check,
not a syntax-only rewrite. The Python typing specification permits readable,
writable protocol attributes with compatible invariant types; a concrete field
of the same type can satisfy that contract. A read-only property remains valid
when the actual API requires read-only access or covariant return typing.

The `settings-protocol-branch-property` rule is retired with its fixtures and
snapshots: PascalCase namespace spelling and a `Settings(Protocol)` declaration
do not establish a variance error or a read-only API requirement. Its universal
claim that no concrete settings class satisfies a writable field was false.
Neither a replacement name exclusion nor an N802 exemption repairs that claim.
Ruff continues to validate getter names, including protocol properties, without
the retired rule's N802 identity exemption. UP040 publication policy is separate
and unchanged. This preserves ADR-028 Decision 8: transformations must not create
new lint or type violations. See the [typing specification](https://typing.python.org/en/latest/spec/protocol.html#protocol-members).

The universal `rewire-test-assert-*` family implements the
workspace assertion standard.
Its matcher owner is `flext-tests`; its patterns, native fixtures, and generated
snapshots have one writable owner in `flext-infra`. Consumers reexport the same
`tm` binding rather than copying matcher or transformation implementations.
Rules must preserve Python's evaluation and scope semantics, not merely remove
the `assert` token. In particular, narrowing and lazy-message guards use the
terminating public matcher without rebinding a referenced name or changing the
exception category. Runtime comparisons against the original snippets validate
these properties; snapshot text alone does not prove them.

### Pyrefly strict-typing cures (campaign flext-osru9, census 2026-10-28)

Fleet pyrefly findings reduce to a small set of root causes; each has one
canonical cure, applied at the origin and never by suppression or rule change
(the lint-rule SSOT stays frozen):

- **`implicit-any-lambda`** — Pyrefly cannot infer a lambda parameter from a
  generic `Callable[[SubjectT], bool]` parameter (`tm.that(..., where=...)`,
  `tm.ok(..., where=...)`, `Result.map(...)`). Cure: replace the lambda with a
  local typed function (`def pred(value: <concrete subject type>) -> bool`)
  and pass it by name. Lambdas cannot carry parameter annotations in Python,
  so no call-site annotation exists; the typed def is the strict form.
- **`missing-attribute` / `missing-module-attribute` on consumed APIs** — a
  consumer names an attribute that a refactor removed or moved (for example
  `u.chk`, `cli.ensure_dir`). Cure: rewire the consumer to the current owner
  found by source search; never recreate the removed attribute.
- **`missing-module-attribute` / `missing-import` on generated packages** — a
  stale generated lazy `__init__.py` re-exports modules or letters that moved.
  Cure: regenerate at the owner (`make gen`); hand-editing a managed
  projection is prohibited.
- **`invalid-annotation` on test fixtures** — a fixture declares
  `self: SomeSettings` while its test class does not subclass that settings
  type. Cure: drop the false self-type and inject legitimate settings
  (conftest fixture or the canonical settings constructor).
- **Bare generic containers** (`list[dict]`, `dict` literals as data
  contracts) — cure with a Pydantic model per payload shape (see the root
  `scripts/modularize_*` FleetMove/ModularizationMove repair) plus explicit
  element-type annotations on every empty container.
- **AST optionality** — `node.end_lineno`, `node.end_col_offset` and friends
  are `int | None`; guard with an explicit `if ... is None: continue` before
  arithmetic instead of asserting or casting.

The railway-result and rewire-test-assert families above remain the model for
turning each cure into one generalized, budgeted rule once the census reaches
zero; detection-only candidates first, with `expected` receipts.

The universal `railroad-result.yml` rule family implements Railway-Oriented
Programming (ROP) normalization over `FlextResult` (`r`): terminal `if res.failure:
return ...` passthrough guards collapse to `return res`, single-argument transform
and chain tails collapse to `.map(...)` and `.flat_map(...)`, ternary/branch
unwraps collapse to `.unwrap_or(...)` and `.map_or(...)`, and manual
`model_validate` `try`/`except` blocks collapse to `r[M].from_validation(...)`.
Because the native `ast-grep` cascade repeats up to `mod.phases.pass-cap`,
consecutive failure guards at the end of a function collapse bottom-up across
passes without custom AST walkers, and unused local `outcome = r[...]` aliases
and their orphaned imports are pruned automatically.

A rule file holds one or more YAML documents (`---`). Each is standard ast-grep
(`id`, `language`, `severity`, `message`, `files`, `rule`, `constraints`, `utils`
references through `matches:`, `transform`, `fix`). Shared guards live in
`config/rules/ast-grep/utils/`.

The engine reads these `metadata` keys, declared in
`flext-infra/src/flext_infra/_constants/refactor.py` (`CODEMOD_RULE_*`):

| Key | Meaning |
|---|---|
| `scope` | File scopes as computed fact selectors: `{in: [selector], not-in: [selector]}` narrows the rule's `files`/`ignores`. No rule is partitioned or rendered per package. |
| `owner` | The distribution that owns the law; that project's own plan drops the rule. |
| `consumers_of` | The rule applies only to projects whose runtime closure includes that distribution. |
| `relocation` | A detection-only rule names the Rope relocation that repairs it and captures the symbol as `$NAME`. `family-declaration` also captures the owning facade letter as `$FAMILY` (see below). |
| `expected` | The exact finding count receipt; any other count fails. |
| `owned_path_suffix` | Reject owned `.pyi` paths from discovery even without a syntax tree; detection-only, with no exemption. |

### One relocation law: every declaration lives in its family

ADR-028 Decisions 3-4 give each kind of declaration one owner family: type
aliases, `TypeVar`s and other typing declarations in `t`, constants in `c`,
`Protocol` classes in `p`, models in `m`, reusable behavior in `u`. The
`family-declaration` relocation is the single executor of that law
(`refactor/namespace_relocations.py`); a rule only declares the kind, its
`$NAME`, and the owning letter as `$FAMILY`. The rules that name it are
`ban-manual-typing-alias-outside-typings` (`t`),
`ban-manual-protocol-outside-protocols` (`p`),
`ban-class-wrong-facade-layer` (`c`, `m`, `u` by the class kind) and
`ban-behavior-in-declaration-layer` (`u`).

- **Destination.** The module of the owning family named after the declaring
  module, inside the namespace root that hosts the family directories:
  `_models/matchers.py` relocates a typing to `_typings/matchers.py`
  (`u.Infra.family_declaration_target_file`; the directory and the letter come
  from the facade-family census). An absent destination is scaffolded; the
  class-nesting cutover then nests the moved declaration in the module's single
  family class and the facade composition publishes it by MRO
  (`t.Tests.MatchExpectedValue`).
- **Consumers.** The native Rope move rewires every consumer, in other members
  and external consumers included.
- **Private names.** The family namespace has no privacy, so `_Name` is
  renamed to `Name` through the native Rename before the move. A collision at
  the source or the destination, or a quoted forward reference to the name,
  stops the relocation with the first failure.
- **Behavior.** A module-level function moves whole. A static or class method
  is transferred into the destination owner class by the native class transfer
  and rejected whole when a reference is unproven (the rejection is logged and
  the finding stays). Instance methods, properties and dunders need a receiver
  rewrite that no derivation supplies and stay findings.
- **Nesting yields.** Class nesting leaves a typing declaration outside the
  typings family, and a function in a constants, typings or models module,
  at module level so this relocation finds it.

### One construction owner: the model, never the protocol

A `p` member is a structural Protocol: it describes a contract and never builds
or validates data. `ban-protocol-constructor` is the single rule of this law and
covers every form at any namespace depth: `p.X(...)`, `p.NS.X(...)` and the
model classmethods `.model_validate`, `.model_validate_json`,
`.model_validate_strings` and `.model_construct` on a `p` path. A protocol
read in an annotation, `isinstance` check or `match` class pattern is the
intended use and never fires; only a call whose callee is the `p` path does,
and only in a module that binds `p` from a FLEXT package. The runtime failure
this prevents is real: a protocol class has no `model_validate`.

The rewrite is exact and derived, never listed. The computed fact utility
`protocol-model-path` (producer `codemod_requested_utilities`, the same
request mechanism as `generic-specialization`) holds the dotted `p` paths that
a reached facade package publishes as a model (`FacadeMemberKind.MODEL`) at the
same path in `m`, read from the native source of both letters. A path that any
reached package publishes without a model twin is excluded. The rule's
rewriter retargets only that path from `p` to `m`, so a protocol without a model
at its path keeps its text and stays a finding for its owner: the rule reports
it and never guesses. The unbound `m` the rewrite introduces is bound by the
generated `import-header` stage of the same `make mod` fixed point. The
fixture value of the utility is `fixture-utils/protocol-model-path.yml`; the
rule test covers the constructor, validation, nested-path, no-model, annotation,
`isinstance` and unbound-facade cases.

See using-pydantic for the
protocol contract this rule enforces.

### One rule per law

| Workspace declaration | Accepted decision |
| --- | --- |
| This section, One rule per law; the generated Infra copy links back here. | FLEXT ADR-028 Decision 18 |

For semantic discovery and refactoring, ADR-028 Decision 17
uses Rope's native workspace and refactorings. The later
ADR-018 projection correction
assigns declaration-only generation to one typed AST inventory; generation does
not perform semantic moves or consumer rewrites. ast-grep rules, text operations
and generated projections retain their declared owners.

Operator order of 2026-10-25, verbatim (tracker memory
`operator-order-2026-10-25-rules-generalized-min-count`; ADR-028 Decision 18):
"o sed, rope e ast-grep rules tem que estar sempre alinhados com as regras
corretas de forma generalizada e automaticas usando sempre máximo de
funcionalidades padrões das ferrametnas que as embasam e reduzir ao máximo ao
número de rules que são executadas". sed, Rope and ast-grep rules stay aligned
with the governing rules. They are generalized and automatic, use as much of each
tool's standard functionality as possible, and keep the number of executed rules
as small as possible. Renames and moves that Rope's refactorings express run
through them, not through a custom planner (ADR-028 Decision 17), and a rename of
library-named exports to capability names follows ADR-028 Decision 15.

#### One pipeline, one owner per law

ADR-028 Decision 18 item 2
(operator orders 2026-11-02; epic `flext-lqfqo`) makes `check`, `gen` and `mod` one
pipeline with one rule set. Each law is owned by exactly one stage or generalized rule,
perfected with fixtures and the native test, and applied through the pipeline;
duplicates are exterminated. `--laws <set>` diagnoses or repairs one law; the import law
always runs. Violating sources are never edited by hand. The law sets below are tracked
under bead `flext-brr3y`:

| Theme | Rules (unified into one law) | Bead |
| --- | --- | --- |
| T0 law selector | engine: `--laws`, per-repo expansion by `enforce` | `flext-brr3y.4` |
| TH1 protocol construction | `ban-protocol-constructor` | `flext-brr3y.5` |
| TH2 letter binding, import header | the authored letter-binding rule (import header, own letter, duplicate and foreign origins, module path, module object, aliases); `add-facade-module-*-alias`; missing-import companions | `flext-brr3y.6` |
| TH3 protocol projection | model-to-protocol generator; holder composition; duplicate retirement | `flext-brr3y.7` |
| TH4 Pydantic ConfigDict | `config-dict-type-from-typings`, `rewire-relaxed-*` | `flext-brr3y.8` |
| TH5 deep import | `rewire-deep-import-to-package`, `ban-consumer-submodule-import`, `ban-test-deep-module-import` | `flext-brr3y.9` |
| TH6 private import | `ban-private-import`, `ban-own-family-internal-import`, `ban-test-own-private-import` | `flext-brr3y.10` |
| TH7 pytest to tests facade | `rewire-pytest-*`, `rewire-test-monkeypatch-*`, `rewire-test-assert-*` | `flext-brr3y.11` |
| TH8 family placement | `namespace-law`, `namespace-structure`, layer and placement bans, `relocate-facade-*` | `flext-brr3y.12` |
| TH9 `m.` in type positions | `rewire-model-protocol-field-to-port`, `ban-model-typing-in-protocol-projection` | `flext-brr3y.13` |
| TH10 exceptions to axioms | exemption and ignore owners | `flext-brr3y.14` |

- **One authored rule per law.** Facts that vary per package, letter or namespace
  are bound as fact utilities and parameters of one authored rule, never rendered
  into rule documents. A per-namespace variation is an `any` branch or a
  `rewriter` of that one rule, selected through the bound fact.
- **File scope by fact selectors only.** ast-grep scopes a whole document by
  `files`/`ignores`; `scope: {in: [...], not-in: [...]}` binds a computed fact
  selector as native `ignores`. A law is stated from the code itself; no rule is
  split per package or file group.
- **One law, one rule, every library.** A law that applies to many libraries,
  packages or letters derives the set from its facts (installed distributions,
  facade exports, the import law). It never ships one clone per member.
- **Retired migrations leave.** A one-shot migration rule (`retire-*`,
  `rewire-*`, `remove-*`, `rename-*`) whose fleet census is zero is deleted with
  its test and snapshot.
- **Executed rules are a gate metric.** The number of rules a `make mod` run
  executes per project is reported from `mod-findings.json` (`totals_by_rule`).
  Its budget is a typed configuration key, never a number in code.

These laws are authored rules over computed facts, each with fixtures (epic
`flext-lqfqo`):

- **Letter binding and import header.** A name a module uses but binds nowhere is
  bound from the package that publishes it, grouped into one statement per package;
  the own letter, competing and duplicate origins, module-path and module-object
  reads are rewritten to the binding ADR-028 Decision 10 elects.
- **Constant type sets.** `isinstance(value, <set>)` outside the constants family,
  when the set is exactly a declared tuple-of-classes constant, is replaced by that
  constant. The constant is elected by value from the facade's facts; two unmarked
  constants with the same set fail with both paths.
- **No deprecation window in this pipeline.** Every consumer of a renamed or moved
  name in the governed roots is rewired in the same cutover, so no pass-through or
  compatibility read survives.

A package initializer never publishes a child package object. Consumers import a
child's names from the child package (ADR-028).

The predicates (`stdlib-module`, `own-package`, `runtime-alias`, `emitted-symbol`,
`member-relocation` and the rest) and their argument contracts are declared in
`CodemodContextPredicate` in the same module. Prefer a predicate over a literal list:
it derives the set from its source of truth.

`concrete-mixin-runtime-ellipsis` also rejects concrete class-local typing shells
whose bodies contain only ellipsis or discarded parameters, including the
`del`/`NotImplementedError` variant. Recompose the real implementation and state
owners instead of moving signatures into another typing-support class. Its
`protocol-declaration` and `resolved-symbol` predicates retain real Protocols and
resolve the typing guard; `declaration-decorator` compares the captured native
function's decorators with its configured module/symbol pair, preserving aliased
ABC and implemented overload declarations without a spelling-based exemption.
Ordinary type-only imports, type aliases and productive method bodies are not
shells. Generated headers do not exempt candidates.

The single-nested-class family (`ban-loose-module-object`, `require-module-class`,
`declare-single-class-export`, `ban-pytest-hook-outside-conftest`) binds every
Python file of the repository: src, tests, scripts, examples and docs code. A
module holds one top-level class, its canonical published bindings, `__all__` and
imports. The exception data is closed and each entry has a fixture: an
executable module's guarded `main()` entry point and generator-emitted dunders
(generator law p.11), plus the tests-tier delta of ADR-028 p.6 — pytest
fixtures, `pytest_*` hooks and `pytest_plugins` at module level, admitted only in
`conftest.py` and `tests/fixtures/**`. Ordinary test modules are not exempt.
Test-suite modules follow the same publication law as `src`: one `Tests<Stem>`
class plus its `__all__`, enforced by `declare-single-class-export` with no path
exemption. Any further exception needs the operator.

`ban-stub-file` is also admitted by owned path, independently of ast-grep matches.
Both `mod` and the codemod `check` gate consume `discover_python_files` through the
same source inventory and common admission. Empty, invalid, generated-header and
outside-`src` stubs remain findings.
Hidden or analysis-ignored descendants of declared source roots remain owned
stub-policy inputs without widening ordinary `.py` analysis. Stub leaf links keep
their own path spelling and do not grant permission to follow foreign targets.

Artifact declarations marked
`retired_typing_root` elect retired first-party typing trees even when ordinary
source scans ignore them; environment, installed dependency, foreign Git and
unmanaged submodule boundaries remain excluded. Reports distinguish `owned-path`
evidence from `ast-grep` evidence and retain authenticated source states.

The typed tooling document rejects stub options and producers in native pairs and
derived declarations. Its canonical loader checks rooted native/topology search paths
once against the elected project and actual installed interpreter roots, without
banning source import paths or installed third-party typing. `CodegenFilePlan` rejects
non-absent `.pyi` output, and authenticated
publication revalidates the same model law before effects, including unchecked copied
plans. Retirement uses `CodegenFilePlan` with absent desired bytes and mode;
`SemanticFilePlan` with no desired content remains a no-op, not a deletion.

## Text rules

`sed.yaml` holds `rules:`, a list of entries. The keys are the `CODEMOD_TEXT_KEY_*`
constants in `_constants/refactor.py`; any other key fails:

| Key | Meaning |
|---|---|
| `id`, `description` | Unique identity and the law it states. |
| `find`, `replace` | Python `re` pattern and replacement (`\g<name>` groups). |
| `flags` | Any of `IGNORECASE`, `MULTILINE`, `DOTALL`. |
| `include`, `exclude` | Repository-relative globs. Markdown targets must be included explicitly. |
| `distributions` | Project names the rule targets; empty means all. |
| `expected` | Exact match count receipt. |
| `capture_equals` | Required equality between captures. |

The engine refuses findings in generated files (fix the generator instead), and a
rewritten Python file must still parse (`codemod/text_gates.py`). Select native
Rope bindings, references and refactorings when a transformation depends on
identity, scope, inheritance or coordinated consumer changes. Existing purely
syntactic rules retain their ast-grep owner until their own complete migration;
FLEXT-owned CPython AST and LibCST interpretation are being retired separately.

## Tests and snapshots

- Every ast-grep rule has `tests/<id>-test.yml` with `valid` and `invalid` cases, plus a
  snapshot under `tests/__snapshots__/<id>-snapshot.yml` for each invalid case.
- `make mod` verifies them and fails on a changed fix output, a missing snapshot, or a
  snapshot of a removed rule or case.
- The existing snapshot-refresh owner performs regeneration through its
  verified root Make route. Review its diff in the same commit as the rule;
  snapshot refresh never proves behavioral acceptance.
- Invalid fixtures retain the rejected input, not its corrected output. For
  adapter construction, `m.TypeAdapter(...)` and `t.TypeAdapter(...)` exercise the
  rewrite, while `u.type_adapter(...)` is a valid fixed point. Regenerate snapshots
  from those authored inputs; never repair stale labels by editing generated output.
- A fixture asserting a semantic class match must contain valid Python; malformed
  source belongs to a separate parser-boundary case.

## Workflow

1. Reconcile the owning ADR, existing Bead, plan and catalog before code.
2. Extend the existing native owner and public-boundary fixtures, preserving
   positive, negative, metadata and consumer cases.
3. Refresh owned snapshots through the verified root Make route and review the
   diff. Repair that route first if it is unavailable; preserve the same owner.
4. Qualify the pattern on its admitted pilot and consumer closure before a
   mutating workspace rollout.
5. Complete the root cycle `gen → fmt → fix → mod → check → tests`, plus applicable
   runtime and build proof, in the coordinated writer windows.
6. Integrate the rule, fixtures, snapshots, rewrites and evidence together.
