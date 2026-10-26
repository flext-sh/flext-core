<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `<workspace-root>/docs/guides/codemod-rules.md`; adjust that workspace source, never this member projection. -->

# flext-core - Codemod Rules

> Project profile: `flext-core`

<!-- TOC START -->

- [Catalogs and layers](#catalogs-and-layers)
- [Make verbs](#make-verbs)
- [ast-grep rules](#ast-grep-rules)
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

## Catalogs and layers

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

Common verbs are declared in
`flext-infra/src/flext_infra/config/codegen.yaml` (`make.verbs`). Infra declares
its owned snapshot operation in `flext-infra/config/codegen.yaml`
(`repository.extra_verbs`), with the native handler in `flext-infra/custom.mk`.

| Surface | Effect |
|---|---|
| `make mod` | Apply ast-grep, semantic and text rules to a proven fixed point; committed snapshots are verified, never rewritten. |
| `flext-infra refactor mod-text` | Replay only the authenticated declarative text rules. |
| `flext-infra refactor ast` | Scan report of the ast-grep and sed cascades over the governed corpus; `--apply` reaches the mechanical fixed point. |
| `make mod-snapshots` (Infra root) | Regenerate every owned rule-test snapshot through the existing native refresh owner, for a reviewed commit. |

`make mod-snapshots` is an Infra extension, not an implicit part of `make mod`.
Its handler invokes the existing `flext-infra refactor mod-snapshots` operation
with the repository root and apply flag. The source declaration requires native
generation before the verb is available. No consumer owns another snapshot
engine or edits the generated Makefile.
Fixture refresh elects authored catalogs through the existing provider
composition directly. Generated rules have no fixtures, so refresh does not
build a source engine or render project-dependent rules to discover their
authored owners.

The catalog applies through ast-grep's own `scan --update-all`; the pipeline order,
native tool use and validation in the real tree are ADR-028 Decisions 11-13. A draft
rule is prototyped with `ast-grep scan --rule <draft> <scratch-copy>`, and its proof is
the real tree under `make mod`.

`make mod` runs the native cascade first. The owner is
`flext_infra/_utilities/codemod_native_cascade.py`, called from
`codemod/batch_apply.py`. The cascade covers every fixable rule that is neither
an owned-path rule nor a Rope relocation, plus the generated rules, and runs the
`mod.phases.stages` declared in `tooling.yaml`:

1. `header`: the generated `import-header` rules.
2. `catalog`: every other native rule.
3. `import-fix`: Ruff `F401`/`I001`.

The stages repeat over one staged preview until no byte changes. Each stage's
staged `sgconfig.yml` lives under `.reports/codemod/native/<stage>`, so
`ast-grep scan -c` reproduces the stage by hand.

Machine-only native configurations use JSON-form YAML documents, serialized by
the existing typed JSON owner. Elected scan rules share one multi-document YAML
file; global utilities retain one document per file as ast-grep requires.
The generated scope manifest is `scopes.json` and is read by the JSON owner:
selector keys may exceed YAML's implicit-key length limit. Its declared filename
participates in the provider digest, so the format change elects a new immutable
provider rather than reinterpreting a previously published YAML manifest.
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

The producers and renderers registered in `Infra.codegen.ast_grep` render each
governed project's generated utilities and rules (`codemod_native_generated`).
`u.Infra.codemod_rule_plan` materializes them as one native provider,
`generated`, under `.reports/codemod/native/generated/<content digest>/`
(`sgconfig.yml`, `ruleDirs/`, `utilDirs/`). It then elects that provider after the
catalogs, like any other provider. Generated rules retain their renderer IDs.
Each repository has its own plan and native stage; the workspace coordinator visits
repositories in dependency order in one process. Each stage scans only its
repository's owned sources; declared members enter their own stages.
Generated providers also carry
utilities and file scopes when they contribute no elected rules, so catalog rules can
still consume those declarations. Each project's scan, `check`
codemod gate and cascade consume its own generated provider through the same plan as its
authored rules. `ast-grep scan -c` on that project provider reproduces the rule set.
Generated rules carry no fixtures: renderer tests own their proof. The retired
per-rule
repository projections under `config/rules/ast-grep/{utils,rules/generated}` are
declared at the generation owner; authored files are never retired by that declaration.

| Workspace declaration | Infra implementation |
| --- | --- |
| [ADR-028 Decision 17](https://github.com/flext-sh/flext/blob/0.12.0-dev/docs/architecture/adr/028-facade-boundary-orchestrators-declarative-enforcement.md#decision-17) requires one snapshot/planner per repository. | [Codemod native utilities](https://github.com/flext-sh/flext-infra/blob/0.12.0-dev/src/flext_infra/_utilities/codemod_native_utilities.py) |
| [ADR-028 Decision 18](https://github.com/flext-sh/flext/blob/0.12.0-dev/docs/architecture/adr/028-facade-boundary-orchestrators-declarative-enforcement.md#decision-18) owns one generalized source rule per law. | Repository plans preserve renderer IDs; the source catalog remains the single rule owner. |

Every finding is the native match; no Python admission runs after the scan. A
project fact a catalog rule needs is a stable generated utility (`own-package`,
`stdlib-module`, `runtime-package`, `facade-package`, `local-alias`,
`runtime-alias`, `package-export`, `compose-module`, ...) that the rule references
through `matches:`, or a file scope the generated provider publishes in its
`scopes.json`. Catalog fixtures resolve those identities through
`config/rules/ast-grep/fixture-utils/`, which only the fixture stage reads. A native
rule that still reports an actionable finding after the cascade fails `make mod`.

## ast-grep rules

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

A rule file holds one or more YAML documents (`---`). Each is standard ast-grep
(`id`, `language`, `severity`, `message`, `files`, `rule`, `constraints`, `utils`
references through `matches:`, `transform`, `fix`). Shared guards live in
`config/rules/ast-grep/utils/`.

The engine reads these `metadata` keys, declared in
`flext-infra/src/flext_infra/_constants/refactor.py` (`CODEMOD_RULE_*`):

| Key | Meaning |
|---|---|
| `scope` | File scopes the generated provider publishes: `{in: [selector], not-in: [selector]}` narrows the rule's `files`/`ignores`; `partition` renders one generated rule `<id>--<package>[--<key>]` per file group, and selecting the catalog id selects every partition. |
| `owner` | The distribution that owns the law; that project's own plan drops the rule. |
| `consumers_of` | The rule applies only to projects whose runtime closure includes that distribution. |
| `relocation` | A detection-only rule names the Rope relocation that repairs it and captures the symbol as `$NAME`. |
| `expected` | The exact finding count receipt; any other count fails. |
| `owned_path_suffix` | Reject owned `.pyi` paths from discovery even without a syntax tree; detection-only, with no exemption. |

### One rule per law

| Workspace declaration | Accepted decision |
| --- | --- |
| This section, One rule per law; the generated Infra copy links back here. | [FLEXT ADR-028 Decision 18](https://github.com/flext-sh/flext/blob/0.12.0-dev/docs/architecture/adr/028-facade-boundary-orchestrators-declarative-enforcement.md#decision-18) |

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

- **One document per law.** Facts that vary per package, letter or namespace
  are rendered into the values of a single document, never into one document
  per value. Those values are the regex of `constraints` on metavariables,
  `any` branches, rule-local `utils`, and `rewriters` selected by a `rewrite`
  transform. `relocate-facade-component--<package>` is the pattern: one
  document with one `any` branch and one rewriter per facade namespace.
- **Partitions only where the file scope differs.** ast-grep scopes a whole
  document by `files`/`ignores`; one branch cannot have its own file scope. A
  `partition` is used only when the verdict depends on which file group holds
  the match. When the law can be stated from the code itself, state it there
  and use `scope: {not-in: [...]}`, which renders as plain native `ignores`.
  `ban-foreign-namespace-source`, `ban-foreign-canonical-alias-import` and
  `ban-foreign-namespace-annotation-source` are single documents that way:
  facade modules compose upstream letters by design, so they are scoped out
  rather than partitioned per letter.
- **One law, one rule, every library.** A law that applies to many libraries,
  packages or letters derives the set from its facts (installed distributions,
  facade exports, the import law). It never ships one clone per member.
- **Retired migrations leave.** A one-shot migration rule (`retire-*`,
  `rewire-*`, `remove-*`, `rename-*`) whose fleet census is zero is deleted with
  its test and snapshot.
- **Executed rules are a gate metric.** The number of rules a `make mod` run
  executes per project is reported from `mod-findings.json` (`totals_by_rule`).
  Its budget is a typed configuration key, never a number in code.

Generated rules come from renderers registered in `Infra.codegen.ast_grep.renderers`.
Each renderer receives one facts model.
`codemod_package_export_facts` supplies, for every package: its `__all__`, its
directory, and its child modules and packages. Two renderers consume it:

- `codemod_import_header_rules` emits the `import-header--<package>[--<scope>]`
  rules.
  - Trigger: the module uses a declared name that it binds nowhere. Bindings are
    judged by the universal `name-binding` utility, embedded as each rule's local
    utility.
  - Effect: one `from <package> import ...` line is inserted after the module's
    first import, carrying every name the package resolves in that scope. The
    `import-fix` stage prunes the unused names.
  - Package choice: a module inside a declaring package imports from the deepest
    declaring package that contains it. Any other module imports from the unique
    shallowest declaring package. A name declared by more than one
    shallowest package is never inserted.
- `codemod_module_member_rules` emits the `module-member--<package>` rules.
  - Effect: `<child>.<Name>` is rewritten to `Name` when `<child>` is a child
    package bound by `from <package> import <child>`. The `<child>` package must
    publish `Name`, and the header must resolve it.
  - Skip: a module that binds `<child>` or `Name` outside an import keeps the
    attribute access unchanged.
  - Follow-up: the header stage then imports `Name`, and Ruff removes the
    module-object import.

A renderer without `facts` receives only the project root:

- `codemod_type_set_rules` emits the single `type-set` rule for the project's
  constants facade.
  - Shape: one rule, `isinstance($VALUE, $SET)`. Each tuple-of-classes constant
    contributes one rule-local util that matches its exact set, and one
    rewriter that replaces the set with the constant. A `rewrite` transform
    selects the rewriter whose util matched.
  - Election: by value, with no naming filter and no list. Two unmarked
    constants holding the same set fail generation with both paths. A
    compat-marked alias (below) is not elected.
  - Trigger: `isinstance(value, <set>)` outside the constants family, when the
    second argument is exactly a declared set as a tuple or a `|` union, in any
    operand order.
  - Spellings: a builtin by its bare name, a stdlib class through its top-level
    module (`ast.FunctionDef`), and any class through each constants-facade
    alias that binds it (`c.Infra.Cst.ClassDef`). A set with a member that has
    no admitted spelling gets no util.
  - Effect: the set is replaced by the constant (`c.Infra.AST_FUNCTION_TYPES`).
  - Partner: `ban-inline-type-set-outside-constants` detects every inline set,
    including those no constant declares yet; declaring the constant makes the
    generated rule repair it.
- `codemod_compat_rules` emits the single `compat-read` rule for constants kept
  through a deprecation window.
  - Marker: the constant's annotation carries the stdlib marker as metadata,
    `X: Annotated[ClassVar[...], warnings.deprecated("compat: <old> -> <new>;
    fix: <rewire>")] = <new owner>`. The message grammar is enforced; a
    malformed marker fails generation.
  - Trigger: every read of the old facade path. The declaration itself is never
    matched, so the alias does not fail CI and every use of it does.
  - Effect: each read is rewritten to the new owner named in the marker. The
    alias is deleted when no read remains in the fleet or external consumers.

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
rewritten Python file must still parse (`codemod/text_gates.py`). Prefer an ast-grep
rule whenever the target has syntax.

## Tests and snapshots

- Every ast-grep rule has `tests/<id>-test.yml` with `valid` and `invalid` cases, plus a
  snapshot under `tests/__snapshots__/<id>-snapshot.yml` for each invalid case.
- `make mod` verifies them and fails on a changed fix output, a missing snapshot, or a
  snapshot of a removed rule or case.
- Infra's `make mod-snapshots` owns regeneration. Review its diff in the same
  commit as the rule; snapshot refresh never proves behavioral acceptance.
- Invalid fixtures retain the rejected input, not its corrected output. For
  adapter construction, `m.TypeAdapter(...)` and `t.TypeAdapter(...)` exercise the
  rewrite, while `u.type_adapter(...)` is a valid fixed point. Regenerate snapshots
  from those authored inputs; never repair stale labels by editing generated output.

## Workflow

1. Search both catalogs and `utils/` for the law; widen an existing rule first.
2. Write or widen the rule and its test. Prototype only on scratch copies.
3. Run Infra's `make mod-snapshots` and review the snapshot diff.
4. Run `flext-infra refactor ast` to see every occurrence across the fleet.
5. Apply with `make mod`, then run `make tests`.
6. Commit the rule, test, snapshot and rewrites together.
