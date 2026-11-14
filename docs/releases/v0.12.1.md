# Release v0.12.1

<!-- TOC START -->

- [Scope](#scope)
- [Projects impacted](#projects-impacted)
- [Pull requests since last release](#pull-requests-since-last-release)

<!-- TOC END -->

## Scope

- Release version: 0.12.1
- Projects packaged: 2

## Projects impacted

- root
- flext-core

## Pull requests since last release

- fix(utilities): u stops inheriting the models facade; regenerate and relock (#779)
- perf(result): hold each parametrized result class strongly (#778)
- feat(lazy): defer a configuration root to its first attribute read (#777)
- fix(lazy): a facade composed from a stand-in resolves an unlisted member (#776)
- chore(deps): apply the uv dependency cooldown and relock (#775)
- fix(lazy): a stale stand-in resolves an unlisted member from its class (flext-df0b6)
  (#774)
- fix(dispatcher): JSON null is a valid nested success value (#773)
- perf(lazy): root import binds the version without loading a facade (flext-df0b6)
  (#772)
- chore(lock): pin FLEXT upstreams to their 0.12.0-dev tips (#771)
- chore(release): land upload-20261007 into 0.12.0-dev (#770)
- fix(logging): bound the drain of a reconfigured async writer (#766)
- Wip/sweep preserve/relock af9a2f9b (#769)
- Wip/sweep preserve/algar updated at 20261108 416ea732 (#768)
- Upload 20261007 (#767)
- Land upload-20261007 into 0.12.0-dev (2026-11-08 convergence) (#765)
- fix(imports): break the protocols/typings runtime cycle (#763)
- Land upload-20261007 into 0.12.0-dev (2026-11-07 convergence) (#762)
- fix(protocols): domain-event capability binds to the owned event model (#761)
- chore(protocols): project effective fields declared on pydantic mixins (#760)
- chore(tests): project the data-contract test into the namespace exports (#759)
- test(protocols): prove p.DataContract validation through the owning model (#758)
- feat(protocols): p.DataContract Pydantic schema hook (model-free typing, model
  validation) (#757)
- fix(protocols): p.BaseModel declares model\_validate/model\_validate\_json (#756)
- fix(ci): drop the hardcoded bootstrap align step (#755)
- Upload 20261007: integra a linha de upload em 0.12.0-dev (facades p.BaseModel) (#751)
- fix: integrate upload-20261007 into 0.12.0-dev (#746)
- Upload 20261007 (#745)
- \[WIP\] upload-20261007: joint plan flext-1engx integration (#743)
- Cadi 20261024 (#744)
- Upload 20261007: validate\_value accepts read-only JSON containers (#742)
- fix(core): pydantic empty facade and typing reuse (#741)
- fix(core): pydantic empty facade and beartype memoization (#740)
- fix: add empty facade to FlextModelsPydantic (#739)
- fix: add empty facade to FlextModelsPydantic (#738)
- fix: cache beartype AST traversals and modernize service handler signatures (#737)
- fix: eliminate redundant beartype traversal and apply security remediation (#736)
- Upload 20261007 (#735)
- fix: integrate strict declaration and runtime repairs (#734)
- Upload 20261007 (#733)
- Upload 20261007 (#732)
- merge: fleet stabilization and strict-test campaign into 0.12.0-dev (#731)
- Fix/core u runtime bases 20261017 (#728)
- Upload 20261007 (#729)
- fix(runtime): compose the utility runtime owner and preserved histories (#727)
- Integrate current Core contracts and reconcile native logging ownership (#726)
- Fix/fleet converge 20261015 (#725)
- chore(upload-audit-20261008): preserve upload-audit-20261008 changes … (#724)
- chore(lazy-types-4pt43): preserve fix/lazy-types-4pt43 changes across… (#723)
- Upload 20261007: fleet integration line into 0.12.0-dev (#719)
- Upload 20261007 (#717)
- WIP: stabilize upload lifecycle and native logging failures (#716)
- Upload 20261007 → 0.12.0-dev: wave fixes (6603c2c6f4) (#715)
- fix(config): flatten YAML merge keys in the unique-key loader (#713)
- upload-20261007 → 0.12.0-dev: lane fixes wave (14 commits, 0 behind) (#714)
- fixes (#712)
- Stabilize Core contracts required by workspace automation (#711)
- Upload 20261007 (#710)
- Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007 (#709)
- Upload 20261007 (#708)
- Land upload-20261007 on 0.12.0-dev: cured generator round, adapter facade cutover,
  fleet cure wave (#707)
- fix(core): zero the lint stragglers — long lines wrapped, docstring c… (#706)
- Upload 20261007 (#705)
- Fix/export pydantic settings source (#704)
- feat(settings): re-export PydanticBaseSettingsSource for consumer settings sources
  (#703)
- chore(integration): land the local integration state (operator-ordered full pouso)
  (#701)
- \[WIP\] feat(pydantic): u.empty(TypeForm(contract)) canonical empty-collection default
  (#700)
- Wip/empty default law 20261010 (#702)
- fix(beartype): drop the typing\_extensions patch; beartype 0.23 handles TE aliases
  natively (#699)
- fix(enforcement): judge an accessor name at the class that introduces it (#698)
- chore(editor): per-member pyright configuration for Pylance (useNearestConfiguration)
  (#697)
- fix: wave-1 lint cures in logging protocols and config (flext-h4bcs) (#696)
- chore: regenerate projections (dev final round) (#695)
- merge: absorb the upload lane tip into the dev branch (fleet convergence) (#694)
- merge: absorb the upload stabilization wave into 0.12.0-dev (#693)
- test(core): publish the u alias in the tests utilities facade (#692)
- Upload 20261007 (#691)
- Fix/result no shim 20261008 (#690)
- \[WIP\] feat(logging): typed structlog processing stages (preservation,… (#689)
- Stash preserve/flext core 0 20261007 (#688)
- \[WIP\] upload-20261007 alignment — CI signal only, do not merge (#687)
- fix(stabilize): land wave471 (#685) relocked on the stabilized producers (#686)
- fix(core): remove the PartFinal compat alias and mark pydantic call protocols
  runtime-checkable (#684)
- chore(unify): land upload-20261007 into 0.12.0-dev (#681)
- fix/core baseline green 20261007 (#680)
- fix(deps): restore the standalone git-pin render in the codegen/dev groups (#679)
- fix(models): the 2.13.5 field-serializer cure — classmethod (cls, value) shape (#676)
- fix(deps): republish the standalone pyproject — the member-shape rider returned over
  the inline pins (#678)
- fix(core): complete the pydantic typed owner + cure CI=Y gate residue (75→1) (#677)
- fix(models): the timestamp serializer takes the pydantic 2.13.5 bound-method signature
  (self, value) (#675)
- fix(models): P0 — serialize\_timestamps value-named again (the day-wave merge
  reintroduced the self-named spelling; pydantic 2.13.5 import breaks) (#674)
- chore(integration): land hotfix/upload-20261007 into 0.12.0-dev — flext-core day-wave
  (complexity cure, git-pin projection, rope splits) (#673)
- fix(logging): implement the missing \_should\_include\_stack\_trace method (#672)
- fix(core): tip-test cures — serializer value-named signature + BaseError
  keyword-options extraction (#671)
- fix(packaging): restore the canonical git-pin render for the fleet sibling groups
  (#669)
- hotfix/upload 20261007 (#668)
- fix(models): the plain field\_serializer takes a value-named parameter + the baseline
  wave's first pass (#667)
- fix(tests): the undocumented-operation scenario without a class-binding mutation
  (#666)
- fix(census): honor the documented lazy-published family-member exemption (#665)
- fix(packaging): restore the canonical git-pin render for the codegen sibling group
  (#664)
- hotfix: land the local 0.12.0-dev state (1 commits) (#663)
- fix(beartype): drop the dead hoisted helpers binding (#662)
- fix(models): name the plain field serializer parameter by its value role (#661)
- fix(core): pydantic 2.13 stabilization wave (cherry-picks from the dd261005 bench)
  (#660)
- chore(converge): fleet SSOT lint scopes + namespace mixins restore (#658)
- fix(typings): JsonValue class member republishes pydantic.JsonValue (SSOT) (#659)
- fix(family-surface): skip members that publish no importable root (#657)
- fix(family-surface): resolve the import module through the package index (#656)
- merge: integrate the dd261005 stabilization wave into 0.12.0-dev (#655)
- Dd261005 (#654)
- fix(coretypings): flatten JsonValue to the concrete recursive union (#653)
- integrate: wip/wt-sweep-20261005/flext-core-mise (#652)
- integrate: feat/lazy-member-20261002 (#651)
- integrate: preserve/s2-core-mod-partial-20261005 (#650)
- integrate: chore/stabilize-tip-20261003 (#649)
- integrate: refactor/core-strict-imports (#648)
- integrate: wip/wt-sweep-20261005/core-std (#647)
- integrate: wip/typings-alias-export-20261005 (#646)
- integrate: fix/core-mise-lock-dedup-20260930 (#645)
- integrate: dd261005 (#644)
- fix(gen): repair the tests namespace restructuring blockers (gen fixed point) (#643)
- wip: preserve lane WIP before worktree retirement sweep 20261005 (mer… (#634)
- dd261005 (#642)
- S2 sweep(core): safe autofix violations sweep — check findings 1863 -\> 1806 (-57)
  (#641)
- fix(core): complete the \_models extraction — facade re-exports, x base binding, lazy
  module restore, fleet relock (#640)
- wip: preserve lane WIP before worktree retirement sweep 20261005 (mer… (#629)
- Merge remote-tracking branch 'origin/0.12.0-dev' into docs/core-loggi… (#631)
- chore(overlay): tool\_version\_pins overlay + preserved working state (#639)
- merge: absorb origin/0.12.0-dev integration tip (flext-qwvb5 wave) (#630)
- Merge origin/0.12.0-dev (flext-core#535 perf enforcement short-circui… (#626)
- fix(config): reuse canonical safe YAML parser (flext-1xfks) (#623)
- fix(lazy): regenerate the utilities initializer onto the direct import map (P0 gen
  deadlock) (#622)
- chore(core): preserve canonical gate output for producer repair (#618)
- chore(deps): reconcile standalone core lock and generated providers (#616)
- fix(gen): land the drifted api-reference projections (gen fixed point) (#620)
- fix(core): restore pydantic validator overload fidelity on the m facade (#619)
- wip(core): preserve superproject-checkout residue before integration … (#613)
- feat(family-surface): admit consumer facades via dependency edge (#612)
- fix(deps): relock uv.lock to structlog 26.1.0 (val202610042236) (#617)
- WIP: restore typed executable service and retry guides (#615)
- fix(utilities): publish TypeAdapter on the pydantic facade (#614)
- fix(docs): cure the docs-audit issues blocking CI (#611)
- Merge remote-tracking branch 'origin/fix/core-mise-pin-valid-release'… (#609)
- \[WIP\] merge(integration): adopt #539 and #533 into the core stabiliza… (#608)
- fix(constants): import datetime and Path for the runtime guard tuples (#607)
- fix(lock): regenerate mise.lock so it parses (duplicate jscpd platform key) (#551)
- fix(config): own StrictYamlConfigSource in a public module — restore the root export
  (#606)
- Refactor/core strict imports (#605)
- fix(enforcement): restore the beartype rule tag and finish the package-data catalog
  cutover (#604)
- fix(enforcement): migrate catalog beartype sources to predicate\_kind
  (val202610031920) (#603)
- chore(deps): roll flext-infra pin to 098d8b071+ — make gains test-file
  (val202610031920) (#602)
- fix(constants): own the runtime type-guard tuples in the constants namespace (#597)
- fix(enforcement): list EnforcementInfraRuleSource in the rule-source union (#600)
- feat(config): expose StrictYamlConfigSource on the package root (#599)
- feat(config): expose the unique-key YAML loader on the transport facade (#598)
- merge: re-travelled mise.lock — CI setup cure; publication proof carried in history
  (#595)
- fix(constants): publish PRIMITIVES\_TYPES in the constants namespace (#596)
- fix(gen): regenerate fleet-converged surfaces (PRIMITIVES\_TYPES publication) — make
  gen exit-0 (#593)
- fix(constants): resolve shipped conflict markers + stale enforcement symbol at the tip
  (#592)
- fix(constants): import TYPE\_CHECKING in enforcement part\_07 (#591)
- chore(gen): land the gen/fmt fixed-point drift on the tip (#590)
- fix(deps): raise structlog floor to \>=26.1.0 (#589)
- fix(lock): drop the duplicated jscpd macos-x64 platform block (#588)
- integrate: fix/no-python-rules-catalog-20261001 (#584)
- integrate: fix/core-mise-pin-valid-release (#583)
- integrate: wip/stash-8be93d4e72-adopt (#586)
- integrate: fix/core-jscpd-lock-migration-20260929 (#582)
- integrate: fix/core-enforcement-flat-constants-20260929 (#581)
- integrate: preserve/core-tip-fix-20261001 (#585)
- chore(core): tip generation adoption + E501 cures (hand-authored sources) (#580)
- chore(upg): relock onto flext-infra 6ec104abf (#1409) past the cyclic facade binding
  (flext-911cm) (#579)
- feat(lazy): PR #569 recovery — restore FlextLazyMember, cure grind regressions (#578)
- chore(upg): adopt flext-infra tip 9792b4789 (pin survives an unchanged mise.lock)
  (#577)
- fix(metadata): enumerate installed distributions over one meta\_path snapshot
  (u.installed\_distributions) (#575)
- fix(index): drop the undeclared flext-infra gitlink staged by #573 (#574)
- chore(core): pre-codemod verbs stable after upg with the cured publisher — checkpoint
  2 (#573)
- feat(lazy): FlextLazyMember defers capability mixin members to first access (#572)
- chore(upg): converge flext-core onto the flext-infra tip c23721075 (#570)
- chore(core): checkpoint the stabilization grind — WIP recorded by operator order
  (#571)
- chore(gen): regenerate projections — drop stale structlog\<26 constraint (#568)
- build(mise): converge the toolchain trio — v2 lock, coherent pin/laun… (#567)
- chore(deps): absorb the cured fleet — floors, locks and projections aligned (#566)
- test(core): define public root lazy helper contract (#562)
- Chore/core repin regen 20261001 (#565)
- build(mise): pin mise 2026.9.16 + regenerate lock (fixes CI setup failure) (#563)
- chore(deps+gen): repin cli/infra/tests to their fixed tips and regenerate (#564)
- Fix/exterminate class body aliases 20261001 (#561)
- fix(core): recognize LibCST visitor metadata protocol (#552)
- \[WIP\] fix(enforcement): delete the rule tables #558 restored; keep their guidance as
  data (#560)
- fix(logging): apply settings log level to every logger, cached ones included (#559)
- \[WIP\] fix(core): replace class-body aliases with real types (#558)
- \[WIP\] refactor(enforcement): rules are package data, not Python tables (#557)
- chore(core): integrate the fmt-fixes line into 0.12.0-dev (#556)
- fix(core): dedup the enforcement enums and land the canonical generated forms (#550)
- fix(core): make u.dump fail loud on invalid inline options (V8 S8) (#555)
- fix(enforcement): report every classvar constant outside \_constants in one pass
  (#549)
- fix(enforcement): exempt libcst METADATA\_DEPENDENCIES from classvar relocation (#554)
- fix(core): remove the duplicated mise.lock sections (TOML parse error) (#548)
- \[WIP\] fix(core): one owner for the runtime type tuples (flext-hbq7n) (#547)
- fix(core): project mypy wall-time budget + enum dedup (hbq7n follow-up) (#546)
- fix(core): migrate the mise lock to the single-resolution era and prune duplicate
  enums (#545)
- chore(gen): consume the relock projections (mise 96527c3 era) (#542)
- fix(core): dedupe the scrambled jscpd resolutions in mise.lock (#544)
- fix(core): restore the keyword constructors and YAML source that #516 broke
  (flext-hbq7n) (#543)
- fix(core): relock onto the current flext-cli (fixes the red tip) (#532)
- Recovery/p0 20260927/fleet closure lanes core regen flext core (#515)
- fix(core): cure 19 runtime-census findings + strict pydantic JSON typing (flext-x8799)
  (#534)
- Fix/exceptions root import cycle 20260929 (#537)
- build(deps): update pydantic-settings requirement from \<2.15,\>=2.14.2 to
  \>=2.14.2,\<2.16 (#536)
- fix(core): move the runtime type tuples to FlextConstantsGuards; lazy loader fails
  loud; relock (#538)
- \[WIP\] fix(enforcement): retire the whole-file regex fix actions (flext-itpd1.6)
  (#540)
- fix(core): accept the regenerated projections and reach the gen fixed… (#541)
- fix(core): restore importability and adopt the census-cure lane (#531)
- \[WIP\] preserve flext-core primary-checkout dirty state (codemod/exception typing)
  (#516)
- fix(enforcement): correct the c.EnforcementRuleSeverity/SourceKind constant prefixes
  (#530)
- fix(core): type config YAML payloads with the Pydantic JsonValue contract (#526)
- chore: land pending primary-checkout commits (flext-core) (#533)
- fix(typings): land the remaining runtime type tuples at the constants owner
  (flext-x8799) (#539)
- perf(enforcement): short-circuit the field-description check before annotation eval
  (#535)
- fix(enforcement): reference the flat c.EnforcementRuleSeverity/SourceKind owners
  (#525)
- fix(enforcement): report every class past the ENFORCE-069 depth budget (flext-blgw9)
  (#523)
- Recovery/p0 20260927/flext work int flext core (#524)
- \[WIP\] preserve flext-core state from lane flext (#519)
- fix(enforcement): preserve actual import provenance (#518)
- docs(agents): render the concrete raw parent URL in the standalone mode header (#522)
- docs(agents): prescribe selector-free root verbs in the Commands block (#521)
- fix(enforcement): audit only the classes a target declares, never the ones it
  references (#513)
- fix(enforcement): route smell and bypass fix actions to manual; drop the fixer-less
  gate kind (#514)
- refactor: remove Pydantic model factories and adopt the stabilization branch (#512)
- fix(enforcement): CONFIG\_FILENAMES is a framework ClassVar hook (#511)
- fix(enforcement): derive the ENFORCE-046 peer from the lazy-import and family-surface
  owners (#510)
- fix(enforcement): ENFORCE-046 exempts the Pattern-B peer its catalog row declares
  (#509)
- feat(protocols): p.Service is the truthful service surface (V8 S4) (#508)
- feat(container): truthful container with one validated write path (V8 S2) (#507)
- feat(service): typed service operations via u.service\_operations (V8 S3) (#506)
- docs: repair python fences so the markdown-code gate compiles them (#505)
- fix(collection): make u.process fail loud on the first item failure (#504)
- fix(lazy): derive module initialization from public import semantics (#503)
- fix(enforcement): derive runtime\_checkable from class source, not a private probe
  (#502)
- test: exterminate test-law violations in flext-core (#501)
- chore(gen): regenerate member runtime projections on the V8 integration tips (#500)
- feat(service): protocol ports, validated runtime seeds and typed runtime hook (V8 S1)
  (#499)
- fix(gen): regenerate projections with the facade-parent-only alias planner (#498)
- fixes (#497)
- \[WIP\] chore(gen): project the phase-1 wave generator (flext-itpd1.3) (#496)
- merge: absorb builder-flatten lane (flat builder DSL + lazy-map cache keyed by source
  map)
- merge: land rope lane cures — facade letters on facade parents, retired hand-kept
  export registries, fix/fmt output (flext-3csgu, flext-w2iw4)
- chore(gen): rope wave-7 generator fixed point and pydantic-settings cure (#492)
- chore(gen): regenerate against the integrated flext-infra (#494)
- fixes (#491)
- merge(0.12.0-dev): land rope gen cures — FlextLazyPart01 rename + recovered
  enforcement work
- fix(tests): type the beartype subprocess output local (#490)
- merge(0.12.0-dev): land rope recovery type-stability fixes (no-ff)
- merge: adopt integration tip (no-ff)
- \[WIP\] absorb current Core integration into recovery (#488)
- merge: absorb integration tip (dependabot cooldown propagation)
- merge(0.12.0-dev): adopt published tip; keep D205 docstring from origin
- chore(gen)+fix(enforcement): regenerate with the published generator; settings rule
  reaches declared settings models (flext-0in0k C1/F0.3) (#483)
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge commit '6dc473d8d' into 0.12.0-dev
- Wip/stabilize 0.12 flext core 20260919 (#481)
- merge: bring this mirror checkout up to the published integration (fix-forward)
- merge: absorb rope-modernize lane into 0.12.0-dev (core #479)
- chore(deps): bump actions/upload-artifact from 4 to 7 (#476)
- chore(deps): bump actions/cache from 4 to 6 (#477)
- chore(deps): update structlog requirement from \<26,\>=25.5.0 to \>=25.5.0,\<27 (#478)
- merge: absorb integration tip (validated green state) into core primary
- \[WIP\] Preserve reconciliation work for handoff (#475)
- \[WIP\] preserve generated residue policy and canonical verb documentation (#474)
- merge: integrate workspace hygiene lane into 0.12.0-dev (no-ff)
- \[WIP\] stabilize generated configuration and absorb checkout changes (#473)
- Merge remote-tracking branch 'origin/feature/plan-reconciliation' into 0.12.0-dev
- Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
- chore(flext-core): preserve lane-r41 changes across 1 paths (#470)
- chore(flext-core): preserve wip/duplicate/flext/flext-core changes across 2 paths
  (#469)
- \[WIP\] fix(typings): m is the single owner of ConfigDict; t stops expo… (#468)
- feat(models): re-export the pydantic names the abstraction boundary forces through
  (#467)
- chore(gen): regenerate projections from the corrected generator (#465)
- fix(tests): the proxy stand-in satisfies the type checker it is checked by (#464)
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
- Merge commit '8e96614be09736aa6d0b8b12ff78dd9bbd64836a' into 0.12.0-dev
- \[WIP\] chore(docs): delete dead doc-example goldens carrying a machine-absolute path
  (flext-uwhzz) (#431)
- chore(fleet): sync toolchain config, regenerate lazy exports, migrate beads to shared
  symlink (#429)
- chore(integration): adopt fleet toolchain lane #424 and pass the full gate chain
  (flext-1wjg1.16) (#428)
- fix: dedupe pydantic symbols\_to\_replace into class-body constant (flext-1sm3w)
  (#424)
