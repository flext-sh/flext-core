# Release v0.12.1

## Scope

- Release version: 0.12.1
- Projects packaged: 2

## Projects impacted

- root
- flext-core

## Pull requests since last release

feat(lazy): defer a configuration root to its first attribute read (#777)
Merge branch '0.12.0-dev' into perf/lazy-instance-export-20261113
fix(lazy): a facade composed from a stand-in resolves an unlisted member (#776)
chore(deps): apply the uv dependency cooldown and relock (#775)
merge: adopt 0.12.0-dev (#774 lazy stand-in resolution) into the lazy-instance lane
fix(lazy): a stale stand-in resolves an unlisted member from its class (flext-df0b6)
(#774)
fix(dispatcher): JSON null is a valid nested success value (#773)
perf(lazy): root import binds the version without loading a facade (flext-df0b6) (#772)
chore(lock): pin FLEXT upstreams to their 0.12.0-dev tips (#771)
chore(release): land upload-20261007 into 0.12.0-dev (#770)
Merge branch '0.12.0-dev' into upload-20261007
fix(logging): bound the drain of a reconfigured async writer (#766)
Wip/sweep preserve/relock af9a2f9b (#769)
Wip/sweep preserve/algar updated at 20261108 416ea732 (#768)
Upload 20261007 (#767)
Merge origin/0.12.0-dev into fix/blocked-pipe-20261108
Land upload-20261007 into 0.12.0-dev (2026-11-08 convergence) (#765)
{"retained\_commits":\[\],"head\_entries":\[{"path":"uv.lock","mode":"100644","oid":"53a4e0e0ce8a44c9c9fe0d66b310a6be2e0abb1e"}\],"repo\_root":"/home/marlonsc/worktrees/flext-core/relock","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"3d21fccb93b63c4219047811f6a8ba6b9db2bf8d","branch":"fix/relock-receipt-root-20261108","paths":\["uv.lock",".flext/executor.setup.lock"\],"index\_entries":\[{"path":"uv.lock","mode":"100644","oid":"53a4e0e0ce8a44c9c9fe0d66b310a6be2e0abb1e"}\],"files":\[{"path":".flext/executor.setup.lock","mode":"100644","permissions":420,"oid":"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"},{"path":"uv.lock","mode":"100644","permissions":420,"oid":"27dc8967f5efab162b1d2d40b6a32085f6076ee5"}\]}
{"retained\_commits":\[\],"head\_entries":\[\],"repo\_root":"/home/marlonsc/worktrees/flext-core/algar-updated-at-20261108","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"3e3dc349111ac3032ef40a468ddd02413e65659a","branch":"fix/algar-updated-at-20261108","paths":\[".flext/executor.setup.lock"\],"index\_entries":\[\],"files":\[{"path":".flext/executor.setup.lock","mode":"100644","permissions":420,"oid":"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"}\]}
merge: origin/preserve/flext-core/rig-wip-20261108 into upload-20261007 (fleet
convergence 2026-11-08)
merge: origin/preserve/flext-core/relock-receipt-root-wip-20261108 into upload-20261007
(fleet convergence 2026-11-08)
Merge remote-tracking branch 'refs/remotes/origin/upload-20261007' into upload-20261007
Merge commit '2788ae53bbd3c3ca7f087d5becb0650eba272549' into upload-20261007
Merge commit 'b35d34fa36edf82559014991df4850b13aa036d0' into upload-20261007
Merge commit '0e6e0cd8494be396a8aaa2a1fa45e755567b49e3' into upload-20261007
fix(entity): assign updated\_at through the declaring base (#764)
fix(imports): break the protocols/typings runtime cycle (#763)
Land upload-20261007 into 0.12.0-dev (2026-11-07 convergence) (#762)
fix(protocols): domain-event capability binds to the owned event model (#761)
chore(protocols): project effective fields declared on pydantic mixins (#760)
chore(tests): project the data-contract test into the namespace exports (#759)
test(protocols): prove p.DataContract validation through the owning model (#758)
feat(protocols): p.DataContract Pydantic schema hook (model-free typing, model
validation) (#757)
fix(protocols): p.BaseModel declares model\_validate/model\_validate\_json (#756)
fix(ci): drop the hardcoded bootstrap align step (#755)
Upload 20261007: integra a linha de upload em 0.12.0-dev (facades p.BaseModel) (#751)
\[WIP\] Merge fix/core-typealias-omit-20261105 into upload-20261007 (adopted from
gruponos session)
\[WIP\] merge: fix/core-lazy-install-args-20261105 (base reds of upload-20261007) into
the omit lane
Merge branch 'wip/sweep-preserve/flext-core-0422a41c' into upload-20261007
Merge branch
'preserve/flext-core/flext\_.worktrees\_dcrecon-tooling-20261101\_flext-core-20261102'
into upload-20261007
Merge branch
'preserve/flext-core/flext-worktrees\_ruff-agent-exclusions-20261019\_flext-core-20261102'
into upload-20261007
Merge branch
'preserve/flext-core/flext-worktrees\_governance-docs-20261019\_flext-core-20261102'
into upload-20261007
Merge branch 'preserve/b-20261025' into upload-20261007
Merge branch 'wip/sweep-preserve/flext-core-b9b07046' into upload-20261007
Merge branch 'wip/sweep-preserve/flext-core-5ac45f02' into upload-20261007
Merge branch 'fix/protocol-model-dump-native-kwargs-20261105' into upload-20261007
Merge branch 'wip/sweep-preserve/core-clean-f199e1f7' into upload-20261007
Merge branch 'wip/sweep-preserve/lazy-deferred-20261105-877307cb' into upload-20261007
Merge branch 'wip/sweep-preserve/flext-core-8b456d52' into upload-20261007
Merge branch 'fix/lazy-deferred-capabilities-20261105' into upload-20261007
Merge branch 'preserve/lane-c-constants-flext-core-v2' into upload-20261007
Merge branch 'preserve/lane-c-constants-flext-core-20261104' into upload-20261007
Merge branch 'fix/rope-dc11-pin-20261101' into upload-20261007
{"retained\_commits":\[\],"head\_entries":\[\],"repo\_root":"/home/marlonsc/tmp/kilo/core-clean","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"781b60393a7512d2d90a19ce8ffced3c2e6d9aa0","branch":"fix/contract-core-20261105","paths":\[\],"index\_entries":\[\],"files":\[\]}
{"retained\_commits":\[\],"head\_entries":\[{"path":"mise.lock","mode":"100644","oid":"bf353b8def70fe11cfa5c102cb7d5c929615e1fd"}\],"repo\_root":"/home/marlonsc/flext/.worktrees/dcrecon-tooling-20261101/flext-core","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"8e390c0c6e857810b14cd4760f2caec8a7ca1214","branch":null,"paths":\["mise.lock"\],"index\_entries":\[{"path":"mise.lock","mode":"100644","oid":"bf353b8def70fe11cfa5c102cb7d5c929615e1fd"}\],"files":\[{"path":"mise.lock","mode":"100644","permissions":420,"oid":"3b209b69257f4256f15fa0c9de79599c67e7e721"}\]}
{"retained\_commits":\[\],"head\_entries":\[\],"repo\_root":"/home/marlonsc/flext-core-worktrees/lazy-deferred-20261105","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"8928b3cb3157e39bb861eeecd768b274e023961a","branch":"fix/lazy-deferred-capabilities-20261105","paths":\[".flext/.venv.lifecycle.lock",".flext/.venv.setup.lock"\],"index\_entries":\[\],"files":\[{"path":".flext/.venv.lifecycle.lock","mode":"100644","permissions":420,"oid":"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"},{"path":".flext/.venv.setup.lock","mode":"100644","permissions":420,"oid":"e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"}\]}
{"retained\_commits":\[\],"head\_entries":\[{"path":"docs/guides/codemod-rules.md","mode":"100644","oid":"67ca817386887139ea105a4e3ab50a6a29a9fbca"},{"path":"docs/guides/engine.md","mode":"100644","oid":"df198a0bc36627d2c8a3c4dc5883399e4a5a108c"},{"path":"docs/guides/make-commands.md","mode":"100644","oid":"b908b44b54be203ce90af929aca4c0945c945390"},{"path":"docs/guides/using-flext-core.md","mode":"100644","oid":"f601277b701272275119d0845c6b957e5677db22"},{"path":"src/flext\_core/\_protocols/handler.py","mode":"100644","oid":"2c429d309cbe096e99e82754e7395135f867aae5"},{"path":"src/flext\_core/mixins.py","mode":"100644","oid":"7f4e268ac2b02f71ca5061a2a684bf55da48f6db"},{"path":"tests/benchmark/test\_lazy\_performance.py","mode":"100644","oid":"fd800bd0b7abd6e7c97c36f32365a8d7819e8e04"}\],"repo\_root":"/home/marlonsc/flext/flext-core","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"5587155ff90c69b30697ccc894f59a79be84856c","branch":"upload-20261007","paths":\["docs/guides/codemod-rules.md","docs/guides/engine.md","docs/guides/make-commands.md","docs/guides/using-flext-core.md","src/flext\_core/\_protocols/handler.py","src/flext\_core/mixins.py","tests/benchmark/test\_lazy\_performance.py"\],"index\_entries":\[{"path":"docs/guides/codemod-rules.md","mode":"100644","oid":"740e594e0a7f80e6abf855613e3317eb2df47d55"},{"path":"docs/guides/engine.md","mode":"100644","oid":"43f91327dc4c63cc7efda4657bc3c0d8a96ea7fa"},{"path":"docs/guides/make-commands.md","mode":"100644","oid":"11891df9dd1aef9d1426d359298ffe2413c456b5"},{"path":"docs/guides/using-flext-core.md","mode":"100644","oid":"445973e69b68c1c9a3f82c1bd791610685615eb5"},{"path":"src/flext\_core/\_protocols/handler.py","mode":"100644","oid":"2c429d309cbe096e99e82754e7395135f867aae5"},{"path":"src/flext\_core/mixins.py","mode":"100644","oid":"7f4e268ac2b02f71ca5061a2a684bf55da48f6db"},{"path":"tests/benchmark/test\_lazy\_performance.py","mode":"100644","oid":"fd800bd0b7abd6e7c97c36f32365a8d7819e8e04"}\],"files":\[{"path":"docs/guides/codemod-rules.md","mode":"100644","permissions":420,"oid":"50f9266d845b21b109951ac16b3784a62626205b"},{"path":"docs/guides/engine.md","mode":"100644","permissions":420,"oid":"485671fbc9dc92af044f51f85fce2809fe9343eb"},{"path":"docs/guides/make-commands.md","mode":"100644","permissions":420,"oid":"11891df9dd1aef9d1426d359298ffe2413c456b5"},{"path":"docs/guides/using-flext-core.md","mode":"100644","permissions":420,"oid":"a13c8a766f52e91a4f320cb675e719a61e63bf66"},{"path":"src/flext\_core/\_protocols/handler.py","mode":"100644","permissions":420,"oid":"89a03ea4d7875668cce12da004f09fac9f7b9e9e"},{"path":"src/flext\_core/mixins.py","mode":"100644","permissions":420,"oid":"71e24af6d96d91a81c72b8af113f550595e48c7d"},{"path":"tests/benchmark/test\_lazy\_performance.py","mode":"100644","permissions":420,"oid":"f83925d01a9a3966ac202be278f76ff0e989d625"}\]}
fix(lazy): resolve letter aliases during package initialization (#753)
fix(tests): drop dead StatusLiteral type alias from constants owner (#752)
fix: integrate upload-20261007 into 0.12.0-dev (#746)
Merge remote-tracking branch 'refs/remotes/origin/upload-20261007' into upload-20261007
fix(constants): generic type-set constants keep isinstance narrowing (#748)
formats
Merge remote-tracking branch 'refs/remotes/origin/upload-20261007' into upload-20261007
Merge pull request #747: core exceptions-family violation warnings, ADR-005
config\_sources, orphan skills retirement
\[WIP\] merge origin/upload-20261007 into the incident core lane
\[WIP\] merge fix/core-chain-link-20261101 (3b1d3072c3): u's dead pydantic copies
removed
\[WIP\] merge preserve/core-mod-20261101 (a17cbdbab7): PartNN chains reabsorbed
\[WIP\] merge preserve/core-mod-20261101 (a17cbdbab7) into fix/rope-dc11-pin-20261101
\[WIP\] merge upload-20261007 into fix/rope-dc11-pin-20261101
\[WIP\] merge preserve/incident-core-20261101 (dbb34e18e3) into
fix/rope-dc11-pin-20261101
\[WIP\] merge origin/upload-20261007 into the adopted incident core lane
Merge commit 'e20f6fab1838442a95646a99ff1eea7fb2ce0889' into upload-20261007
Merge commit 'c96017c9a1510a7733bff1b9d7bb11e212661a20' into upload-20261007
formats
Upload 20261007 (#745)
\[WIP\] upload-20261007: joint plan flext-1engx integration (#743)
Merge commit 'e8bfa90995f4e9dc28b63c1ba87db7730d7989e1' into upload-20261007
Merge commit '5b67a90b93207b76005b5f78f0b00605c69314bb' into upload-20261007
Merge commit 'c46b94d5b635d536c42b712a28e2074e9d4b6365' into upload-20261007
{"retained\_commits":\[\],"head\_entries":\[{"path":"Makefile","mode":"100644","oid":"8efc2e24d5a47e567edd6603b2abef68d5759a38"},{"path":"src/flext\_core/\_utilities/\_beartype/module\_visitor.py","mode":"100644","oid":"a62aabd25ba9820acb49f8956e3df3e53d326856"},{"path":"tests/unit/test\_beartype\_engine\_annotations.py","mode":"100644","oid":"9f8e0f97a3f050c6f368dfecfd8a3713e33d5152"},{"path":"tests/unit/test\_container\_resolution.py","mode":"100644","oid":"fc6fc9251b95499d57d3384bbc2f88b82a53f5b8"},{"path":"tests/unit/test\_handler\_discovery\_module.py","mode":"100644","oid":"650eb72737b45fb3930e09fc647eb0189c1cc1b2"},{"path":"tests/unit/test\_type\_alias\_resolution.py","mode":"100644","oid":"e8e6892da3602765be8b91491dfe82d23933ae93"}\],"repo\_root":"/home/marlonsc/flext-worktrees/ruff-agent-exclusions-20261019/flext-core","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"774862d6b5add55e2751974a56f95afdc6d47d45","branch":"fix/ruff-agent-exclusions-20261019","paths":\["Makefile","src/flext\_core/\_utilities/\_beartype/module\_visitor.py","tests/unit/test\_beartype\_engine\_annotations.py","tests/unit/test\_container\_resolution.py","tests/unit/test\_handler\_discovery\_module.py","tests/unit/test\_type\_alias\_resolution.py"\],"index\_entries":\[{"path":"Makefile","mode":"100644","oid":"8efc2e24d5a47e567edd6603b2abef68d5759a38"},{"path":"src/flext\_core/\_utilities/\_beartype/module\_visitor.py","mode":"100644","oid":"a62aabd25ba9820acb49f8956e3df3e53d326856"},{"path":"tests/unit/test\_beartype\_engine\_annotations.py","mode":"100644","oid":"9f8e0f97a3f050c6f368dfecfd8a3713e33d5152"},{"path":"tests/unit/test\_container\_resolution.py","mode":"100644","oid":"fc6fc9251b95499d57d3384bbc2f88b82a53f5b8"},{"path":"tests/unit/test\_handler\_discovery\_module.py","mode":"100644","oid":"650eb72737b45fb3930e09fc647eb0189c1cc1b2"},{"path":"tests/unit/test\_type\_alias\_resolution.py","mode":"100644","oid":"e8e6892da3602765be8b91491dfe82d23933ae93"}\],"files":\[{"path":"Makefile","mode":"100644","permissions":420,"oid":"c5f2d28bed774f82c75fe6b4e9018c5d03c7fab4"},{"path":"src/flext\_core/\_utilities/\_beartype/module\_visitor.py","mode":"100644","permissions":420,"oid":"7c32d1f650d3794793787f2eacf574abe3d945b6"},{"path":"tests/unit/test\_beartype\_engine\_annotations.py","mode":"100644","permissions":420,"oid":"3ad3133ab22d3210af045ffdb08f0eda5e017607"},{"path":"tests/unit/test\_container\_resolution.py","mode":"100644","permissions":420,"oid":"3f51344cb42eb364164e27245979e4f04cc17d94"},{"path":"tests/unit/test\_handler\_discovery\_module.py","mode":"100644","permissions":420,"oid":"dd82491804dbea0ebb139770ab99237871c11663"},{"path":"tests/unit/test\_type\_alias\_resolution.py","mode":"100644","permissions":420,"oid":"8962f3ff589d53612318e30826d603b39ea923ae"}\]}
{"retained\_commits":\[\],"head\_entries":\[{"path":"AGENTS.md","mode":"100644","oid":"a5d05ec64aa7c4a081d1f62ccf5904e767d2cc63"}\],"repo\_root":"/home/marlonsc/flext-worktrees/governance-docs-20261019/flext-core","common\_dir":"/home/marlonsc/flext/.git/modules/flext-core","head":"bc25084cb4fd7ceaa99dae768b2cbdc432d7822d","branch":"docs/governance-docs-20261019","paths":\["AGENTS.md"\],"index\_entries":\[{"path":"AGENTS.md","mode":"100644","oid":"a5d05ec64aa7c4a081d1f62ccf5904e767d2cc63"}\],"files":\[{"path":"AGENTS.md","mode":"100644","permissions":420,"oid":"d3dec3f7ce70b8387a0aa10c318fd0e203bae47c"}\]}
Cadi 20261024 (#744)
\[WIP\] merge cadi-20261024 (CA/DI lane) into upload-20261007
\[WIP\] merge pydantic-max-20261024
\[WIP\] merge root-recorded flext-core tip into cadi-20261024
\[WIP\] merge \~/flext upload-20261007 into pydantic-max-20261024
\[WIP\] merge origin/pydantic-max-20261024 into cadi-20261024 (adopt pydantic lane core
work)
\[WIP\] merge pydantic-max-20261024
\[WIP\] merge origin/upload-20261007 into cadi-20261024 (adopt published core work)
\[WIP\] merge pydantic-max-20261024 (flext-core presets, t.Port)
\[WIP\] merge \~/flext upload-20261007 into pydantic-max-20261024
\[WIP\] merge pydantic-max-20261024 (flext-core)
\[WIP\] merge \~/flext upload-20261007 into pydantic-max-20261024
Upload 20261007: validate\_value accepts read-only JSON containers (#742)
Merge commit '1fb24ea5dc7aabfffed8b3a6e8f0cac06ec01cad' into upload-20261007
Merge commit '0507db399d12c9f4e032f9f85478b3d8242fd18b' into upload-20261007
Merge commit '02d68d3c20f3fdec44340ca5df440e5abb8efaf7' into upload-20261007
fix(core): pydantic empty facade and typing reuse (#741)
fix(core): pydantic empty facade and beartype memoization (#740)
fix: add empty facade to FlextModelsPydantic (#739)
fix: add empty facade to FlextModelsPydantic (#738)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
fix: cache beartype AST traversals and modernize service handler signatures (#737)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
fix: eliminate redundant beartype traversal and apply security remediation (#736)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
Upload 20261007 (#735)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
fix: integrate strict declaration and runtime repairs (#734)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
Upload 20261007 (#733)
merge: adopt core integration tip
Upload 20261007 (#732)
Merge commit '176e920fe952d81f546fa561cf455666efd6cbaf' into upload-20261007
merge: fleet stabilization and strict-test campaign into 0.12.0-dev (#731)
merge: adopt promoted integration and align declared dependency line
Fix/core u runtime bases 20261017 (#728)
Upload 20261007 (#729)
chore(integration): reconcile preserved checkout history
merge: preserve reviewed historical workspace projections
\[WIP\] Merge origin/upload-20261007 into the core runtime-bases lane
fix(runtime): compose the utility runtime owner and preserved histories (#727)
merge: preserve historical Core projections under current owners
merge: preserve historical Core projections under current owners
merge: preserve historical Core projections under current owners
merge: preserve historical Core projections under current owners
merge: adopt the published Core integration base
fix(runtime): integrate the utility runtime dependency owner
Merge commit '80f133721ac769908653ffbc9363abfceab7ec2e' into upload-20261007
Integrate current Core contracts and reconcile native logging ownership (#726)
\[WIP\] Merge origin/upload-20261007 into the core logging stream port lane
Merge commit 'c18fc673fee8c69c163dc168ea888dbd64473135' into upload-20261007
Merge commit 'e83a8f55f34fbf9faec2b5d9becbe57cdc247955' into upload-20261007
Merge commit '5b2af7e45066b50de92409db935d6481f053e6e8' into upload-20261007
Merge remote-tracking branch 'origin/fix/stabilize-0121-20261008' into upload-20261007
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Fix/fleet converge 20261015 (#725)
\[WIP\] Merge origin/0.12.0-dev into fix/fleet-converge-20261015
chore(upload-audit-20261008): preserve upload-audit-20261008 changes … (#724)
Merge branch '0.12.0-dev' into upload-audit-20261008
chore(lazy-types-4pt43): preserve fix/lazy-types-4pt43 changes across… (#723)
Merge branch '0.12.0-dev' into fix/lazy-types-4pt43
Upload 20261007: fleet integration line into 0.12.0-dev (#719)
\[WIP\] Merge origin/upload-20261007 into fix/fleet-converge-20261015
Merge commit '54cb7bac790dbb0a383b354f08c147012afece69' into upload-20261007
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
\[WIP\] Merge origin/cure/pydantic-typed-owner-20261007 into fix/fleet-converge-20261015
(snapshot hunks; survivor wins conflicts)
\[WIP\] Merge origin/fix/make-ci-upload-20261013 into fix/fleet-converge-20261015
(snapshot hunks; survivor wins conflicts)
\[WIP\] Merge origin/fix/projections-4pt43 into fix/fleet-converge-20261015
\[WIP\] Merge origin/upload-20261007 into fix/fleet-converge-20261015
Upload 20261007 (#717)
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge commit '163a3158a265a117441c1e2222066f31c6db6403' into 0.12.0-dev
WIP: stabilize upload lifecycle and native logging failures (#716)
Merge commit '18514aca09964a630b6f830aaea50bb63a2a5ad5' into upload-20261007
Upload 20261007 → 0.12.0-dev: wave fixes (6603c2c6f4) (#715)
Merge commit 'a3060398a880129c0ab49f3420d3e69f414085ec' into upload-20261007
fix(config): flatten YAML merge keys in the unique-key loader (#713)
upload-20261007 → 0.12.0-dev: lane fixes wave (14 commits, 0 behind) (#714)
merge: adopt 0.12.0-dev fixes onto upload-20261007 (union, no side dropped)
fixes (#712)
Stabilize Core contracts required by workspace automation (#711)
Upload 20261007 (#710)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007 (#709)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
Upload 20261007 (#708)
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
Land upload-20261007 on 0.12.0-dev: cured generator round, adapter facade cutover, fleet
cure wave (#707)
fix(core): zero the lint stragglers — long lines wrapped, docstring c… (#706)
Upload 20261007 (#705)
Fix/export pydantic settings source (#704)
Merge commit '934ca82be7220993019a794a09c9724866e22a04' into upload-20261007
Merge commit '52dc73e7aa91a790fb8fcd1f443bc67c3751a17e' into upload-20261007
feat(settings): re-export PydanticBaseSettingsSource for consumer settings sources
(#703)
chore(integration): land the local integration state (operator-ordered full pouso)
(#701)
\[WIP\] feat(pydantic): u.empty(TypeForm(contract)) canonical empty-collection default
(#700)
Wip/empty default law 20261010 (#702)
Merge commit '5395d334bbc2b9ca58063c96db43b0e0fc4e9237' into upload-20261007
Merge commit '9f37cfab59196b132c57d6edb722d1c991dd5180' into upload-20261007
fix(beartype): drop the typing\_extensions patch; beartype 0.23 handles TE aliases
natively (#699)
Merge commit 'b17cbab4072573b107ae57124b60c0096bd107de' into upload-20261007
fix(enforcement): judge an accessor name at the class that introduces it (#698)
chore(editor): per-member pyright configuration for Pylance (useNearestConfiguration)
(#697)
fix: wave-1 lint cures in logging protocols and config (flext-h4bcs) (#696)
Merge remote-tracking branch 'origin/0.12.0-dev' into fix/lint-core-wave1-20261009
chore: regenerate projections (dev final round) (#695)
merge: union with the 0.12.0-dev integration tip (upload wave absorption)
merge: absorb the upload lane tip into the dev branch (fleet convergence) (#694)
merge: absorb the upload lane tip into the dev branch (fleet convergence)
merge: absorb the upload stabilization wave into 0.12.0-dev (#693)
Merge remote-tracking branch 'origin/upload-20261007' into merge/upload-to-0120-20261009
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
test(core): publish the u alias in the tests utilities facade (#692)
Upload 20261007 (#691)
Fix/result no shim 20261008 (#690)
\[WIP\] feat(logging): typed structlog processing stages (preservation,… (#689)
Merge branch '0.12.0-dev' into fix/stabilize-0121-20261008
Stash preserve/flext core 0 20261007 (#688)
Merge branch '0.12.0-dev' into stash-preserve/flext-core-0-20261007
merge: absorb origin/upload-20261007 into fix/result-no-shim-20261008
merge: absorb the 0.12.0-dev integration tip back into the upload lane
\[WIP\] upload-20261007 alignment — CI signal only, do not merge (#687)
Merge commit '6f1abb1ad00b03311c9767cec1b5413c5c5dd0cd' into upload-20261007
merge: adopt the npm-free relock lane into upload-20261007
merge: align upload-20261007 with origin/0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into land/wave471-core
fix(stabilize): land wave471 (#685) relocked on the stabilized producers (#686)
merge: adopt the newer flext-core#685 commits
Merge remote-tracking branch 'origin/upload-20261007' into land/wave471-core
Merge commit 'f6e028fa6ccddc51b84785f0c55ca0f29c3f4ff0' into upload-20261007
Merge commit '58366f14c6388e09af75b3405d850764b52f044a' into upload-20261007
Merge commit 'd3def6612ed97dd7606e77be3ff2ef442e050138' into upload-20261007
merge: adopt flext-core#685 (wave471 upload landing) into the stabilization lane
Merge origin/lane/core-serializer-gen-20261007 (#683) into land/wave471-core
Merge remote-tracking branch 'origin/upload-20261007' into land/wave471-core
Merge remote-tracking branch 'origin/0.12.0-dev' into upload-20261007
merge: adopt recorded integration pointer without parking work
fix(core): remove the PartFinal compat alias and mark pydantic call protocols
runtime-checkable (#684)
chore(unify): land upload-20261007 into 0.12.0-dev (#681)
Merge branch 'dup-261007' into hotfix/upload-20261007
wip
fix/core baseline green 20261007 (#680)
merge(core): adopt the day-wave tip's CI-cure follow-ups into the fusion — serializer
classmethod cure and gate residue absorbed
merge(core): fuse the baseline-green refactor lane with the day-wave tip —
serializer/params runtime cures preserved, structural splits adopted
fix(deps): restore the standalone git-pin render in the codegen/dev groups (#679)
merge: integrate the 0.12.0-dev tip into the git-pin render cure lane
fix(models): the 2.13.5 field-serializer cure — classmethod (cls, value) shape (#676)
fix(models): resolve the four project\_metadata part conflicts — parts-import form
(theirs), byte-identical bodies verified by compile (bead flext-gluhs)
fix(deps): republish the standalone pyproject — the member-shape rider returned over the
inline pins (#678)
merge: integrate the 0.12.0-dev tip into the standalone-pyproject lane
fix(core): complete the pydantic typed owner + cure CI=Y gate residue (75→1) (#677)
merge: adopt 0.12.0-dev tip
Merge remote-tracking branch 'origin/0.12.0-dev' into hotfix/field-serializer-20261007
fix(models): the timestamp serializer takes the pydantic 2.13.5 bound-method signature
(self, value) (#675)
merge: integrate the 0.12.0-dev tip (#674 value-named rename) into the serializer cure
lane
Merge remote-tracking branch 'origin/0.12.0-dev' into hotfix/field-serializer-20261007
fix(models): P0 — serialize\_timestamps value-named again (the day-wave merge
reintroduced the self-named spelling; pydantic 2.13.5 import breaks) (#674)
Merge commit 'd935eff297e0c8357c8c2fbad960e87202b7d77e' into hotfix/upload-20261007
Merge commit '413e06a09ed65d20918e5e3e264e72d2e7ffaffc' into hotfix/upload-20261007
chore(integration): land hotfix/upload-20261007 into 0.12.0-dev — flext-core day-wave
(complexity cure, git-pin projection, rope splits) (#673)
fix(models): update serialize\_timestamps method to use self instead of value
Merge commit 'e932e5e305f95a3328b403b331ff4c73f38da0bb' into hotfix/upload-20261007
fix(logging): implement the missing \_should\_include\_stack\_trace method (#672)
fix(core): tip-test cures — serializer value-named signature + BaseError keyword-options
extraction (#671)
fix(packaging): restore the canonical git-pin render for the fleet sibling groups (#669)
hotfix/upload 20261007 (#668)
Merge commit '2436b83247bee8c4f88730d0a0073fa7f4d35a82' into hotfix/upload-20261007
Merge commit 'bc57d93b8e091854dddc9a8d0fc0e8c8f0cbf54a' into hotfix/upload-20261007
Merge commit 'afd0cd3281e861363a05c1cb701c6672e11bc22c' into hotfix/upload-20261007
fix(models): the plain field\_serializer takes a value-named parameter + the baseline
wave's first pass (#667)
fix(tests): the undocumented-operation scenario without a class-binding mutation (#666)
merge: adopt the producer typed-owner line (dd261005) onto the 0.12.0-dev tip
fix(census): honor the documented lazy-published family-member exemption (#665)
fix(packaging): restore the canonical git-pin render for the codegen sibling group
(#664)
hotfix: land the local 0.12.0-dev state (1 commits) (#663)
fix(beartype): drop the dead hoisted helpers binding (#662)
fix(models): name the plain field serializer parameter by its value role (#661)
fix(core): pydantic 2.13 stabilization wave (cherry-picks from the dd261005 bench)
(#660)
chore(converge): fleet SSOT lint scopes + namespace mixins restore (#658)
fix(typings): JsonValue class member republishes pydantic.JsonValue (SSOT) (#659)
Merge branch '0.12.0-dev' into dd261005
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into dd261005
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
fix(family-surface): skip members that publish no importable root (#657)
fix(family-surface): resolve the import module through the package index (#656)
merge: integrate the dd261005 stabilization wave into 0.12.0-dev (#655)
merge: integrate the dd261005 stabilization lane (rope-resolved projections, lazy cycle
break, fleet warning gates) into 0.12.0-dev
Dd261005 (#654)
merge: adopt the stabilized wave tip into dd261005
fix(coretypings): flatten JsonValue to the concrete recursive union (#653)
merge: consolidate pending core repair into dd261005
Merge superproject gitlink 82948066 into dd261005
integrate: wip/wt-sweep-20261005/flext-core-mise (#652)
Merge branch 'wip/wt-sweep-20261005/flext-core-mise' into 0.12.0-dev (merge-admin
integration sweep 20261006; tip-side on collisions, lane adopted where non-colliding)
integrate: feat/lazy-member-20261002 (#651)
Merge branch 'feat/lazy-member-20261002' into 0.12.0-dev (merge-admin integration sweep
20261006; tip-side on collisions, lane adopted where non-colliding)
integrate: preserve/s2-core-mod-partial-20261005 (#650)
Merge branch 'preserve/s2-core-mod-partial-20261005' into 0.12.0-dev (merge-admin
integration sweep 20261006)
integrate: chore/stabilize-tip-20261003 (#649)
Merge branch 'chore/stabilize-tip-20261003' into 0.12.0-dev (merge-admin integration
sweep 20261006)
integrate: refactor/core-strict-imports (#648)
Merge branch 'refactor/core-strict-imports' into 0.12.0-dev (merge-admin integration
sweep 20261006)
Merge remote-tracking branch 'origin/wip/wt-sweep-20261005/flext-core-mise' into
dd261005
integrate: wip/wt-sweep-20261005/core-std (#647)
Merge branch 'wip/wt-sweep-20261005/core-std' into 0.12.0-dev (merge-admin integration
sweep 20261006)
integrate: wip/typings-alias-export-20261005 (#646)
Merge branch 'wip/typings-alias-export-20261005' into 0.12.0-dev (merge-admin
integration sweep 20261006)
integrate: fix/core-mise-lock-dedup-20260930 (#645)
Merge branch 'fix/core-mise-lock-dedup-20260930' into 0.12.0-dev (merge-admin
integration sweep 20261006)
Merge remote-tracking branch 'origin/chore/upg-infra-0a08f13e5-flext-core' into dd261005
integrate: dd261005 (#644)
Merge branch 'dd261005' into 0.12.0-dev (merge-admin integration sweep 20261006)
Merge remote-tracking branch 'origin/chore/stabilize-tip-20261003' into dd261005
Merge remote-tracking branch 'origin/preserve/s2-core-mod-partial-20261005' into
dd261005
Merge remote-tracking branch 'origin/refactor/core-strict-imports' into dd261005
Merge remote-tracking branch 'origin/wip/typings-alias-export-20261005' into dd261005
Merge superproject gitlink c425f4ae into dd261005
fix(gen): repair the tests namespace restructuring blockers (gen fixed point) (#643)
wip: preserve lane WIP before worktree retirement sweep 20261005 (mer… (#634)
Merge branch '0.12.0-dev' into recovery/p0-20260927/flext-work--int--flext-core
dd261005 (#642)
S2 sweep(core): safe autofix violations sweep — check findings 1863 -\> 1806 (-57)
(#641)
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
fix(core): complete the \_models extraction — facade re-exports, x base binding, lazy
module restore, fleet relock (#640)
wip: preserve lane WIP before worktree retirement sweep 20261005 (mer… (#629)
Merge remote-tracking branch 'origin/0.12.0-dev' into docs/core-loggi… (#631)
chore(overlay): tool\_version\_pins overlay + preserved working state (#639)
merge: absorb origin/0.12.0-dev integration tip (flext-qwvb5 wave) (#630)
Merge branch '0.12.0-dev' into wip/wt-sweep-20261005/flext-core-refresh
Merge origin/0.12.0-dev (flext-core#535 perf enforcement short-circui… (#626)
Merge branch '0.12.0-dev' into wip/wt-sweep-20261005/0.12.0-dev
fix(config): reuse canonical safe YAML parser (flext-1xfks) (#623)
fix(lazy): regenerate the utilities initializer onto the direct import map (P0 gen
deadlock) (#622)
chore(core): preserve canonical gate output for producer repair (#618)
chore(deps): reconcile standalone core lock and generated providers (#616)
Merge branch 'fix/core-lock-runtime-idl68-20261004' into
wip/core-native-gate-effects-idl68-20261005
Merge branch '0.12.0-dev' into fix/core-lock-runtime-idl68-20261004
fix(gen): land the drifted api-reference projections (gen fixed point) (#620)
merge: absorb origin/0.12.0-dev integration tip (flext-qwvb5 wave)
fix(core): restore pydantic validator overload fidelity on the m facade (#619)
wip(core): preserve superproject-checkout residue before integration … (#613)
Merge branch '0.12.0-dev' into wip/flext-core-checkout-residue-20261004
merge: integrate origin/0.12.0-dev (ab63322bf1, family-surface #612) into
lane/validator-facade-20261005
feat(family-surface): admit consumer facades via dependency edge (#612)
Merge origin/0.12.0-dev (6a69b412d) into family-surface consumer discovery lane
Merge remote-tracking branch 'origin/0.12.0-dev' into
docs/core-logging-settings-01ifa-20261005
Merge remote-tracking branch 'origin/0.12.0-dev' into
wip/core-native-gate-effects-idl68-20261005
fix(deps): relock uv.lock to structlog 26.1.0 (val202610042236) (#617)
WIP: restore typed executable service and retry guides (#615)
Merge remote-tracking branch 'origin/0.12.0-dev' into
docs/core-runtime-contract-ha4nw-20261004
fix(utilities): publish TypeAdapter on the pydantic facade (#614)
Merge remote-tracking branch 'origin/0.12.0-dev' into
docs/core-runtime-contract-ha4nw-20261004
fix(docs): cure the docs-audit issues blocking CI (#611)
Merge remote-tracking branch 'origin/fix/core-mise-pin-valid-release'… (#609)
Merge branch '0.12.0-dev' into int-test
\[WIP\] merge(integration): adopt #539 and #533 into the core stabiliza… (#608)
fix(constants): import datetime and Path for the runtime guard tuples (#607)
fix(lock): regenerate mise.lock so it parses (duplicate jscpd platform key) (#551)
Merge branch '0.12.0-dev' into fix/core-mise-lock-dedup-20260930
fix(config): own StrictYamlConfigSource in a public module — restore the root export
(#606)
Refactor/core strict imports (#605)
Merge branch '0.12.0-dev' into refactor/core-strict-imports
fix(enforcement): restore the beartype rule tag and finish the package-data catalog
cutover (#604)
fix(enforcement): migrate catalog beartype sources to predicate\_kind (val202610031920)
(#603)
chore(deps): roll flext-infra pin to 098d8b071+ — make gains test-file (val202610031920)
(#602)
fix(constants): own the runtime type-guard tuples in the constants namespace (#597)
Merge branch '0.12.0-dev' into fix/constants-runtime-type-tuples-20261003
fix(enforcement): list EnforcementInfraRuleSource in the rule-source union (#600)
feat(config): expose StrictYamlConfigSource on the package root (#599)
feat(config): expose the unique-key YAML loader on the transport facade (#598)
merge: re-travelled mise.lock — CI setup cure; publication proof carried in history
(#595)
merge: integrate the landed #596 namespace cure; this lane's remaining delta is the
re-travelled mise.lock (CI setup cure)
fix(constants): publish PRIMITIVES\_TYPES in the constants namespace (#596)
merge: re-integrate landed #593 lineage keeping runtime-proven PRIMITIVES\_TYPES
publication; part\_08 restored per tip structure
fix(gen): regenerate fleet-converged surfaces (PRIMITIVES\_TYPES publication) — make gen
exit-0 (#593)
merge: integrate 0.12.0-dev (#592 marker/stale-symbol fix) into the core-gen-publication
lane
merge: integrate develop — regenerated surfaces win conflicts (fresh gen run)
fix(constants): resolve shipped conflict markers + stale enforcement symbol at the tip
(#592)
fix(constants): import TYPE\_CHECKING in enforcement part\_07 (#591)
chore(gen): land the gen/fmt fixed-point drift on the tip (#590)
fix(deps): raise structlog floor to \>=26.1.0 (#589)
fix(lock): drop the duplicated jscpd macos-x64 platform block (#588)
integrate: fix/no-python-rules-catalog-20261001 (#584)
integrate: B-wins conflict resolution
merge: integrate 0.12.0-dev into fix/no-python-rules-catalog-20261001
integrate: fix/core-mise-pin-valid-release (#583)
merge: integrate 0.12.0-dev into fix/core-mise-pin-valid-release
Merge remote-tracking branch 'origin/fix/core-mise-pin-valid-release' into int-test
integrate: wip/stash-8be93d4e72-adopt (#586)
Merge branch '0.12.0-dev' into wip/stash-8be93d4e72-adopt
integrate: fix/core-jscpd-lock-migration-20260929 (#582)
Merge branch '0.12.0-dev' into fix/core-jscpd-lock-migration-20260929
integrate: fix/core-enforcement-flat-constants-20260929 (#581)
integrate: wip/stash-8be93d4e72-adopt (B-wins conflict resolution)
integrate: fix/no-python-rules-catalog-20261001 (B-wins conflict resolution)
integrate: fix/core-mise-pin-valid-release (B-wins conflict resolution)
integrate: fix/core-jscpd-lock-migration-20260929 (B-wins conflict resolution)
integrate: fix/core-enforcement-flat-constants-20260929 (B-wins conflict resolution)
integrate: preserve/core-tip-fix-20261001 (#585)
chore(core): tip generation adoption + E501 cures (hand-authored sources) (#580)
merge origin/0.12.0-dev — dev locks/docs win (fresher relock #1409), E501
relative-import cures kept in the generated inits
chore(upg): relock onto flext-infra 6ec104abf (#1409) past the cyclic facade binding
(flext-911cm) (#579)
feat(lazy): PR #569 recovery — restore FlextLazyMember, cure grind regressions (#578)
merge 0.12.0-dev (ea2e46be4, #577 upg onto flext-infra 9792b4789) into the #569 recovery
lane
chore(upg): adopt flext-infra tip 9792b4789 (pin survives an unchanged mise.lock) (#577)
Merge origin/chore/upg-infra-tip-20261002 (stream P preservation 10caf4020) into the upg
lane
Merge origin/0.12.0-dev into chore/upg-infra-tip-20261002
fix(metadata): enumerate installed distributions over one meta\_path snapshot
(u.installed\_distributions) (#575)
fix(index): drop the undeclared flext-infra gitlink staged by #573 (#574)
chore(core): pre-codemod verbs stable after upg with the cured publisher — checkpoint 2
(#573)
Merge remote-tracking branch 'origin/0.12.0-dev' into stab/cycle-20261001
feat(lazy): FlextLazyMember defers capability mixin members to first access (#572)
merge 0.12.0-dev (#570 upg) into feat/lazy-member-v2-20261002
chore(upg): converge flext-core onto the flext-infra tip c23721075 (#570)
\[WIP\] merge: adopt 0.12.0-dev (98c4eab2b) into the upg lane
chore(core): checkpoint the stabilization grind — WIP recorded by operator order (#571)
\[WIP\] merge: adopt 0.12.0-dev (5b8266ac6) into the lazy-member lane
chore(gen): regenerate projections — drop stale structlog\<26 constraint (#568)
build(mise): converge the toolchain trio — v2 lock, coherent pin/laun… (#567)
chore(deps): absorb the cured fleet — floors, locks and projections aligned (#566)
test(core): define public root lazy helper contract (#562)
Chore/core repin regen 20261001 (#565)
build(mise): pin mise 2026.9.16 + regenerate lock (fixes CI setup failure) (#563)
chore(deps+gen): repin cli/infra/tests to their fixed tips and regenerate (#564)
Fix/exterminate class body aliases 20261001 (#561)
fix(core): recognize LibCST visitor metadata protocol (#552)
Merge branch '0.12.0-dev' into fix/core-libcst-metadata-protocol-20260930
\[WIP\] fix(enforcement): delete the rule tables #558 restored; keep their guidance as
data (#560)
Merge remote-tracking branch 'origin/0.12.0-dev' into pr-558
Merge remote-tracking branch 'origin/0.12.0-dev' into
fix/no-python-rules-catalog-20261001
fix(logging): apply settings log level to every logger, cached ones included (#559)
merge(0.12.0-dev): bring the lane up to date
merge(0.12.0-dev): bring the lane up to the integration tip
\[WIP\] fix(core): replace class-body aliases with real types (#558)
merge(0.12.0-dev): bring the lane up to date
\[WIP\] refactor(enforcement): rules are package data, not Python tables (#557)
merge(0.12.0-dev): bring the lane up to date, preserving its changes
chore(core): integrate the fmt-fixes line into 0.12.0-dev (#556)
merge(fmt-fixes): integrate the fmt-fixes line into 0.12.0-dev
fix(core): dedup the enforcement enums and land the canonical generated forms (#550)
fix(core): make u.dump fail loud on invalid inline options (V8 S8) (#555)
fix(enforcement): report every classvar constant outside \_constants in one pass (#549)
fix(enforcement): exempt libcst METADATA\_DEPENDENCIES from classvar relocation (#554)
chore: absorb the integration tip (6c003af64 mise.lock dedup #548)
fix(core): remove the duplicated mise.lock sections (TOML parse error) (#548)
Merge branch 'fix/core-mise-lock-cure-20260930' into fix/mise-lock-toml-dedupe-20260929
merge(core): adopt the current 0.12.0-dev tip into
fix/regenerate-stale-projections-20260929
merge: sync to the tip for the lock TOML fix
\[WIP\] fix(core): one owner for the runtime type tuples (flext-hbq7n) (#547)
fix(core): project mypy wall-time budget + enum dedup (hbq7n follow-up) (#546)
fix(core): migrate the mise lock to the single-resolution era and prune duplicate enums
(#545)
Merge branch '0.12.0-dev' into fix/core-jscpd-lock-migration-20260929
chore(gen): consume the relock projections (mise 96527c3 era) (#542)
fix(core): dedupe the scrambled jscpd resolutions in mise.lock (#544)
Merge branch '0.12.0-dev' into fix/member-projections-relock-20260929
merge: sync with the integration tip for the next fix
fix(core): restore the keyword constructors and YAML source that #516 broke
(flext-hbq7n) (#543)
merge(integration): adopt #532 into the core repair lane (flext-hbq7n)
fix(core): relock onto the current flext-cli (fixes the red tip) (#532)
Merge branch '0.12.0-dev' into recovery/p0-20260927/flext-work--int--flext-core
\[WIP\] merge(integration): adopt the 21:16-21:29Z core merges into the relock lane
Recovery/p0 20260927/fleet closure lanes  core regen  flext core (#515)
Merge branch '0.12.0-dev' into
recovery/p0-20260927/fleet-closure-lanes--core-regen--flext-core
fix(core): cure 19 runtime-census findings + strict pydantic JSON typing (flext-x8799)
(#534)
Merge branch '0.12.0-dev' into fix/census-cure-045-079-20260928
Fix/exceptions root import cycle 20260929 (#537)
Merge branch '0.12.0-dev' into fix/exceptions-root-import-cycle-20260929
build(deps): update pydantic-settings requirement from \<2.15,\>=2.14.2 to
\>=2.14.2,\<2.16 (#536)
Merge branch '0.12.0-dev' into dependabot/pip/pydantic-settings-gte-2.14.2-and-lt-2.16
\[WIP\] merge(integration): adopt #539 and #533 into the core stabilization lane
fix(core): move the runtime type tuples to FlextConstantsGuards; lazy loader fails loud;
relock (#538)
Merge branch '0.12.0-dev' into fix/core-stabilization-20260929
\[WIP\] fix(enforcement): retire the whole-file regex fix actions (flext-itpd1.6) (#540)
fix(core): accept the regenerated projections and reach the gen fixed… (#541)
Merge branch '0.12.0-dev' into fix/core-gen-fixed-point-scripts-20260929
fix(core): restore importability and adopt the census-cure lane (#531)
Merge branch '0.12.0-dev' into fix/core-census-adopt-20260929
\[WIP\] preserve flext-core primary-checkout dirty state (codemod/exception typing)
(#516)
fix(enforcement): correct the c.EnforcementRuleSeverity/SourceKind constant prefixes
(#530)
fix(core): type config YAML payloads with the Pydantic JsonValue contract (#526)
Merge branch '0.12.0-dev' into wip/primary-dirty-adopt-20260928
chore: land pending primary-checkout commits (flext-core) (#533)
fix(typings): land the remaining runtime type tuples at the constants owner
(flext-x8799) (#539)
\[WIP\] merge: adopt #539 (flext-x8799 tip residue) into the core stabilization lane
Merge origin/0.12.0-dev (flext-core#535 perf enforcement short-circuit) into 0.12.0-dev
merge: absorb integration tip (parallel census cures #523/#525/#535; honest CI
retrigger)
perf(enforcement): short-circuit the field-description check before annotation eval
(#535)
Merge remote-tracking branch 'origin/0.12.0-dev' into
fix/exceptions-root-import-cycle-20260929
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
merge: bring the integration branch into the core enum reference fix
merge: adopt census-cure lane fix-forward
fix(enforcement): reference the flat c.EnforcementRuleSeverity/SourceKind owners (#525)
fix(enforcement): report every class past the ENFORCE-069 depth budget (flext-blgw9)
(#523)
Recovery/p0 20260927/flext work  int  flext core (#524)
merge: absorb integration tip 0.12.0-dev into preserve lane
recovery/p0-20260927/flext-work--int--flext-core (PR #520 adoption)
\[WIP\] preserve flext-core state from lane flext (#519)
chore(lane): absorb the integration tip (adoption after \>1h silence, curation
itpd1.3.10.11)
fix(enforcement): preserve actual import provenance (#518)
docs(agents): render the concrete raw parent URL in the standalone mode header (#522)
docs(agents): prescribe selector-free root verbs in the Commands block (#521)
fix(enforcement): audit only the classes a target declares, never the ones it references
(#513)
Merge origin/0.12.0-dev into fix/census-declared-inner-classes-20260927
fix(enforcement): route smell and bypass fix actions to manual; drop the fixer-less gate
kind (#514)
refactor: remove Pydantic model factories and adopt the stabilization branch (#512)
Merge recovery/stabilization-0.12-20260926-core into 0.12.0-dev adoption lane
Merge 0.12.0-dev tip into core stabilization lane
fix(enforcement): CONFIG\_FILENAMES is a framework ClassVar hook (#511)
fix(enforcement): derive the ENFORCE-046 peer from the lazy-import and family-surface
owners (#510)
fix(enforcement): ENFORCE-046 exempts the Pattern-B peer its catalog row declares (#509)
merge: align core stabilization with 0.12.0-dev tip
feat(protocols): p.Service is the truthful service surface (V8 S4) (#508)
Integrate current 0.12.0-dev tip into core stabilization
feat(container): truthful container with one validated write path (V8 S2) (#507)
Merge origin/0.12.0-dev into feat/v8-s2-container
feat(service): typed service operations via u.service\_operations (V8 S3) (#506)
Merge origin/0.12.0-dev into feat/v8-s3-operations
docs: repair python fences so the markdown-code gate compiles them (#505)
fix(collection): make u.process fail loud on the first item failure (#504)
fix(lazy): derive module initialization from public import semantics (#503)
fix(enforcement): derive runtime\_checkable from class source, not a private probe
(#502)
test: exterminate test-law violations in flext-core (#501)
chore(gen): regenerate member runtime projections on the V8 integration tips (#500)
feat(service): protocol ports, validated runtime seeds and typed runtime hook (V8 S1)
(#499)
Merge current integration before introspection repairs
fix(gen): regenerate projections with the facade-parent-only alias planner (#498)
fixes (#497)
\[WIP\] chore(gen): project the phase-1 wave generator (flext-itpd1.3) (#496)
Merge branch '0.12.0-dev' into fix/phase1-wave-20260924
Merge branch '0.12.0-dev' into preserve/audit-fix-20260923
merge: absorb builder-flatten lane (flat builder DSL + lazy-map cache keyed by source
map)
\[WIP\] merge: adopt the rope recovery lane (facade letters on facade parents, retired
export registries)
merge: land rope lane cures — facade letters on facade parents, retired hand-kept export
registries, fix/fmt output (flext-3csgu, flext-w2iw4)
\[WIP\] merge origin/0.12.0-dev into recovery lane (flext-w2iw4)
chore(gen): rope wave-7 generator fixed point and pydantic-settings cure (#492)
Merge remote-tracking branch 'origin/chore/regen-20260923' into
recovery/rope-automation-20260921
Merge commit 'c95b0cdc7cd360683eca50889f25ac4633420a51' into chore/regen-20260923
Merge remote-tracking branch 'origin/chore/regen-20260923' into
recovery/rope-automation-20260921
Merge remote-tracking branch 'origin/chore/regen-20260923' into
recovery/rope-automation-20260921
chore(gen): regenerate against the integrated flext-infra (#494)
merge(0.12.0-dev): absorb integration tip 2026-09-23 wave-5 (no-ff)
merge(0.12.0-dev): absorb integration tip 2026-09-23 (no-ff)
merge(0.12.0-dev): absorb integration tip 2026-09-22 wave-4 (no-ff)
fixes (#491)
Merge branch '0.12.0-dev' into fix/core-audit-reval260922
merge(0.12.0-dev): land rope gen cures — FlextLazyPart01 rename + recovered enforcement
work
fix(tests): type the beartype subprocess output local (#490)
merge(0.12.0-dev): absorb integration tip wave-2 (no-ff)
merge(0.12.0-dev): land rope recovery type-stability fixes (no-ff)
merge(0.12.0-dev): align recovery with integration tip 20260922 (no-ff)
merge: adopt integration tip (no-ff)
\[WIP\] Preserve recovery state and correct census classification (#489)
merge: 0.12.0-dev tip into branch (stabilization)
Merge remote-tracking branch 'origin/0.12.0-dev' into recovery/rope-automation-20260921
Merge remote-tracking branch 'origin/0.12.0-dev' into recovery/rope-automation-20260921
\[WIP\] absorb current Core integration into recovery (#488)
\[WIP\] absorb current Core integration into recovery
merge: absorb integration tip (dependabot cooldown propagation)
fix(security): semgrep remediation — flext-core (flext-p57t.5) (#487)
chore(recovery): consolidate Core automation history (#486)
Merge branch '0.12.0-dev' into recovery/rope-automation-20260921
chore(gen): regenerate projections with the published flext-infra d1c55b54fc
(flext-0in0k C1) (#484)
Merge branch '0.12.0-dev' into lane/c1-regen-flext-core
Merge branch '0.12.0-dev' into fix/semgrep-core-reval260921
Merge branch '0.12.0-dev' into fix/semgrep-core-reval260921
\[WIP\] recover canonical relative lazy-import resolution
\[WIP\] reconcile ConfigDict residue with unlocked dependency policy
\[WIP\] consolidate recovered Core generation snapshots
merge(0.12.0-dev): adopt published tip; keep D205 docstring from origin
merge: absorb stabilize-lane convergence (namespace scope + wave fixes)
Merge remote-tracking branch 'origin/0.12.0-dev' into
wip/stabilize-0.12-flext-core-20260919
chore(gen)+fix(enforcement): regenerate with the published generator; settings rule
reaches declared settings models (flext-0in0k C1/F0.3) (#483)
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
fix(enforcement): ENFORCE-042 targets declared settings models, not names; holders are
plain classes again (flext-0in0k.22) (#482)
Merge commit '6dc473d8d' into 0.12.0-dev
Wip/stabilize 0.12 flext core 20260919 (#481)
merge: bring the integration tip into this lane (fix-forward)
merge: bring this mirror checkout up to the published integration (fix-forward)
Merge remote-tracking branch
'refs/remotes/origin/wip/stabilize-0.12-flext-core-20260919' into
wip/stabilize-0.12-flext-core-20260919
merge: absorb rope-modernize lane into 0.12.0-dev (core #479)
chore(deps): bump actions/upload-artifact from 4 to 7 (#476)
chore(deps): bump actions/cache from 4 to 6 (#477)
chore(deps): update structlog requirement from \<26,\>=25.5.0 to \>=25.5.0,\<27 (#478)
merge: absorb integration tip (validated green state) into core primary
merge: absorb integration tip into core-479 rope-modernize lane
Merge branch '0.12.0-dev' into feature/rope-modernize
Merge remote-tracking branch 'refs/remotes/origin/feature/rope-modernize' into
feature/rope-modernize
Merge concurrent rope-modernize updates into core-479
Merge 0.12.0-dev into core-479
Merge remote rope-modernize lane into core-479
merge: absorb 0.12.0-dev tip into rope-modernize lane (fix-forward)
Merge remote-tracking branch 'origin/0.12.0-dev' into feature/rope-modernize
merge: absorb 0.12.0-dev tip into rope-modernize lane (fix-forward)
Merge branch '0.12.0-dev' into feature/rope-modernize
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
\[WIP\] Preserve reconciliation work for handoff (#475)
\[WIP\] preserve generated residue policy and canonical verb documentation (#474)
merge: integrate workspace hygiene lane into 0.12.0-dev (no-ff)
Merge current development into operational stabilization
\[WIP\] stabilize generated configuration and absorb checkout changes (#473)
Merge remote-tracking branch 'origin/feature/plan-reconciliation' into 0.12.0-dev
Merge published integration snapshot into reconciliation
Merge recorded 0.12.0-dev gitlink into plan reconciliation
merge: incorporate checkout publication before resolving root gitlinks
merge: absorb published checkout changes for stabilization
Merge 0.12.0-dev into plan reconciliation
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
chore(gen): regenerate projections from flext-infra b13793fb1 (#471)
chore(flext-core): preserve lane-r41 changes across 1 paths (#470)
chore(flext-core): preserve wip/duplicate/flext/flext-core changes across 2 paths (#469)
\[WIP\] merge origin/0.12.0-dev into the namespace lane
Merge commit '7bb0cd7b9b32ac1ca399775d1960e01f0bb566d8' into 0.12.0-dev
\[WIP\] fix(typings): m is the single owner of ConfigDict; t stops expo… (#468)
feat(models): re-export the pydantic names the abstraction boundary forces through
(#467)
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
chore(gen): regenerate projections from the corrected generator (#465)
fix(tests): the proxy stand-in satisfies the type checker it is checked by (#464)
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
fix(core): R38/R39 quality round — fix all non-baseline make check findings (#463)
Merge remote-tracking branch 'origin/0.12.0-dev' into chore/absorb-inflight-20260911b
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
fix(core): absorb 0.12.0-dev merge and fix root causes surfaced by make check (#462)
Merge branch '0.12.0-dev' into chore/absorb-inflight-20260911b
\[WIP\] merge origin/0.12.0-dev into chore/absorb-inflight-20260911b
refactor: T1+T2 architecture-scale refactor - updates for refactor to… (#461)
Merge commit '186485f180543be95027c7641bcbe0f22eda324c' into wip/dupl… (#460)
Chore/absorb inflight 20260911b (#459)
Merge branch '0.12.0-dev' into chore/absorb-inflight-20260911b
chore(flext-core): preserve constants-remediation-0.12 changes across 1 paths (#458)
Merge branch '0.12.0-dev' into constants-remediation-0.12
WIP on 0.12.0-dev: 356096d29 fix: mypy return type for list\_plugins - use tuple instead
of list
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge commit 'cef13e1d4ee26b37a9ad27f504068c6c4bd784bb' into 0.12.0-dev
Merge commit '186485f180543be95027c7641bcbe0f22eda324c' into
wip/duplicate/flext/flext-core
chore: absorb origin/0.12.0-dev into lane (fix-forward, .bak removal convergence)
chore: remove 11 tracked .bak files from interrupted safety cycle (#456)
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
fix(settings): env-file protocol has one owner — the settings layer (#454)
docs(agents): repoint ghost ADR-012 citation to ADR-005 (#453)
fix(project-metadata): ingress owns the \[tool.flext.project.budget\] table (#452)
chore(projections): absorb regenerated projections from config SSOT (#451)
gov: consumer import grammar lane — FamilySurface + atomic primitives (F1 retirement)
(#450)
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'refs/remotes/origin/0.12.0-dev' into 0.12.0-dev
wip:release/checkpoint-0.12.0 (#448)
Merge branch '0.12.0-dev' into release/checkpoint-0.12.0
Merge branch '0.12.0-dev' into release/checkpoint-0.12.0
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge commit '8e96614be09736aa6d0b8b12ff78dd9bbd64836a' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge remote-tracking branch 'origin/0.12.0-dev' into 0.12.0-dev
Merge commit '319687b8368cc7243b583ca920f1e3d9c77daac2' into 0.12.0-dev
Merge commit '611999b858dc086d009274e6f8982588dae9da5b' into release/checkpoint-0.12.0
Merge commit '0621c04cc557f4d18b9d753a28c53b08f55f0b58' into 0.12.0-dev
merge: absorb origin/0.12.0-dev (no-ff)
fix(deps): declare every direct third-party runtime import (#446)
fix(deps): declare the structlog runtime dependency (#445)
fix(deps): canonical https source URL for the flext-core self-source (#443)
fix(deps): declare the beartype runtime dependency (#442)
feat(models): advanced pydantic facade exports + gate defect adoption (#441)
merge: align with 0.12.0-dev integration (no-ff)
Merge commit 'de690c992959f14318727a0bb41eb1ecd6e16ff6' into 0.12.0-dev
merge: absorb origin/0.12.0-dev (wave-2 landing, no-ff)
merge: absorb origin/0.12.0-dev into the codemod owner-repairs lane (divergence
cooperation, no-ff)
merge: absorb origin/0.12.0-dev (divergence cooperation, no-ff)
fix(runtime): exterminate reverse root-alias imports in the lazy resolution chain (#438)
merge: land fix/fleet-toolchain-gc-fc2 into 0.12.0-dev (no-ff)
merge: land core PR #433 1wjg1.16-core-integration into 0.12.0-dev (no-ff)
merge: integrate origin/0.12.0-dev into fix/fleet-toolchain-gc-fc2 (divergence
cooperation, no-ff)
merge: integrate origin/0.12.0-dev into  (divergence cooperation, no-ff)
merge: land integrate/012-20260907 into 0.12.0-dev (flext-yirgp; operator-authorized
administrative merge, no-ff, conflict-free by construction)
merge: absorb origin/0.12.0-dev into fleet lane (flext-yirgp, no-ff)
Integration/sweep 20260830 (#435)
chore(deps): bump actions/deploy-pages from 5.0.0 to 5.0.1 (#437)
Merge branch '0.12.0-dev' into integration/sweep-20260830
\[WIP\] conformance sweep: fix-verb unlock (ruff fix=false removal), ch… (#432)
fix: isolate model imports from dependency runtime (#412)
Merge origin/0.12.0-dev into feature/flext-1wjg1.16-core-integration (integration
alignment, no-ff)
Merge branch '0.12.0-dev' into fix/model-import-runtime-isolation-20260828
Merge commit '8e96614be09736aa6d0b8b12ff78dd9bbd64836a' into fix/fleet-toolchain-gc-fc2
Merge 0.12.0-dev into feat/agents-skill-sync (integration alignment, no-ff)
\[WIP\] merge: align lane with origin/0.12.0-dev before the conformance cutover
\[WIP\] chore(docs): delete dead doc-example goldens carrying a machine-absolute path
(flext-uwhzz) (#431)
chore(fleet): sync toolchain config, regenerate lazy exports, migrate beads to shared
symlink (#429)
Merge remote integration lane into conformance checkpoint
fix(flext-core): merge fix/portable-refs into fix/fleet-toolchain-gc-fc2
Merge 0.12.0-dev into work — resolve generated-facade, docs, and toolchain conflicts
(#430)
\[WIP\] merge origin/fix/fleet-toolchain-gc-fc2 (lane A, PR #429) into
feature/flext-1wjg1.16-core-integration (flext-1wjg1.16.2)
\[WIP\] merge origin/0.12.0-dev into feature/flext-1wjg1.16-core-integration
(flext-1wjg1.16.2)
chore(integration): adopt fleet toolchain lane #424 and pass the full gate chain
(flext-1wjg1.16) (#428)
\[WIP\] merge origin/fix/fleet-toolchain-gc-fc2 (#424) into
feature/flext-1wjg1.16-core-integration
fix: dedupe pydantic symbols\_to\_replace into class-body constant (flext-1sm3w) (#424)
Merge branch '0.12.0-dev' into fix/fleet-toolchain-gc-fc2
Merge remote-tracking branch 'origin/fix/privateattr-typed-contract' into
integration/sweep-20260830
Merge remote-tracking branch 'origin/feature/typed-readme-sections' into
integration/sweep-20260830
