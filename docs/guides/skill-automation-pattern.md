<!-- AUTO-GENERATED FILE — regenerate through `make gen` from the workspace root. -->
<!-- Source of truth: `docs/guides/skill-automation-pattern.md`; adjust that source, never this projection. -->

# flext-core - Skill Automation Pattern

> Project profile: `flext-core`

<!-- TOC START -->
- [Ownership](#ownership)
- [Required change shape](#required-change-shape)
- [Canonical execution](#canonical-execution)
<!-- TOC END -->

Skills document intent and route execution to canonical owners. They do not
create parallel rule engines, registries, scripts, command grammars, or manual
consumer rewiring.

## Ownership

- Typed configuration owns enforceable policy data.
- Canonical `c`, `t`, `p`, `m`, and `u` facades own reusable declarations and
  behavior.
- flext-infra owns semantic discovery, ast-grep/Rope transformations, local LSP
  analysis, Git repositories, generation, and enforcement.
- ai-hub owns GitHub and CRG runtime services. FLEXT may consume its public
  commands, hooks, MCP routes, and daemons as optional enrichment, never as a
  library dependency; absence of that runtime is not an error.
- A skill points to those owners and explains when to use them.

Generated baselines, projections, and reports are evidence, never a writable
policy source or an allowlist.

## Required change shape

1. Research the current owner, consumers, fallbacks, tests, docs, and generated
   projections.
2. Encode the generalized invariant at its typed flext-infra owner.
3. Add a semantic transformation that can reproduce every consumer rewire.
4. Rewire consumers, remove the old owner, and prove zero residue.
5. Update the skill and canonical documentation in the same change.
6. Regenerate and run every declared gate.

Tests for skill automation use public facades, `tm`, the unified `conftest.py`,
and typed shared fixtures. Mocks, fakes, stubs, patching, private construction,
and hardcoded project-owned values are prohibited.

## Canonical execution

Run only from the workspace root:

```bash
make setup
make help
make gen
make mod
make gen
make gen
make fix
make fmt
make check
make test
make conform
make waza
```

The final generation run proves the fixed point. `make mod` owns
structural transformations; no direct script or tool invocation is valid.
`make test` always retains Testmon.

Do not add project, file, pattern, action, phase, fix, or changed-only selectors.
A missing capability is implemented at the canonical Make/flext-infra owner
before the declared verb is rerun.
