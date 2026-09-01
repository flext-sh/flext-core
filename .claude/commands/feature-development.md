---
description: "Implement one approved feature through project owners, consumers, runtime, and gates."
argument-hint: "<approved feature contract or specification>"
disable-model-invocation: true
---

# Feature development

Treat `$ARGUMENTS` as the approved feature contract. Refuse implementation when
the requested outcome, consumer, boundary, or acceptance behavior is genuinely
ambiguous.

When the input is an implementation plan, consume its approved
`implementation-plan-review` result. A blocker or unresolved material decision
keeps implementation closed; do not silently complete the plan while coding.

1. Read project instructions, architecture, public interfaces, existing owners,
   dependencies, consumers, runtime entry points, tests, and native gates.
2. Search for equivalent or reusable behavior. Apply YAGNI to remove unsupported
   scope, elect one SSOT, and change design boundaries only where the surviving
   requirement proves they are necessary.
3. Add a focused failing acceptance test when the project contract calls for
   tests, then implement the complete feature through the selected owner.
   Simplify each changed unit inline and remove obsolete paths while rewiring all
   current consumers, documentation, examples, and generated projections.
4. Exercise the real public feature before broad tests. Propagate failures
   loudly; never add fallback, hardcode, suppression, compatibility residue, or
   a weaker assertion to obtain green output.
5. Run every affected native format, lint, type, test, build, security,
   documentation, generation, and integration gate.

Return the accepted behavior, owner and reuse decision, consumers changed,
runtime proof, exact gate exits, and any blocker. Do not describe a partial,
branch-only, or unintegrated implementation as complete.
