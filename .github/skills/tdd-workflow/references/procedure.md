# Test-driven development

## Contract

TDD proves observable behavior. Runtime evidence and an approved public
contract define the target; tests do not invent or rewrite that contract.

## Procedure

1. Read project law, the public surface, affected consumers, and the native
   focused test command.
2. Observe the real behavior before changing a test. If the intended behavior
   is unresolved, stop with the missing decision.
3. Add the smallest test that reproduces one observable requirement or defect.
   Assert outputs, effects, errors, ordering, or durable artifacts rather than
   implementation structure.
4. Run the native focused command and capture the expected RED with command,
   working directory, exit code, and decisive output. A test that passes before
   the correction does not prove the defect. Preserve any unexpected nonzero,
   timeout, signal, exception, and causal chain; do not retry or select another
   command to obtain the expected failure.
5. Correct the canonical owner with the minimum complete implementation. Do not
   add fallback behavior, dual paths, hardcoded results, or weakened assertions.
6. Rerun the same focused command to GREEN, then run affected integration and
   public-surface checks declared by the project.
7. Refactor only while the same behavior remains green. Remove superseded code
   and rewire every affected consumer in the same atomic change. A prerequisite
   failure before RED leaves the test, owner, and runtime unchanged.

## Test selection

- Unit tests cover pure behavior and boundary conditions.
- Integration tests cover real component contracts and persistence boundaries.
- End-to-end tests cover critical user or operator journeys through the public
  surface.
- Use test doubles only at a genuine external boundary and assert the observable
  contract, never internal calls.
- Coverage reports reveal missing paths; an arbitrary percentage is not proof
  of correctness.

## Completion

Report the RED and GREEN commands, exact covered behavior, runtime artifact,
affected consumers, first causal failure, zero-effect proof, residue search, and
remaining blocker. Never claim completion from a test name, test count, or
coverage percentage alone.
