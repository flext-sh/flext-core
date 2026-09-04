# SOLID remediation procedure

## Evidence gate

Consume the current consumers retained by `yagni` and the authority map selected
by `ssot`; `search-first` owns the global route. Before restructuring, resolve
public inputs and outputs, errors and causes, side effects, ordering, lifecycle,
performance and security constraints, callers, construction, runtime, and native
contract gates. Missing observable responsibility, variation, or substitutability
evidence blocks redesign.

## Five decisions

- **SRP:** separate policy, I/O, transport, presentation, and orchestration only
  when real consumers give them distinct reasons to change. Size alone is not a
  violation.
- **OCP:** reuse a stable variation boundary only when a second current
  implementation shares its behavioral contract. Do not add speculative plugins,
  registries, factories, or switches.
- **LSP:** every implementation preserves preconditions, postconditions, errors,
  effects, and ordering through the same contract suite. Model separate
  capabilities rather than weaken a contract.
- **ISP:** shape the smallest consumer-owned capability. Split interfaces that
  force no-op, stub, exception, or unused methods; do not create one interface per
  class without substitution.
- **DIP:** policy depends on typed capabilities; the established composition root
  selects and injects concrete I/O. Service locators, globals, dynamic lookup,
  provider branches, and untyped bags are not dependency inversion.

## Atomic remediation

1. Add or strengthen public/contract proof for the confirmed violation.
2. Introduce or reuse the smallest consumer-owned boundary at the SSOT owner.
3. Rewire every current consumer and composition path before publication.
4. Remove concrete construction from policy, provider/type switches, duplicated
   rules, broad contracts, dead implementations, fixtures, shims, and old paths.
5. Recheck `yagni`, `ssot`, and all five decisions; apply `simplify` to the final
   graph.

Run the real public runtime and one contract suite across every implementation,
including the first causal error and zero-effect failure. Then run the declared
native gates and search for superseded construction or duplicate owners. Never
keep old/new paths, normalize failure through catch/retry/fallback, or claim SOLID
from pattern names alone.
