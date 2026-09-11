# Backend boundary procedure

## Preflight

Before changing code or infrastructure, resolve the current public request path,
service and persistence owners, schemas, tenant and authorization boundaries,
transaction semantics, cardinality and pagination limits, failure contract,
selected stack, runtime proof, consumers, and native gates.

Missing, conflicting, or inaccessible evidence stops at the first defect with
zero effects. Do not select a framework, database, cache, queue, provider,
credential, library, or undeclared default by preference. Use a canonical
calculated default when the selected typed owner already defines one.

## Necessity and ownership

Retain only structures required by a current behavior and consumer. Use the
repository's existing typed facade and composition boundary. A repository,
service, cache, queue, middleware, interface, or worker with no current consumer
is speculative and must not be introduced.

Keep one owner for each invariant:

- transport validates request shape and delegates;
- authorization and tenant scope guard every affected read and write;
- domain/service code owns business invariants;
- persistence owns query shape, transaction, ordering, and bounded pagination;
- the declared asynchronous owner owns durable enqueue, execution state, and
  causal failure when asynchronous work is actually required.

Do not copy schemas, defaults, error maps, or business rules between layers.

## Correction

1. Trace the real query and mutation graph before selecting a change.
2. Remove N+1 access by using the existing owner's bounded batch/join primitive;
   keep tenant and authorization predicates on the owning query.
3. Make multi-write invariants one atomic transaction. No success may be
   returned until the complete effect is committed.
4. Add cache or asynchronous execution only when current workload, consistency,
   invalidation, durability, and observability contracts prove necessity.
5. Propagate the first database, child-process, timeout, signal, enqueue, or
   publication failure unchanged. Never catch and normalize, retry, switch
   infrastructure, return empty success, or continue partially.
6. Rewire every current consumer and remove obsolete layers, duplicate queries,
   fallbacks, fixtures, and configuration in the same cutover.

## Proof

Exercise the public request path with valid data, missing/invalid input, tenant
isolation, empty results, boundary cardinality, and owner failure. Record query
count, transaction outcome, causal error, and response contract. Stop each native
gate invocation on its first failure, correct the canonical owner, and rerun every
invalidated gate in the same change until the complete selected set is green.

Publish only the complete verified change. Leave no partial write, unobserved
job, cache entry, old query path, compatibility layer, or generated residue.
