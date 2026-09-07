# REST contract procedure

## Preflight

Read the project rules, existing resource/schema owner, routes, consumers,
authentication and authorization policy, naming and error conventions, current
version, and native contract gates. For a collection, also resolve workload,
stable unique ordering, mutation behavior, page-size policy, and consumer needs.
Missing or conflicting evidence blocks design; do not fill it with a generic REST
default.

## Derive the contract

1. Model the current domain capability as resources and relationships. Reuse the
   project's established URL, field, envelope, and version authority.
2. Select each HTTP method from the observable operation and its idempotency.
   Define success and domain-error statuses without turning failures into `200`,
   empty collections, alternate resources, or generic internal errors.
3. State request and response schemas, required fields, bounds, content type,
   authentication, resource-level authorization, and information-disclosure
   behavior. Keep the schema owner singular; documentation and generated clients
   are projections.
4. For collections, choose pagination only from the preflight evidence. A cursor
   encodes the complete stable ordering tuple and remains opaque to consumers.
   Bounds come from owned policy, never an arbitrary number in this skill.
5. For mutations, validate every prerequisite before the first effect. Apply the
   repository's transaction and idempotency owners so the operation publishes
   completely or leaves zero effects.
6. Preserve the first causal application or dependency failure. Do not retry,
   fall back to another provider or operation, publish a partial response, or
   catch a failure merely to normalize it.

## Evolution

Classify a change from current consumer and schema evidence. Rewire affected
consumers and remove the superseded contract in one approved cutover. Do not keep
old and new routes, envelopes, versions, or fields as an unowned compatibility
path. If product policy requires a staged migration, its owner, end condition,
and removal proof are prerequisites rather than defaults supplied here.

## Output and proof

Return one concrete contract: resource and method, parameters, schemas, statuses,
errors, authentication and authorization, pagination or idempotency when needed,
owner, affected consumers, and native contract/runtime proof. Reject generic HTTP
advice unrelated to the supplied project. Completion requires schema validation,
material happy-path output, first-failure/zero-effect proof, consumer rewiring,
and no obsolete contract residue.
