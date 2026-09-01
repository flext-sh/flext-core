# FLEXT development procedure

## Establish the active owners

1. Work from the authorized physical repository and read its `AGENTS.md`,
   manifests, dependency graph, configuration, generator policy, Make surface,
   CI, tests, public consumers, and generated markers.
2. In a fleet, derive members and gitlinks from the active umbrella; treat each
   member as an independent repository. Historical branches, archives, other
   checkouts, provider homes, and generated outputs are evidence, never owners.
3. Elect the smallest current owner before editing. `flext-core` owns the runtime
   foundation; runtime consumers depend toward it. `flext-infra` owns reusable
   build, conform, code-generation, and policy machinery and is not a consumer
   runtime dependency. `flext-tests` owns reusable test fixtures and helpers.

Missing, branch-mismatched, external, or ambiguous ownership stops with zero
effects. Never resolve a missing owner from another checkout or a personal home.

## Preserve the architecture

- Compose the public facade in semantic order `c -> t -> p -> m -> u`; operational
  facets include `r`, `e`, `x`, `h`, `d`, and `s`. Reverse runtime imports are
  forbidden; type-only reverse references stay under `TYPE_CHECKING`.
- Keep one thin package API/MRO facade and one generated lazy package root. Do not
  create an eager export path, custom import router, compatibility alias, renamed
  service base, parallel namespace, or duplicate facade.
- Keep configuration and typed settings at the dependency foundation. Read their
  validated public namespaces; leaf modules do not reread environment or files.
- Represent project-owned structured boundary data with its declared Pydantic 2
  model and read-only protocol contracts. Validate once at ingress, preserve the
  typed object internally, and serialize once at true egress. Declaration layers
  remain data-only; behavior belongs to utilities, services, bases, APIs, or CLIs.
- Delegate generic process, serialization, schema, template, CLI, dependency-
  injection, and test machinery to their existing public FLEXT owners. Add no
  local substitute when an owner is missing or broken.

Project-specific namespace names, analyzer exceptions, package layouts, and
domain rules remain project-owned. Reproduce a tool diagnostic against the active
declared versions before changing its owner; never add a suppression or change
valid architecture merely to silence analysis.

## Change sources and migrate atomically

Treat `config/*.yaml`, schemas, typed settings, templates, and generator policy as
canonical where the repository declares them. Edit the owner, regenerate every
affected facet, root export, managed project section, Make/CI surface, and document,
then remove the superseded implementation and consumers in the same cutover.

Do not hand-edit generated files or create a second generation route. A broken or
missing canonical Make verb is repaired generically in its FLEXT owner and then
reused. Keep third-party forks and content-only repositories outside fleet mutation
unless the operator explicitly places them in scope.

## Validate and integrate

Run the real public import, facade, service, API, CLI, or generated consumer path.
Then run generation twice and require zero second-pass diff, followed by the
repository's formatter, lint, type, test, build, documentation, and integration
gates through the active root Make dispatcher.

Land fleet members on their declared integration lanes first. Prove each integrated
SHA, then update umbrella gitlinks, regenerate the root, and rerun combined gates.
The first runtime, child process, generator, fixed-point, or gate failure escapes
causally; no fallback, retry, partial publication, or success claim is valid.
