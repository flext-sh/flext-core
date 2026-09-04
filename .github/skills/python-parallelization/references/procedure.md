# Python parallelization procedure

## Establish evidence

Measure the sequential baseline and locate the constrained work. Classify it as
CPU-bound Python, blocking I/O, cooperative asynchronous I/O, or vectorizable
library work. Record input sizes, latency and throughput goals, memory limits,
dependency safety, ordering requirements, and external concurrency limits.

## Select the smallest correct primitive

- Prefer existing library vectorization for measured array or tabular work.
- Use async only when the complete call chain supports non-blocking I/O and
  cancellation.
- Use bounded threads for blocking I/O only when dependency thread-safety and
  resource limits are established.
- Use processes for measured CPU-bound Python only when serialization, startup,
  memory, and shutdown costs remain bounded.

Define ownership, maximum in-flight work, backpressure, timeout, cancellation,
result ordering, first-failure behavior, atomic publication, and cleanup before
implementation. Never create an unbounded task set, detach owned workers, retry,
fall back to another primitive, hide worker errors, or publish partial results.

## Prove the change

Use deterministic inputs to prove result and failure equivalence. Exercise
timeout, cancellation, partial failure, shutdown, and resource cleanup. Compare
throughput, tail latency, CPU, memory, and open-resource counts with the same
sequential baseline. Keep the sequential implementation when the measured gain
does not justify the coordination cost; this is the preflight decision, not an
error-triggered fallback. The first worker failure remains causal, pending work is
cancelled, cleanup attaches secondary failure, and zero results are published.
