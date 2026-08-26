# Experiment register

**Status:** Draft design
**Target:** `expedantic`
**Date:** 2026-08-26

## 1. Summary

Add a small experiment register to `expedantic` for recording what was run and
under which configuration and code state.

The public design has one new declaration:

```python
from expedantic import ConfigBase, RegisterBase


class Config(ConfigBase):
    tag: str
    seed: int = 0
    steps: int = 10_000


class Run(RegisterBase):
    _path = "results/register/runs.jsonl"

    tag: str
    seed: int
```

A caller explicitly marks the beginning and end of the part of the program it
considers one run:

```python
cfg = Config.parse_args(require_default_file=True)
run = Run.start(config=cfg, tag=cfg.tag, seed=cfg.seed)

try:
    result = train(cfg)
    evaluate_and_save(result, cfg)
except BaseException as exc:
    run.finish(status="failed", error=exc)
    raise
else:
    run.finish(
        status="completed",
        summary={"last_step": result.last_step},
    )
```

`RegisterBase` observes these two explicit calls. It does not launch, supervise,
interrupt, resume, or otherwise own the process.

The first version deliberately does **not** introduce public `RunConfig`,
`ConfigResolution`, `RunRegistry`, `RunStore`, collector, artefact, or lifecycle
event abstractions. Those names describe possible implementation details or
future extension points, not primitives justified by the present use case.

## 2. Motivation

A training log and a configuration file do not by themselves answer basic audit
questions reliably:

- Which effective configuration did this invocation use after command-line
  overrides?
- Which committed code revision produced it, and was the tree dirty?
- Did the invocation complete, fail, or never record an end?
- How did two recorded runs differ after their original YAML files changed?

The current `curl-field` implementation demonstrates the useful minimum:
record a full resolved configuration at start, append a terminal record, and
support later listing and configuration diffing. It also exposes design hazards
that the reusable implementation must avoid: collision-prone run identifiers,
order-sensitive reconciliation, remote PID inference, and marking a run
`finished` from a `finally` block even when training raised.

This feature belongs in `expedantic` because `ConfigBase` already performs the
configuration resolution whose result must be captured. It does not belong in
the logger: metric rows and run identity have different invariants.

## 3. Design principles

### 3.1 One additional public declaration

`ConfigBase` declares configuration. `LoggerBase` declares repeated measurement
fields. `RegisterBase` declares the static, queryable fields attached to one
run.

The register must not grow a parallel family of builders, registries, stores,
collectors, handles, and event types before there are multiple real
implementations requiring them.

### 3.2 The caller owns the boundary

A run is not defined by a Python process, PID, seed, W&B run, or scheduler job.
It is the interval between the caller's `start()` and `finish()` calls.

The caller may place that boundary around a whole process, a single seed in a
multi-seed process, an evaluation, or another logical unit. `expedantic` records
that declaration; it does not infer the scientific unit.

### 3.3 Record observations, not guesses

A missing finish event means only that no finish event is present. It does not
mean “killed”, “crashed”, or even “not currently running”. Host and PID may be
recorded as provenance, but they are never used to rewrite the run outcome.

Likewise, Git provenance records the observed commit and dirty state. It is a
traceability aid, not a promise of bit-for-bit reproducibility.

### 3.4 Side-effect only, but not silently unreliable

Register persistence or automatic-provenance failures must not normally take
down an expensive run. The default error policy is therefore to emit a
`RuntimeWarning` and return failure from the register operation. A strict mode
is available for tests and environments where missing provenance must be fatal.

Declaration validation is different: invalid user-supplied fields are ordinary
programming or input errors and always raise. Errors must not be swallowed
without any signal.

### 3.5 No remote experiment platform

The register is a local, inspectable record. It has no server, database service,
background worker, remote synchronisation, dashboard, sweep engine, scheduler,
model registry, or metric backend.

## 4. Configuration provenance without a new configuration wrapper

### 4.1 Decision

Do not add a public `RunConfig` or make `parse_args()` return a
`ConfigResolution[T]` wrapper.

`ConfigBase.parse_args()` and `ConfigBase.load_from_yaml()` continue to return
the configuration instance exactly as they do today. During parsing they attach
private provenance metadata to the root `ConfigBase` instance:

```python
cfg = Config.parse_args(...)
assert isinstance(cfg, Config)
```

`RegisterBase.start(config=cfg, ...)` reads that metadata when present. A config
constructed programmatically has no parser metadata and is recorded honestly as
programmatic input.

### 4.2 Captured configuration facts

The start record contains:

- the fully qualified `ConfigBase` class name;
- the complete final configuration, using Pydantic JSON serialisation;
- the configuration-file path, when one was used;
- a SHA-256 digest of that file's bytes;
- the normalised command-line override mapping;
- whether the configuration was parsed, loaded from YAML, or constructed
  programmatically.

The final configuration is authoritative. The source information explains how
it was obtained.

Raw `argv` is not stored by default. Parsed overrides are more useful and can
respect Pydantic's existing secret/exclusion semantics. `SecretStr` and fields
marked `exclude=True` remain the supported way to keep values out of serialised
configuration; the register does not invent a second redaction system.

### 4.3 Why private metadata is sufficient

The register is presently the only consumer of the parsing layers. A public
resolution wrapper would force every existing call site to unwrap `.value` and
would create an abstraction whose principal purpose is transporting internal
metadata from `ConfigBase` to `RegisterBase`.

Should another independent consumer later require structured provenance, the
private metadata can be promoted to a public read-only property without
changing `parse_args()`'s return type.

## 5. `RegisterBase`

### 5.1 Declaration

A subclass declares project-specific fields using ordinary Pydantic
annotations:

```python
class Run(RegisterBase):
    _path = "results/register/runs.jsonl"

    tag: str
    seed: int
    study: str | None = None
```

These fields are validated once at `start()` and stored under a distinct
`fields` object in the event. They cannot collide with framework keys such as
`run_id`, `event`, or `recorded_at`.

Subclassing is optional. `RegisterBase.start(...)` may be used directly when no
additional top-level fields are needed.

### 5.2 Start

Working signature:

```python
@classmethod
def start(
    cls,
    *,
    config: ConfigBase | None = None,
    path: str | Path | None = None,
    strict: bool = False,
    **fields: object,
) -> Self | None:
    ...
```

`path` overrides `_path`. Invalid declared fields raise before any event is
written; `None` is reserved for a non-strict persistence failure. At start, the
register:

1. validates the declared fields;
2. generates opaque UUIDs for the run and event;
3. snapshots configuration provenance;
4. samples minimal invocation, host, and Git provenance;
5. appends one start event durably;
6. returns the run instance carrying private `run_id`, path, and closed state.

A UUID is identity. Tags, timestamps, and configuration hashes are searchable
attributes, not identities.

### 5.3 Finish

Working signature:

```python
def finish(
    self,
    *,
    status: Literal["completed", "failed", "aborted"] = "completed",
    summary: Mapping[str, JsonValue] | None = None,
    error: BaseException | None = None,
    strict: bool | None = None,
) -> bool:
    ...
```

A finish event contains the terminal status, timestamp, optional JSON summary,
and, for an error, its qualified type and message. Tracebacks remain in the
ordinary application log; they are not duplicated into the register by
default.

Finishing the same instance twice is rejected in strict mode and warned about
otherwise. It must not append two terminal events accidentally.

### 5.4 No context manager in the first version

A context manager is convenient but encourages callers to equate lexical scope
with the run before the correct boundary is settled. In particular, recording
completion on block exit can still be wrong when final evaluation or artefact
saving occurs outside the block.

The first API therefore keeps start and finish explicit. A context-manager
convenience may be added later without introducing a new primitive if repeated
in-tree use demonstrates that it removes real boilerplate without obscuring the
boundary.

## 6. Event format

The persistent form is append-only JSON Lines with two event kinds.

### 6.1 Start event

```json
{
  "schema_version": 1,
  "event": "start",
  "event_id": "c734c520-5004-45a8-a786-d79a1b28522a",
  "run_id": "63caa4cc-32b3-4ba2-9941-f1efed672868",
  "recorded_at": "2026-08-26T04:50:00.000000Z",
  "register_type": "my_project.Run",
  "fields": {
    "tag": "arm_a",
    "seed": 0
  },
  "config": {
    "type": "my_project.Config",
    "source": "arguments",
    "values": {},
    "file": {
      "path": "configs/arm_a.yaml",
      "sha256": "..."
    },
    "cli_overrides": {}
  },
  "provenance": {
    "entrypoint": "train.py",
    "working_directory": "/workspace/project",
    "host": "worker-1",
    "pid": 1234,
    "git_commit": "...",
    "git_branch": "main",
    "git_dirty": true,
    "git_diff_sha256": "..."
  }
}
```

The exact field spelling is an implementation detail until the first release,
but the semantic partition is fixed: declared fields, configuration snapshot,
and execution provenance are separate.

### 6.2 Finish event

```json
{
  "schema_version": 1,
  "event": "finish",
  "event_id": "f477fc1c-a5ef-42ba-b295-98ca67e7fc1c",
  "run_id": "63caa4cc-32b3-4ba2-9941-f1efed672868",
  "recorded_at": "2026-08-26T06:13:12.000000Z",
  "status": "completed",
  "summary": {
    "last_step": 39999
  }
}
```

A run with a valid start and no finish materialises as `open`.

## 7. Persistence and reconciliation

### 7.1 One backend initially

The first version supports one JSONL path. There is no public store or sink
protocol. Introducing a backend abstraction before a second backend exists
would merely spread the same operation over more names.

Each event is encoded completely and appended in one operating-system write.
The file is flushed and `fsync`ed because only two writes are expected per run.
The implementation creates parent directories as needed.

Cross-machine transport, Git commits, merge drivers, object storage, and remote
publication remain caller policy.

### 7.2 Set semantics

Although JSONL has an order, reconciliation must behave as a set operation. For
an event collection `E`, permutation `pi`, and duplicate subset `D`:

```text
materialise(E) == materialise(pi(E) + D)
```

This is required because concatenation, Git union merges, retries, and manual
combination may reorder or duplicate lines.

Rules:

- deduplicate identical events by `event_id`;
- report an integrity conflict if one `event_id` has different payloads;
- group events by `run_id` independent of line order;
- require one non-conflicting start event;
- require at most one non-conflicting finish event;
- materialise no finish as `open`;
- surface conflicting starts or finishes rather than choosing the last line;
- warn with a line number for malformed input, or raise in strict mode.

Host and PID never participate in reconciliation.

## 8. Reading and diffing

The minimum useful read surface remains attached to the declaration:

```python
runs = Run.records()
run = Run.find("63caa4cc")
diff = Run.diff(run_a, run_b)
```

The exact names may be refined during implementation, but this functionality
must not require a second `RunRegistry` object merely to read the path already
owned by the declaration.

Configuration diffing compares the stored complete configurations, not current
YAML files or current model defaults. Missing keys are represented as typed
add/remove operations internally, not by a magic string such as `"<absent>"`.

A small CLI may call the same methods for `list`, `show`, and `diff`. It is a
view over the local file, not a tracking service.

## 9. Provenance boundary

### 9.1 Captured automatically

The first version captures only inexpensive, generally useful facts available
at `start()`:

- UTC start timestamp;
- entry-point name and working directory;
- hostname and PID as uninterpreted facts;
- resolved configuration and its parser provenance;
- Git commit, branch, dirty flag, and a digest of the tracked diff when inside a
  Git work tree.

Every automatic field is best-effort and nullable. Absence means unavailable,
not a clean or default state.

### 9.2 Supplied by the declaration

Project-specific facts belong in declared fields or the finish summary:

- tag, study, seed, group, or arm label;
- output paths;
- scheduler identifiers;
- last completed iteration;
- domain-specific terminal measurements;
- external service IDs.

No dedicated primitive is needed for each category.

### 9.3 Not captured automatically

The register does not automatically collect:

- the complete environment;
- package inventories;
- hardware inventories beyond explicitly declared fields;
- patches or untracked file contents;
- artefact hashes or uploads;
- W&B, MLflow, Slurm, Kubernetes, or cloud metadata;
- scientific treatment or pairing semantics.

Projects may declare the few of these they actually need. Repeated use across
independent projects can justify later first-class support.

## 10. Relationship to `LoggerBase`

`LoggerBase` records a sequence of measurements and may aggregate many values
into each row. `RegisterBase` records one immutable start and at most one
terminal event for an execution declared by the caller.

They should not be unified through logger fields or logger sinks:

- metric logging may be frequent and lossy without changing run identity;
- register events require UUID identity, idempotent reconciliation, and conflict
  detection;
- logger serialisation may coerce values for a sink, while register provenance
  must have one canonical persistent representation.

A project may declare `run_id` as an ordinary logger field when it needs to join
metric rows to the register. No global active-run context is required in the
first version.

## 11. Explicit non-goals

The first version will not:

- execute or wrap a command;
- install signal or exception hooks;
- monitor process liveness;
- infer killed or crashed states;
- resume or retry runs;
- define experiment, trial, arm, seed, parent, or child semantics;
- log time-series metrics;
- upload or manage artefacts;
- provide remote backends or synchronisation;
- initialise or mirror W&B/MLflow runs;
- generate sweeps or compose configurations;
- add a server, UI, database daemon, or background thread;
- guarantee reproducibility from the record alone.

These exclusions are part of the product definition, not merely deferred work.
They keep `expedantic` a configuration and local-record library rather than a
workflow framework or experiment platform.

## 12. Alternatives considered

### 12.1 Public `ConfigResolution[T]`

Rejected for the initial design. It correctly separates a value from its source
metadata, but changes the established return type of `parse_args()` and makes
all callers pay for a distinction only the register presently consumes.

### 12.2 A generic `RunRegistry` with `RunStore` and collectors

Rejected for the initial design. There is one format, one local backend, and one
small set of automatic observations. Protocols over a single implementation do
not yet reduce coupling; they only multiply concepts.

### 12.3 Implementing the register as a logger

Rejected. A lifecycle record is not an aggregated metric row, and a logger sink
does not provide event identity or order-independent reconciliation.

### 12.4 Automatic process supervision

Rejected. It would improve automatic terminal-state coverage but require signal
handling, process identity, heartbeats, or launcher ownership. It would also
encourage stronger claims than the available evidence supports. An honest open
record is preferable.

### 12.5 Configuration hash as run identity

Rejected. Repeated runs of an identical configuration are distinct evidence and
must remain distinct records. UUIDs identify runs; hashes compare snapshots.

## 13. Migration of `curl-field`

The first consumer should become a thin declaration:

```python
class FlowfieldRun(RegisterBase):
    _path = "results/registry/runs.jsonl"

    tag: str
    seed: int
    output: str
```

The driver should be split so one small outer function owns recording and one
inner function performs the run:

```python
def main() -> None:
    cfg = RunConfig.parse_args(
        require_default_file=True,
        diff_print_mode="none",
    )
    run = FlowfieldRun.start(
        config=cfg,
        tag=cfg.tag,
        seed=cfg.seed,
        output=cfg.out,
    )

    try:
        result = _run(cfg)  # validation, training, evaluation, and saving
    except BaseException as exc:
        if run is not None:
            run.finish(status="failed", error=exc)
        raise
    else:
        if run is not None:
            run.finish(
                status="completed",
                summary={"last_iteration": result.last_iteration},
            )
```

This corrects the existing false-success path: completion is written only after
validation, training, final evaluation, and saving have all succeeded.

The project-local W&B importer remains project-local. It may translate W&B
history into the two generic events, but `expedantic` does not acquire a W&B
dependency or provider abstraction.

The existing ledger is migrated once by assigning deterministic event IDs from
its original rows and preserving the old identifier as a declared legacy field.
Any collision in the old tag-plus-second identifier is reported rather than
silently folded.

## 14. Implementation sequence

### PR 1 — design

- add this document;
- settle the name `RegisterBase` and the exact small public signatures;
- make no runtime changes.

### PR 2 — core

- attach private parser provenance to `ConfigBase` instances;
- add `RegisterBase` and JSONL append/reconciliation;
- export `RegisterBase` from `expedantic`;
- add property-based permutation and duplication tests;
- add failure-policy, secret-serialisation, malformed-line, and conflict tests.

### PR 3 — first consumer

- replace `flowfield.registry` with a `RegisterBase` declaration;
- move the finish boundary after the complete run;
- adapt the existing list/show/diff utility;
- migrate the historical ledger;
- retain W&B-specific import logic outside `expedantic`.

## 15. Acceptance criteria

The implementation is acceptable when:

1. existing `ConfigBase.parse_args()` call sites remain source-compatible;
2. a user can define and use a register with one subclass and two explicit
   lifecycle calls;
3. two runs with the same tag and start second receive different IDs;
4. materialisation is invariant to event order and duplicate identical events;
5. conflicting events are surfaced rather than resolved by line order;
6. a raised exception cannot be recorded as completed by the recommended use;
7. a missing finish remains open without PID-based inference;
8. no optional tracking service or heavy dependency is introduced;
9. the public API adds no storage or collector abstraction;
10. the complete feature can be understood from this document and a small
    example without knowledge of W&B, Hydra, MLflow, or `curl-field`.

## 16. Growth rule

A new public primitive is added only when all three are true:

1. at least two independent in-tree uses need the distinction;
2. expressing it through declared fields or an existing method is materially
   unsafe or repetitive;
3. the new abstraction removes more concepts from callers than it adds to the
   library.

This rule is intended to prevent the register from gradually becoming an
experiment platform by accumulation.
