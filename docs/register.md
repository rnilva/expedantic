# Experiment register

**Status:** Draft design
**Target:** `expedantic`
**Date:** 2026-08-26

## 1. Summary

Add a small experiment register to `expedantic` for recording what was run,
which configuration and declared inputs defined it, and whether the caller's
chosen run boundary completed.

The public design adds one declaration, `RegisterBase`, alongside `ConfigBase`
and `LoggerBase`:

```python
from pathlib import Path

from pydantic import BaseModel, model_validator

from expedantic import ConfigBase, RegisterBase


class Config(ConfigBase):
    tag: str
    seed: int = 0
    steps: int = 10_000


class DatasetSource(BaseModel):
    path: Path
    sha256: str

    @model_validator(mode="after")
    def verify_digest(self):
        # Project-owned validation. The register records the validated result;
        # it does not need a generic dataset-checker abstraction.
        return self


class Run(RegisterBase):
    _path = "results/register/runs.jsonl"

    tag: str
    seed: int
    dataset: DatasetSource
```

The preferred lifecycle spelling is a context manager:

```python
cfg = Config.parse_args(require_default_file=True)

with Run(
    config=cfg,
    tag=cfg.tag,
    seed=cfg.seed,
    dataset=DatasetSource(path="data/train.parquet", sha256="..."),
):
    result = train(cfg)
    evaluate_and_save(result, cfg)
```

Normal block exit records `completed`; an ordinary exception records `failed`;
interruption records `aborted`. The exception always propagates. The caller
chooses the lexical boundary, so the context manager does not imply that
`expedantic` owns the process.

The same instance is also a synchronous decorator through the standard
context-manager protocol:

```python
def main() -> None:
    cfg = Config.parse_args(require_default_file=True)

    @Run(config=cfg, tag=cfg.tag, seed=cfg.seed,
         dataset=DatasetSource(path="data/train.parquet", sha256="..."))
    def execute() -> None:
        result = train(cfg)
        evaluate_and_save(result, cfg)

    execute()
```

Explicit `start()` / `finish()` remains available when the boundary is not
lexical or when the caller needs a custom terminal summary.

`RegisterBase` observes declarations and lifecycle calls. It does not launch,
supervise, interrupt, resume, retry, or otherwise own execution.

The first version deliberately does **not** introduce public `RunConfig`,
`ConfigResolution`, `RunRegistry`, `RunStore`, collector, checker, artefact, or
lifecycle-event hierarchies. Typed nested declaration fields and ordinary
Pydantic validation are the extension surface.

## 2. Motivation

A training log and a configuration file do not by themselves answer basic audit
questions reliably:

- Which effective configuration did this invocation use after command-line
  overrides?
- Which committed code revision produced it, and was the tree dirty?
- Which data, checkpoint, environment contract, or implementation variant did
  the program explicitly claim to use?
- Did the caller's declared run boundary complete, fail, or never record an
  end?
- How did two recorded runs differ after their original YAML files changed?

The current `curl-field` implementation demonstrates the useful minimum:
record a full resolved configuration at start, append a terminal record, and
support later listing and configuration diffing. It also exposes design hazards
that the reusable implementation must avoid: collision-prone run identifiers,
order-sensitive reconciliation, remote PID inference, and marking a run
`finished` from a `finally` block even when training raised.

This feature belongs in `expedantic` because `ConfigBase` already performs the
configuration resolution whose result must be captured, and Pydantic already
provides the typed declaration and validation machinery needed for explicit
project provenance. It does not belong in the logger: metric rows and run
identity have different invariants.

## 3. Design principles

### 3.1 One additional public declaration

`ConfigBase` declares configuration. `LoggerBase` declares repeated measurement
fields. `RegisterBase` declares the static, queryable record attached to one
caller-defined run.

The register must not grow a parallel family of builders, registries, stores,
collectors, checkers, handles, and public event classes before multiple real
implementations require them.

### 3.2 The declaration is the primary provenance surface

Automatic host or Git observations are useful but necessarily incomplete. The
strongest provenance is often known only by the program itself: the resolved
dataset, selected parent checkpoint, environment contract, implementation
variant, generated map family, or external execution identifier.

Those facts are ordinary fields on the `RegisterBase` subclass. They may be
nested Pydantic models with validators, default factories, constrained types,
and discriminated unions. This makes project provenance explicit and type
checked in the code that understands it, rather than hidden behind a generic
OS-level discovery mechanism.

The framework snapshots the validated declaration. It does not need a public
`ProvenanceCollector` or `Checker` protocol merely to call user code that
Pydantic already expresses.

### 3.3 The caller owns the boundary

A run is not defined by a Python process, PID, seed, W&B run, or scheduler job.
It is the interval selected by the caller through a context manager, decorator,
or explicit `start()` / `finish()` pair.

The caller may place that boundary around a whole process, a single seed in a
multi-seed process, an evaluation, or another logical unit. `expedantic` records
that declaration; it does not infer the scientific unit.

A context manager does not weaken this principle. Lexical scope is simply one
precise way for the caller to declare the boundary, and it removes the most
common false-success and forgotten-finish errors.

### 3.4 Record observations, not guesses

A missing finish event means only that no finish event is present. It does not
mean “killed”, “crashed”, or even “not currently running”. Host and PID may be
recorded as observations, but they are never used to rewrite the run outcome.

Likewise, Git state is a traceability aid, not a promise of bit-for-bit
reproducibility. A project that needs a stronger code or environment contract
should declare and validate it explicitly as part of the run model.

### 3.5 Side-effect only, but not silently unreliable

Register persistence or best-effort automatic-observation failures must not
normally take down an expensive run. The default error policy emits a
`RuntimeWarning` and leaves the instance inspectably unrecorded or open. A
strict class setting or method override is available for tests and environments
where missing provenance must be fatal.

Declaration validation is different: invalid user-supplied fields and failed
user validators are ordinary programming or input errors and always raise
before the run body starts. Errors must not be swallowed without a signal.

A persistence error during exception unwinding must never replace the original
application exception. Even in strict mode, the original exception remains
primary; the register failure is warned about or attached as diagnostic context.

### 3.6 No remote experiment platform

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

The reserved `config=` argument on `RegisterBase` reads that metadata when
present. A config constructed programmatically has no parser metadata and is
recorded honestly as programmatic input.

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

### 5.1 Declaration and construction

A subclass is an ordinary Pydantic declaration:

```python
class Run(RegisterBase):
    _path = "results/register/runs.jsonl"
    _strict = False

    tag: str
    seed: int
    dataset: DatasetSource
    study: str | None = None
```

`config` is a reserved base argument and is not part of the project-declared
field namespace. The remaining values are validated as the subclass model.
Nested Pydantic models are serialised recursively and retain their validation
semantics. Public declaration fields are frozen after construction; only private
lifecycle state changes.

Construction is inert:

```python
run = Run(config=cfg, tag=cfg.tag, seed=cfg.seed, dataset=dataset)
```

It validates the declaration but does not write anything until `start()`,
`__enter__()`, or a decorated function invocation. The declaration instance is
also the lifecycle object; there is no separate public handle or registry
object.

A concrete subclass is required. Direct `RegisterBase(...)` usage is not part
of the public surface: the feature is a declaration, not an untyped service
object.

### 5.2 Context-manager mode

The preferred form is:

```python
with Run(config=cfg, tag=cfg.tag, seed=cfg.seed, dataset=dataset) as run:
    train(cfg)
    evaluate_and_save(cfg)
```

Entry performs `start()` and returns the same instance. Exit behaves as follows:

- no exception: append `completed`;
- an `Exception`: append `failed` with its qualified type and message;
- `KeyboardInterrupt`, `SystemExit`, or another non-`Exception`
  `BaseException`: append `aborted`;
- always return `False`, so the original exception propagates.

The scope must include every operation whose success is required for
completion, including validation, training, final evaluation, and saving.
Putting those operations outside the block is a caller boundary error, not a
reason to reject the context-manager API.

### 5.3 Decorator mode

The same declaration instance is a synchronous decorator; no separate decorator
primitive is introduced:

```python
@Run(config=cfg, tag=cfg.tag, seed=cfg.seed, dataset=dataset)
def execute() -> None:
    train(cfg)
    evaluate_and_save(cfg)
```

Decorator calls use exactly the context-manager semantics. Each invocation of
the decorated function reconstructs and revalidates the declaration from its
template values, then receives a fresh run UUID and fresh private lifecycle
state. This means user provenance validators run for every invocation rather
than only when the decorator is defined. Function metadata is preserved with
`functools.wraps`.

The first version deliberately does not infer register fields from function
arguments, inject a run parameter, or introduce a binding-expression DSL. The
configuration and declared provenance must be available when the decorator
instance is constructed. When that is inconvenient, use the context manager.

Async context managers and async decorators are out of scope for the first
version.

### 5.4 Explicit mode

Explicit lifecycle calls remain available for non-lexical boundaries and custom
terminal summaries:

```python
run = Run(config=cfg, tag=cfg.tag, seed=cfg.seed, dataset=dataset)
run.start()

try:
    result = execute(cfg)
except Exception as exc:
    run.finish(status="failed", error=exc)
    raise
except BaseException as exc:
    run.finish(status="aborted", error=exc)
    raise
else:
    run.finish(
        status="completed",
        summary={"last_step": result.last_step},
    )
```

Working signatures:

```python
def start(self, *, strict: bool | None = None) -> Self:
    ...


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

`start()` generates opaque UUIDs, snapshots configuration and observations,
appends one start event, and returns the same instance. `run_id` is generated
before the write and is available inside the body after the start attempt; the
read-only `recorded` state says whether the start event actually persisted.

The instance always exists. A non-strict persistence failure does not turn the
return type into `Self | None`; it emits a warning and exposes a false
`recorded` state. This keeps context-manager and decorator use total and avoids
forcing unrelated training code to branch on a missing handle.

A lifecycle instance is single-use in explicit or context-manager mode.
Starting it twice or finishing it twice is a programming error and raises before
another event is written. Decorator invocations recreate private lifecycle state
and therefore remain independently usable.

Calling `finish()` inside a context block closes the declared run immediately;
`__exit__()` then performs no second write. Consequently, explicit finish inside
a block should be the final operation when a custom summary is needed.

### 5.5 What start records

At start, the register:

1. reconstructs and revalidates the frozen declaration snapshot, so external
   provenance checks run at the actual boundary;
2. generates opaque UUIDs for the run and event;
3. snapshots configuration provenance;
4. snapshots the complete declared record, including nested project provenance;
5. samples minimal best-effort entry-point, host, and Git observations;
6. appends one start event durably;
7. exposes private lifecycle state through read-only properties.

A UUID is identity. Tags, timestamps, and configuration hashes are searchable
attributes, not identities.

## 6. Declared provenance

### 6.1 Typed nested declarations

Project provenance is not limited to scalar labels. A declaration can model
structured inputs directly:

```python
class CheckpointSource(BaseModel):
    path: Path
    sha256: str
    parent_run_id: str | None = None


class EnvironmentContract(BaseModel):
    name: str
    version: str
    digest: str


class FlowfieldRun(RegisterBase):
    _path = "results/registry/runs.jsonl"

    tag: str
    seed: int
    dataset: DatasetSource
    checkpoint: CheckpointSource | None = None
    environment: EnvironmentContract
```

The resulting start event preserves this nested structure under the declared
record. It is therefore queryable and diffable without reconstructing meaning
from tag strings or mutable files.

### 6.2 Capture and checking remain project-owned code

Users may provide provenance through ordinary constructors, class methods,
default factories, and Pydantic validators:

```python
class DatasetSource(BaseModel):
    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> "DatasetSource":
        return cls(path=path, sha256=sha256_file(path))

    @model_validator(mode="after")
    def verify(self):
        if sha256_file(self.path) != self.sha256:
            raise ValueError("dataset digest changed")
        return self
```

This is a genuine provenance check: it is explicit, typed, testable, and owned
by the code that understands the object. `RegisterBase` guarantees that only a
successfully validated declaration reaches the start event.

The framework does not need a generic checker registry. Repeated patterns may
later justify small reusable helper models or functions, but those helpers
remain ordinary values placed in a declaration.

### 6.3 Start-time and terminal facts

Facts that define what is about to run should be resolved before entering the
context or invoking the decorated function. Facts produced by the run belong in
the terminal summary or the ordinary metric/artifact outputs.

The first version has no mutable annotation event. This keeps the persistent
state machine at start plus finish. If a dynamically provisioned external ID or
input is scientifically part of the run definition, provision it before the
register boundary and declare it explicitly.

## 7. Event format

The persistent form is append-only JSON Lines with two event kinds.

### 7.1 Start event

```json
{
  "schema_version": 1,
  "event": "start",
  "event_id": "c734c520-5004-45a8-a786-d79a1b28522a",
  "run_id": "63caa4cc-32b3-4ba2-9941-f1efed672868",
  "recorded_at": "2026-08-26T04:50:00.000000Z",
  "register_type": "my_project.Run",
  "declaration": {
    "tag": "arm_a",
    "seed": 0,
    "dataset": {
      "path": "data/train.parquet",
      "sha256": "..."
    }
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
  "observed": {
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

The semantic partition is fixed:

- `declaration`: user-provided, typed, validated facts;
- `config`: the effective configuration plus parser provenance;
- `observed`: nullable, best-effort framework observations.

The framework never merges these namespaces or lets automatic observations
overwrite a user declaration.

### 7.2 Finish event

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

## 8. Persistence and reconciliation

### 8.1 One backend initially

The first version supports one JSONL path. There is no public store or sink
protocol. Introducing a backend abstraction before a second backend exists
would merely spread the same operation over more names.

Each event is encoded completely and appended through one `O_APPEND`
operating-system write. The file is flushed and `fsync`ed because only two
writes are expected per run. The implementation creates parent directories as
needed. Same-host concurrent writers must produce complete parseable lines and
are covered by a multiprocessing test. Shared network-filesystem atomicity is
not claimed.

Cross-machine transport, Git commits, merge drivers, object storage, and remote
publication remain caller policy.

### 8.2 Set semantics

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

## 9. Reading and diffing

The minimum useful read surface remains attached to the declaration:

```python
runs = Run.records()
run = Run.find("63caa4cc")
diff = Run.diff(run_a, run_b)
```

The exact names may be refined during implementation, but this functionality
must not require a second `RunRegistry` object merely to read the path already
owned by the declaration.

Diffing compares three namespaces independently:

- declared fields, including nested project provenance;
- stored complete configurations;
- automatic observations when the caller explicitly asks to inspect them.

Configuration diffing never consults current YAML files or current model
defaults. Missing keys are represented as typed add/remove operations
internally, not by a magic string such as `"<absent>"`.

A small CLI may call the same methods for `list`, `show`, and `diff`. It is a
view over the local file, not a tracking service.

## 10. Automatic observation boundary

### 10.1 Captured automatically

The first version captures only inexpensive, generally useful facts available
at lifecycle start:

- UTC start timestamp;
- entry-point name and working directory;
- hostname and PID as uninterpreted facts;
- resolved configuration and its parser provenance;
- Git commit, branch, dirty flag, and a digest of the tracked diff when inside a
  Git work tree.

Every automatic field is best-effort and nullable. Absence means unavailable,
not a clean or default state.

### 10.2 Why automatic observation remains small

Automatic collection cannot know which dataset version, generated environment,
checkpoint, semantic implementation variant, or scientific unit matters to a
project. Attempting to discover all of these would produce a tracker platform
and still make weaker claims than explicit declarations.

The automatic snapshot is therefore a baseline, not the main extension
mechanism. The declaration is where users provide stronger provenance.

### 10.3 Not captured automatically

The register does not automatically collect:

- the complete environment;
- package inventories;
- hardware inventories beyond explicitly declared fields;
- patches or untracked file contents;
- artefact hashes or uploads;
- W&B, MLflow, Slurm, Kubernetes, or cloud metadata;
- scientific treatment or pairing semantics.

Projects may declare and validate the few of these they actually need. Repeated
use across independent projects can justify later reusable helper values, but
not an open-ended collector subsystem by default.

## 11. Relationship to `LoggerBase`

`LoggerBase` records a sequence of measurements and may aggregate many values
into each row. `RegisterBase` records one immutable start and at most one
terminal event for a run declared by the caller.

They should not be unified through logger fields or logger sinks:

- metric logging may be frequent and lossy without changing run identity;
- register events require UUID identity, idempotent reconciliation, and conflict
  detection;
- logger serialisation may coerce values for a sink, while register provenance
  must have one canonical persistent representation.

Inside a context block, `run.run_id` may be logged as an ordinary logger field
when metric rows need to join to the register. No process-wide active-run global
is required.

## 12. Explicit non-goals

The first version will not:

- execute or wrap a command;
- install signal or global exception hooks;
- monitor process liveness;
- infer killed or crashed states;
- resume or retry runs;
- define experiment, trial, arm, seed, parent, or child semantics;
- infer fields from decorated function arguments;
- inject a run object into decorated functions;
- provide async lifecycle wrappers;
- add a generic collector or checker plugin protocol;
- add mutable mid-run provenance events;
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

## 13. Alternatives considered

### 13.1 Public `ConfigResolution[T]`

Rejected for the initial design. It correctly separates a value from its source
metadata, but changes the established return type of `parse_args()` and makes
all callers pay for a distinction only the register presently consumes.

### 13.2 A generic `RunRegistry` with `RunStore`, collectors, and checkers

Rejected for the initial design. There is one format, one local backend, and a
small fixed set of automatic observations. Project-specific capture and checking
are already represented by typed declaration values and Pydantic validation.
Protocols over one implementation would multiply concepts without improving the
current boundary.

### 13.3 Implementing the register as a logger

Rejected. A lifecycle record is not an aggregated metric row, and a logger sink
does not provide event identity or order-independent reconciliation.

### 13.4 Classmethod construction through `Run.start(...)`

Rejected in favour of ordinary declaration construction plus an instance
lifecycle. Returning `Self | None` from a classmethod conflicts with context and
decorator use, weakens symmetry with `ConfigBase` / `LoggerBase`, and forces the
application to branch on a missing handle after a non-strict persistence error.

### 13.5 Explicit calls only

Rejected. The caller still owns a context-manager boundary, while the context
protocol reliably maps normal exit and exceptions to terminal records. The
imperative form remains as an escape hatch rather than the only safe spelling.

### 13.6 Decorator argument-binding DSL

Rejected. A decorator that derives fields from call arguments would require
name conventions, extraction callables, or another binding language. The first
version uses a constructed declaration instance as a standard context
decorator; use a context manager when values are not available at decoration
time.

### 13.7 Automatic process supervision

Rejected. It would improve automatic terminal-state coverage but require signal
handling, process identity, heartbeats, or launcher ownership. It would also
encourage stronger claims than the available evidence supports. An honest open
record is preferable.

### 13.8 Configuration hash as run identity

Rejected. Repeated runs of an identical configuration are distinct evidence and
must remain distinct records. UUIDs identify runs; hashes compare snapshots.

## 14. Migration of `curl-field`

The first consumer should become a thin declaration with typed project
provenance:

```python
class FlowfieldRun(RegisterBase):
    _path = "results/registry/runs.jsonl"

    tag: str
    seed: int
    output: str
    environment: EnvironmentContract
    parent_checkpoint: CheckpointSource | None = None
```

The driver should be split so one small outer function owns recording and one
inner function performs the complete run:

```python
def main() -> None:
    cfg = RunConfig.parse_args(
        require_default_file=True,
        diff_print_mode="none",
    )

    with FlowfieldRun(
        config=cfg,
        tag=cfg.tag,
        seed=cfg.seed,
        output=cfg.out,
        environment=resolve_environment_contract(cfg),
        parent_checkpoint=resolve_checkpoint_source(cfg.resume),
    ):
        _run(cfg)  # validation, training, evaluation, and saving
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

## 15. Implementation sequence

### PR 1 — design

- add and review this document;
- settle `RegisterBase`, ordinary construction, and the three lifecycle
  spellings;
- make no runtime changes.

### PR 2 — core

- attach private parser provenance to `ConfigBase` instances;
- add `RegisterBase` as a Pydantic context decorator;
- add JSONL append and order-independent reconciliation;
- export `RegisterBase` from `expedantic`;
- add property-based permutation and duplication tests;
- add context success/failure/abort and repeated-decorator-call tests;
- add nested declared-provenance validation and serialisation tests;
- add failure-policy, secret-serialisation, malformed-line, and conflict tests.

### PR 3 — first consumer

- replace `flowfield.registry` with a `RegisterBase` declaration;
- declare the project provenance that currently survives only in tag strings,
  mutable files, or external systems;
- move the lifecycle boundary around the complete run;
- adapt the existing list/show/diff utility;
- migrate the historical ledger;
- retain W&B-specific import logic outside `expedantic`.

## 16. Acceptance criteria

The implementation is acceptable when:

1. existing `ConfigBase.parse_args()` call sites remain source-compatible;
2. a user can define a register with one Pydantic subclass;
3. the same instance shape supports context-manager, decorator, and explicit
   lifecycle modes without a second public handle;
4. context-manager normal exit, failure, and interruption produce the specified
   terminal status and never suppress the application exception;
5. repeated calls of a decorated function receive distinct run IDs;
6. a frozen declaration may contain nested typed provenance whose validators
   rerun at lifecycle start before the body begins;
7. automatic observations and user declarations remain separate namespaces;
8. two runs with the same tag and start second receive different IDs;
9. materialisation is invariant to event order and duplicate identical events;
10. conflicting events are surfaced rather than resolved by line order;
11. concurrent local writers cannot create interleaved or malformed event lines;
12. a raised exception cannot be recorded as completed by the recommended use;
13. a missing finish remains open without PID-based inference;
14. a non-strict persistence failure does not make the lifecycle object disappear;
15. no optional tracking service or heavy dependency is introduced;
16. the public API adds no storage, collector, or checker abstraction;
17. the complete feature can be understood without knowledge of W&B, Hydra,
    MLflow, or `curl-field`.

## 17. Growth rule

A new public primitive is added only when all three are true:

1. at least two independent in-tree uses need the distinction;
2. expressing it through declared fields, nested Pydantic models, or an existing
   lifecycle method is materially unsafe or repetitive;
3. the new abstraction removes more concepts from callers than it adds to the
   library.

This rule is intended to prevent the register from gradually becoming an
experiment platform by accumulation.
