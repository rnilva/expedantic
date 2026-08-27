# Experiment register

**Status:** Draft design  
**Target:** `expedantic`  
**Date:** 2026-08-26

## 1. Summary

Add a small local experiment register to `expedantic`.

The public surface adds one generic Pydantic declaration, `RunBase[C]`, where
`C` is the project's `ConfigBase` subtype. A concrete run class states how a
validated configuration becomes a complete run declaration through one
project-owned class method:

```python
from pathlib import Path
from typing import ClassVar
from typing_extensions import Self

from pydantic import BaseModel, model_validator

from expedantic import ConfigBase, RunBase


class Config(ConfigBase):
    tag: str
    seed: int = 0
    dataset: Path


class DatasetSource(BaseModel):
    path: Path
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> "DatasetSource":
        return cls(path=path, sha256=sha256_file(path))

    @model_validator(mode="after")
    def verify(self) -> Self:
        if sha256_file(self.path) != self.sha256:
            raise ValueError("dataset changed")
        return self


class Run(RunBase[Config]):
    _path: ClassVar[Path] = Path("results/register/runs.jsonl")

    tag: str
    seed: int
    dataset: DatasetSource

    @classmethod
    def from_config(cls, config: Config, /) -> Self:
        return cls(
            config=config,
            tag=config.tag,
            seed=config.seed,
            dataset=DatasetSource.capture(config.dataset),
        )
```

The preferred lifecycle is a context manager:

```python
cfg = Config.parse_args(require_default_file=True)

with Run.from_config(cfg) as run:
    result = train(cfg)
    evaluate_and_save(result, cfg)
```

The same instance is a synchronous decorator:

```python
@Run.from_config(cfg)
def execute() -> None:
    result = train(cfg)
    evaluate_and_save(result, cfg)
```

Explicit `start()` and `finish()` remain available for a non-lexical boundary
or a custom terminal summary.

`RunBase` records the declaration and lifecycle selected by the caller. It does
not launch, supervise, interrupt, resume, retry, or otherwise own execution.

## 2. Why `RunBase`, not `RegisterBase`

A model instance represents one run, not the collection holding many runs.
`RunBase` therefore matches `ConfigBase` and `LoggerBase` more accurately:

- `ConfigBase` declares one configuration;
- `LoggerBase` declares one measurement schema;
- `RunBase` declares one recorded run.

The append-only file is the register. No public registry or store object is
needed for the first implementation.

## 3. Design principles

### 3.1 One new declaration

The feature must not introduce a parallel family of public builders, handles,
collectors, checkers, stores, artefact references, or lifecycle-event classes.
The concrete `RunBase` subclass is the extension surface.

### 3.2 Configuration is the binding boundary

A run class is generic in its configuration type and must implement
`from_config()`. This is the explicit mapping from configuration to run
provenance. It replaces both repeated constructor plumbing and any proposed
argument-binding DSL.

`from_config()` is ordinary project code. It may:

- promote useful configuration values such as tag and seed;
- resolve a dataset or checkpoint reference;
- compute and verify digests;
- construct nested Pydantic provenance models;
- validate an environment or implementation contract.

The framework does not need to infer fields from names, function signatures,
paths, environment variables, or provider-specific conventions.

### 3.3 Provenance is fixed at the declared start boundary

The run declaration describes what is about to execute. It is validated and
snapshotted at `start()` and is immutable for that run.

There is no mutable mid-run provenance API in the first version. A fact required
to define the run must be resolved by `from_config()` before the boundary. A
fact produced by execution is an outcome and belongs in the terminal summary,
logger output, or project artefact.

This keeps the persistent lifecycle at two events and avoids introducing an
ordering-sensitive annotation stream.

### 3.4 The caller owns the boundary

A run is not intrinsically a process, PID, seed, W&B run, or scheduler job. It
is the interval selected by the caller through a context manager, decorator, or
explicit lifecycle calls.

A context manager is not process ownership. It is structured exception handling
around a caller-selected lexical scope, and it prevents forgotten finishes and
false success records.

### 3.5 Record facts, not guesses

A valid start without a finish materialises as `open`. It does not mean killed,
crashed, stale, or currently running. Host and PID may be recorded as nullable
observations but never determine status.

Git state is also an observation, not a reproducibility guarantee. Projects
needing a stronger code or environment contract declare and validate it through
`from_config()`.

### 3.6 Local record, not experiment platform

The first version has no server, remote backend, dashboard, scheduler, sweep
engine, background worker, model registry, metric backend, artefact upload, or
provider integration.

## 4. Generic base contract

The package currently supports Python 3.10, so the implementation uses
`TypeVar` and `Generic`; the PEP 695 spelling `class RunBase[C: ConfigBase]`
would require Python 3.12.

Conceptually:

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

C = TypeVar("C", bound=ConfigBase)


class RunBase(BaseModel, Generic[C], ABC):
    config: C

    @classmethod
    @abstractmethod
    def from_config(cls, config: C, /) -> Self:
        """Build and validate the complete run declaration."""
        ...
```

`@abstractmethod` is the innermost decorator, as required for abstract
classmethods.

The actual base also provides private lifecycle state and the context-decorator
protocol, but those are not additional public domain objects.

### 4.1 Why `config` is a model field

The typed configuration is part of the run declaration rather than a loosely
typed reserved constructor argument. This gives:

- static association between a run class and its accepted configuration;
- ordinary Pydantic validation and serialisation;
- a single input to `from_config()`;
- a natural recreation path for repeated decorated calls.

The persistent event stores the configuration once under its own `config`
namespace; it is not duplicated under the promoted declaration fields.

### 4.2 Construction and immutability

`from_config()` is the documented constructor. It returns a fully validated,
frozen declaration. Direct BaseModel construction may remain technically
possible, but lifecycle examples and project integrations use `from_config()`.

Freezing a Pydantic model is shallow, while `ConfigBase` instances are normally
mutable. During run-model validation, `RunBase` therefore takes a deep model copy
of `config`, including its private parser-provenance metadata. Mutating the
caller's original config after `from_config()` cannot change the run declaration.

Public model fields cannot be assigned after construction. Only private
lifecycle state changes. This prevents in-memory edits from silently diverging
from the persisted start event.

## 5. Configuration provenance

`ConfigBase.parse_args()` and `load_from_yaml()` remain source-compatible and
continue to return the configuration instance itself.

During resolution, `ConfigBase` attaches private metadata to the root instance:

- source kind: arguments, YAML, or programmatic construction;
- configuration-file path, when present;
- SHA-256 of the configuration file bytes;
- normalised CLI overrides;
- fully qualified configuration class.

`RunBase.start()` records this metadata together with the final Pydantic JSON
representation of `config`.

No public `RunConfig` or `ConfigResolution[T]` wrapper is introduced. If a
second independent feature later needs structured parser provenance, the
private metadata may be promoted to a read-only property without changing
`parse_args()`'s return type.

Pydantic's existing serialisation rules remain authoritative. `SecretStr` and
fields excluded from serialisation are not re-exposed by the register.

## 6. Declared provenance

The concrete run model is the primary provenance surface. Fields may be nested
Pydantic models, constrained types, discriminated unions, or values built by
project-owned factory methods.

```python
class CheckpointSource(BaseModel):
    path: Path
    sha256: str
    parent_run_id: str | None = None


class EnvironmentContract(BaseModel):
    name: str
    version: str
    digest: str


class FlowfieldRun(RunBase[RunConfig]):
    _path: ClassVar[Path] = Path("results/registry/runs.jsonl")

    tag: str
    seed: int
    output: Path
    environment: EnvironmentContract
    checkpoint: CheckpointSource | None = None

    @classmethod
    def from_config(cls, config: RunConfig, /) -> Self:
        return cls(
            config=config,
            tag=config.tag,
            seed=config.seed,
            output=Path(config.out),
            environment=resolve_environment_contract(config),
            checkpoint=resolve_checkpoint_source(config.resume),
        )
```

This is native, explicit provenance checking. The project owns the meaning and
the validation; `RunBase` owns validation-at-boundary and durable recording.
There is no generic checker registry.

## 7. Lifecycle modes

### 7.1 Context manager

```python
with FlowfieldRun.from_config(cfg) as run:
    _run(cfg)  # validation, training, evaluation, and saving
```

`__enter__()` starts the run and returns the same instance. `__exit__()` maps:

- normal exit to `completed`;
- an ordinary `Exception` to `failed`;
- `KeyboardInterrupt`, `SystemExit`, and other non-`Exception`
  `BaseException` values to `aborted`.

It always returns `False`, so application exceptions propagate.

A register-persistence error during exception unwinding must never replace the
application exception, including in strict mode.

### 7.2 Decorator

A `RunBase` instance implements `ContextDecorator` semantics:

```python
@FlowfieldRun.from_config(cfg)
def execute() -> Result:
    result = train(cfg)
    evaluate_and_save(result, cfg)
    return result
```

Each decorated invocation receives a fresh run UUID and fresh private lifecycle
state, while reusing a deep copy of the same immutable declaration. This follows
normal Python decorator semantics: the declaration is fixed when the decorator
expression is evaluated. `from_config()` is not called a second time behind the
user's back.

No function-argument inference or binding language is required. When provenance
must be resolved immediately before each execution, construct the run in a
context manager at that point rather than retaining a long-lived decorator
template. Function metadata is preserved with `functools.wraps`.

Async context managers and decorators are out of scope for the first version.

### 7.3 Explicit mode

```python
run = FlowfieldRun.from_config(cfg)
run.start()

try:
    result = _run(cfg)
except Exception as exc:
    run.finish(status="failed", error=exc)
    raise
except BaseException as exc:
    run.finish(status="aborted", error=exc)
    raise
else:
    run.finish(
        status="completed",
        summary={"last_iteration": result.last_iteration},
    )
```

Working signatures:

```python
def start(self, *, strict: bool | None = None) -> Self: ...


def finish(
    self,
    *,
    status: Literal["completed", "failed", "aborted"] = "completed",
    summary: Mapping[str, JsonValue] | None = None,
    error: BaseException | None = None,
    strict: bool | None = None,
) -> bool: ...
```

A lifecycle instance is single-use. Starting or finishing it twice raises before
another event is written. Calling `finish()` inside a context block is allowed
for a custom summary; `__exit__()` then performs no second write.

A non-strict persistence failure warns and leaves `run.recorded == False`; it
does not make the lifecycle object disappear or change the return type to
`Self | None`.

## 8. Automatic observations

At start, `RunBase` makes a small best-effort snapshot:

- UTC timestamp;
- entry point and working directory;
- hostname and PID as uninterpreted facts;
- Git commit, branch, dirty flag, and tracked-diff digest when available.

Every field is nullable. Absence means unavailable, not clean or default.

The automatic baseline is intentionally small. Dataset identity, checkpoint
identity, environment contracts, implementation variants, scheduler IDs, and
external service IDs belong in `from_config()` when they matter to the project.

## 9. Event format

Persistence is append-only JSON Lines with two event kinds.

### 9.1 Start

```json
{
  "schema_version": 1,
  "event": "start",
  "event_id": "c734c520-5004-45a8-a786-d79a1b28522a",
  "run_id": "63caa4cc-32b3-4ba2-9941-f1efed672868",
  "recorded_at": "2026-08-26T04:50:00.000000Z",
  "run_type": "my_project.FlowfieldRun",
  "declaration": {
    "tag": "arm_a",
    "seed": 0,
    "environment": {"name": "warehouse", "version": "3", "digest": "..."}
  },
  "config": {
    "type": "my_project.RunConfig",
    "source": "arguments",
    "values": {},
    "file": {"path": "configs/arm_a.yaml", "sha256": "..."},
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

The namespaces are deliberately separate:

- `declaration`: project-owned, typed facts produced by `from_config()`;
- `config`: effective configuration and parser provenance;
- `observed`: nullable framework observations.

### 9.2 Finish

```json
{
  "schema_version": 1,
  "event": "finish",
  "event_id": "f477fc1c-a5ef-42ba-b295-98ca67e7fc1c",
  "run_id": "63caa4cc-32b3-4ba2-9941-f1efed672868",
  "recorded_at": "2026-08-26T06:13:12.000000Z",
  "status": "completed",
  "summary": {"last_iteration": 39999}
}
```

A run with a valid start and no finish materialises as `open`.

## 10. Persistence and reconciliation

The first version supports one JSONL path. There is no public store or sink
protocol.

Each complete encoded event is appended through one `O_APPEND` operating-system
write, then flushed and `fsync`ed. Parent directories are created as needed.
Same-host concurrent writers must produce complete parseable lines and are
covered by a multiprocessing test. Shared network-filesystem atomicity is not
claimed.

Although JSONL has a physical order, materialisation has set semantics. For an
event collection `E`, any permutation `pi`, and duplicate subset `D`:

```text
materialise(E) == materialise(pi(E) + D)
```

Rules:

- deduplicate identical events by `event_id`;
- report an integrity conflict when one `event_id` has different payloads;
- group by `run_id` independently of line order;
- require one non-conflicting start;
- permit at most one non-conflicting finish;
- materialise no finish as `open`;
- surface conflicting starts or finishes rather than choosing a last line;
- diagnose malformed lines with line numbers, warning by default and raising in
  strict mode.

Host and PID never participate in materialisation.

## 11. Reading and diffing

The minimum read surface remains on the run declaration:

```python
runs = FlowfieldRun.records()
record = FlowfieldRun.find("63caa4cc")
diff = FlowfieldRun.diff(run_a, run_b)
```

A separate `RunRegistry` object is unnecessary because the declaration already
owns the path and run type.

Diffing compares:

- promoted declaration fields, including nested provenance;
- complete stored configurations;
- automatic observations only when explicitly requested.

It never consults current YAML files or current model defaults. Missing keys are
represented as typed add/remove operations rather than a magic sentinel string.

A small CLI may call the same methods for list, show, and diff. It is a view over
the local file, not a service.

## 12. Relationship to `LoggerBase`

`LoggerBase` records repeated measurements and reductions. `RunBase` records one
immutable start and at most one terminal event.

They remain separate because:

- metrics may be frequent or lossy without changing run identity;
- run events require UUID identity, idempotent materialisation, and conflict
  detection;
- logger sinks may coerce values, whereas the register has one canonical
  representation.

Inside a context, `run.run_id` may be logged as an ordinary logger field when a
join is needed. No process-wide active-run global is required.

## 13. Explicit non-goals

The first version will not:

- execute or wrap commands;
- install signal or global exception hooks;
- monitor liveness or infer killed/crashed status;
- resume or retry work;
- alter the immutable run declaration after construction;
- log time-series metrics;
- upload or manage artefacts;
- provide remote backends or synchronisation;
- initialise or mirror W&B/MLflow runs;
- generate sweeps or compose configurations;
- add a generic collector/checker plugin protocol;
- add a server, UI, daemon, or background thread;
- guarantee reproducibility from the record alone.

These exclusions keep `expedantic` a configuration and local-record library,
not a workflow framework or experiment platform.

## 14. `curl-field` migration

```python
class FlowfieldRun(RunBase[RunConfig]):
    _path: ClassVar[Path] = Path("results/registry/runs.jsonl")

    tag: str
    seed: int
    output: Path
    environment: EnvironmentContract
    parent_checkpoint: CheckpointSource | None = None

    @classmethod
    def from_config(cls, config: RunConfig, /) -> Self:
        return cls(
            config=config,
            tag=config.tag,
            seed=config.seed,
            output=Path(config.out),
            environment=resolve_environment_contract(config),
            parent_checkpoint=resolve_checkpoint_source(config.resume),
        )


def main() -> None:
    cfg = RunConfig.parse_args(
        require_default_file=True,
        diff_print_mode="none",
    )

    with FlowfieldRun.from_config(cfg):
        _run(cfg)  # validation, training, evaluation, and saving
```

This moves completion after the complete run and replaces tag/YAML forensics
with typed project provenance.

The project-local W&B importer remains project-local. Historical rows may be
translated into the two generic events, preserving legacy identifiers as
fields and surfacing old ID collisions.

## 15. Acceptance criteria

1. Existing `ConfigBase.parse_args()` calls remain source-compatible.
2. `RunBase[C]` statically associates a run declaration with one `ConfigBase`
   subtype.
3. Every concrete run implements `from_config(config)`.
4. `from_config()` can construct and validate nested project provenance.
5. The same run shape supports context-manager, instance-decorator, and explicit
   lifecycle modes without another public handle.
6. Repeated decorated calls clone the same immutable declaration, reset private
   lifecycle state, and receive distinct UUIDs.
7. Normal exit, failure, and interruption produce the specified terminal status
   without suppressing application exceptions.
8. The start event separates declaration, configuration, and automatic
   observations.
9. Two runs with the same tag and start second still have distinct IDs.
10. Materialisation is invariant to event order and duplicate identical events.
11. Conflicting events are surfaced rather than resolved by line order.
12. Concurrent local writers cannot interleave event lines.
13. A missing finish remains `open` without PID-based inference.
14. No tracking service or heavy dependency is introduced.
15. No public store, collector, checker, or event hierarchy is added.

## 16. Growth rule

Add a public primitive only when all are true:

1. at least two independent in-tree uses need the distinction;
2. `from_config()`, nested Pydantic values, or an existing lifecycle method
   cannot express it safely and economically;
3. the abstraction removes more concepts from callers than it adds to the
   library.
