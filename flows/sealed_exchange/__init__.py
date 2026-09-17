"""Continuous Sealed Exchange -- private work never waits at the package barrier.

The agents start from separate copies of the same workspace and keep one conversation alive.
A submitted package is sealed immediately but stays invisible until the peer submits. The
earlier agent keeps working privately in the meantime. Once both packages exist, the flow
delivers them and injects the peer package into any live continuation. That injection itself is
the next logical turn boundary; the live model call does not need to return first.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

from hmz.flows import Agent, Session, flow, home
from pydantic import BaseModel, Field

MILLION = 1_000_000.0
CONTINUE = {"continue", "continue.", "resume", "resume.", "继续", "继续。"}
TURN_RETRIES = 1
TIME_BOUNDARY_MESSAGE = "建议现在将阶段性成果交换"
_DEADLINE_LOCK = threading.Lock()


class Agents(NamedTuple):
    """The two independent workers."""

    agent_1: Agent
    agent_2: Agent


class Config(BaseModel):
    """Cadence and stopping conditions."""

    model_config = {"extra": "forbid"}

    soft_minutes: int = Field(
        default=240,
        ge=10,
        description="Minutes before the live lane is asked to package its current stage; "
        "the reminder does not kill work already in flight.",
    )
    grace_minutes: int = Field(
        default=60,
        ge=1,
        description="Minutes allowed for the live lane to finish after the exchange reminder. "
        "When this expires, the controller closes it and exchanges a system snapshot.",
    )
    max_rounds: int = Field(
        default=6,
        ge=1,
        description="Maximum number of two-package exchanges in one run.",
    )
    budget: float = Field(
        default=0.2,
        ge=0,
        description="Millions of output tokens shared by both agents, or 0 for no limit.",
    )
    rest_seconds: float = Field(
        default=2.0,
        ge=0,
        description="Pause after an exchange before the next pair of turns.",
    )


class Submission(BaseModel):
    """One milestone result, which becomes a peer package only when submitted."""

    model_config = {"extra": "forbid"}

    submitted: bool = Field(
        description="True when the current milestone is complete or the timed exchange "
        "reminder asks for the current stage to be packaged."
    )
    summary: str = Field(description="What this milestone accomplished.")
    files: list[str] = Field(
        description="Relative paths to include in the peer package."
    )
    tests: list[str] = Field(
        description="Checks run for this milestone and their results."
    )
    next_step: str = Field(description="Suggested work after the exchange.")
    done: bool = Field(
        description="True only when the overall task, not just this milestone, is complete."
    )


class LaneOutcome(NamedTuple):
    """A normal agent package or a controller-created workspace snapshot."""

    package: Submission
    system_snapshot: bool = False
    session_closed: bool = False


class _LaneStopped(Exception):
    """Internal signal used to unwind speculative private work after global stop."""


def _snapshot(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    subprocess.run(
        ["cp", "--archive", "--reflink=auto", f"{source}/.", str(destination)],
        check=True,
    )


def _new_run(
    source: Path, objective: str, kept: dict[str, Any]
) -> tuple[Path, Path, Path]:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    run_root = home() / "sealed_exchange" / run_id
    workspace_1 = run_root / "agent-1"
    workspace_2 = run_root / "agent-2"
    (run_root / "controller" / "submissions").mkdir(parents=True)
    _snapshot(source, workspace_1)
    _snapshot(source, workspace_2)
    kept.clear()
    kept.update(
        version=4,
        source=str(source),
        objective=objective,
        run_root=str(run_root),
        round=0,
        output=0.0,
        exchanges=[],
    )
    return run_root, workspace_1, workspace_2


def _open_run(
    source: Path, task: str, kept: dict[str, Any]
) -> tuple[str, Path, Path, Path]:
    marker = task.strip().casefold() in CONTINUE
    can_resume = (
        kept.get("version") in {1, 2, 3, 4}
        and kept.get("source") == str(source)
        and (marker or kept.get("objective") == task.strip())
    )
    if can_resume:
        objective = str(kept["objective"])
        run_root = Path(str(kept["run_root"]))
        # Later versions change controller liveness only. Existing lanes and exchange
        # history remain valid and are upgraded in place on their next restart.
        kept["version"] = 4
        return objective, run_root, run_root / "agent-1", run_root / "agent-2"
    if marker:
        raise ValueError(
            "continue requires an unfinished sealed_exchange run in this workspace"
        )
    objective = task.strip()
    if not objective:
        raise ValueError("task must not be empty")
    run_root, workspace_1, workspace_2 = _new_run(source, objective, kept)
    return objective, run_root, workspace_1, workspace_2


def _prompt(
    *,
    objective: str,
    label: str,
    peer: str,
    round_number: int,
    soft_minutes: int,
) -> str:
    previous = round_number - 1
    exchange = (
        "This is the first round; no peer submission is available yet."
        if previous == 0
        else (
            "The preceding peer package is now available at "
            f"`.sealed_exchange/inbox/round-{previous}/from-{peer}/`. Read it before "
            "continuing, compare it with your own work, and use whatever is helpful."
        )
    )
    return f"""You are {label}, one of two independent agents working on the same task in
separate workspace copies. Keep all work inside your current workspace and do not inspect the
other agent's workspace or the flow controller's directories.

Task:
{objective}

This is round {round_number}. {exchange}

The unit of work that triggers an exchange is called a **Sealed Exchange milestone**
(**Sealed Exchange 阶段性里程碑**). The Task section may define exactly what counts as each
milestone; when it does, follow that definition. Otherwise, choose one concrete milestone that
advances the task. Work until the current milestone is actually complete.

The exchange trigger is whichever comes first: completing one concrete milestone or receiving
the message `{TIME_BOUNDARY_MESSAGE}` after about {soft_minutes} minutes. When that timed message
arrives, finish only the operation already in flight, package the useful stage reached so far,
and return `submitted=true`; the timed stage does not need to complete the originally planned
milestone.

If the current Sealed Exchange milestone is not complete when this turn ends, return
`submitted=false`; the flow will immediately send you back into the same session to continue,
without exposing peer work. Do not wait for the other agent inside your turn. The controller
seals the package immediately. If the peer has not submitted yet, you will continue useful
private work in this same session rather than becoming idle; none of the peer's work is exposed
during that time.

In `files`, list relative paths of files from your workspace that you want copied into the peer's
next inbox. An empty list is fine. `done` means you believe the overall task is complete, not merely
that the current Sealed Exchange milestone is complete, and should only be true together with
`submitted=true`.
"""


def _continue_prompt(soft_minutes: int) -> str:
    return f"""Your lane has not submitted yet, so no peer work has been revealed. Continue the
same Sealed Exchange milestone (“Sealed Exchange 阶段性里程碑”) in this session and workspace.
Work through the next useful operation or test. Return `submitted=true` when the milestone is
complete. If you receive `{TIME_BOUNDARY_MESSAGE}` after about {soft_minutes} minutes, finish only
the operation in flight and return `submitted=true` with the useful stage reached so far.
Otherwise return `submitted=false` again and the flow will keep you working privately.
"""


def _timed_exchange_prompt() -> str:
    return f"""{TIME_BOUNDARY_MESSAGE}。Finish only the operation already in flight, preserve the
useful stage reached so far, and return it now as a formal Submission with `submitted=true`.
The configured time boundary itself is a valid exchange trigger even if the originally planned
milestone is not complete."""


def _system_submission(label: str, round_number: int) -> Submission:
    """Describe a controller snapshot without claiming agent-supplied validation."""

    return Submission(
        submitted=True,
        summary=(
            f"{label} did not return a Submission within the exchange grace period. "
            f"The controller froze its current round-{round_number} workspace instead."
        ),
        files=[],
        tests=[
            "No agent-supplied validation was available before the system snapshot."
        ],
        next_step=(
            "Inspect files/workspace/, reproduce any candidate with the authoritative "
            "validator, and adopt only verified results."
        ),
        done=False,
    )


def _submission_markdown(label: str, round_number: int, package: Submission) -> str:
    files = "\n".join(f"- `{item}`" for item in package.files) or "- None"
    tests = "\n".join(f"- {item}" for item in package.tests) or "- None"
    return f"""# {label} submission, round {round_number}

## Summary

{package.summary}

## Shared files

{files}

## Tests

{tests}

## Suggested next step

{package.next_step}

## Overall task complete

{str(package.done).lower()}
"""


def _seal(
    *,
    run_root: Path,
    workspace: Path,
    label: str,
    round_number: int,
    package: Submission,
    system_snapshot: bool = False,
) -> Path:
    outbox = run_root / "controller" / "submissions" / f"round-{round_number}" / label
    if outbox.exists():
        shutil.rmtree(outbox)
    files_root = outbox / "files"
    files_root.mkdir(parents=True)
    (outbox / "submission.md").write_text(
        _submission_markdown(label, round_number, package), encoding="utf-8"
    )
    if system_snapshot:
        shutil.copytree(
            workspace,
            files_root / "workspace",
            ignore=shutil.ignore_patterns(".git", ".sealed_exchange", "__pycache__"),
        )
        (outbox / "SYSTEM-SNAPSHOT.md").write_text(
            "# Controller system snapshot\n\n"
            "The lane did not return a structured Submission within the configured grace "
            "period. `files/workspace/` is an unverified controller snapshot. Independently "
            "validate it before adopting any result.\n",
            encoding="utf-8",
        )
        return outbox
    for item in package.files:
        relative = Path(item)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        source = workspace / relative
        if source.is_file():
            destination = files_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return outbox


def _package_manifest(outbox: Path) -> list[dict[str, Any]]:
    """Fingerprint a sealed package without touching its bytes."""
    records = []
    if outbox.is_symlink() or not outbox.is_dir():
        raise ValueError("pending package must be a real directory")
    for path in sorted(outbox.rglob("*")):
        if path.is_symlink():
            raise ValueError("pending package contains a symlink")
        if path.is_file():
            raw = path.read_bytes()
            records.append(
                {
                    "path": str(path.relative_to(outbox)),
                    "size": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return records


def _legacy_submission(outbox: Path, label: str, round_number: int) -> Submission:
    """Read only the exact canonical legacy Markdown format; never guess fields."""
    text = (outbox / "submission.md").read_text(encoding="utf-8")
    prefix = f"# {label} submission, round {round_number}\n\n## Summary\n\n"
    if not text.startswith(prefix):
        raise ValueError("pending package label/round mismatch")
    remaining = text[len(prefix) :]
    values = []
    for heading in (
        "Shared files",
        "Tests",
        "Suggested next step",
        "Overall task complete",
    ):
        marker = f"\n\n## {heading}\n\n"
        if remaining.count(marker) != 1:
            raise ValueError("ambiguous pending package Markdown")
        value, remaining = remaining.split(marker, 1)
        values.append(value)
    if remaining not in ("true\n", "false\n"):
        raise ValueError("invalid pending package done value")
    summary, files_text, tests_text, next_step = values
    files = []
    if files_text != "- None":
        for line in files_text.splitlines():
            if not line.startswith("- `") or not line.endswith("`"):
                raise ValueError("invalid pending package file list")
            files.append(line[3:-1])
    tests = []
    if tests_text != "- None":
        for line in tests_text.splitlines():
            if not line.startswith("- "):
                raise ValueError("unsupported multiline pending test entry")
            tests.append(line[2:])
    package = Submission(
        submitted=True,
        summary=summary,
        files=files,
        tests=tests,
        next_step=next_step,
        done=remaining == "true\n",
    )
    if _submission_markdown(label, round_number, package) != text:
        raise ValueError("pending package did not round-trip canonically")
    return package


def _build_resume_manifest(run_root: Path, round_number: int) -> dict[str, Any]:
    """Prepare an explicit recovery descriptor; caller decides when to write it."""
    directory = run_root / "controller" / "submissions" / f"round-{round_number}"
    packages = {}
    for outbox in sorted(directory.iterdir()):
        label = outbox.name
        if label not in ("agent-1", "agent-2"):
            raise ValueError("unexpected pending package entry")
        package = _legacy_submission(outbox, label, round_number)
        packages[label] = {
            "submission": package.model_dump(mode="json"),
            "files": _package_manifest(outbox),
            "system_snapshot": (outbox / "SYSTEM-SNAPSHOT.md").is_file(),
        }
    if not packages:
        raise ValueError("no pending package to preserve")
    return {
        "version": 1,
        "run_root": str(run_root.resolve()),
        "round": round_number,
        "packages": packages,
    }


def _deliver(
    outbox: Path, peer_workspace: Path, round_number: int, sender: str
) -> Path:
    inbox = (
        peer_workspace
        / ".sealed_exchange"
        / "inbox"
        / f"round-{round_number}"
        / f"from-{sender}"
    )
    if inbox.exists():
        shutil.rmtree(inbox)
    inbox.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(outbox, inbox)
    return inbox


def _take_turn(session: Session, prompt: str) -> Submission:
    package = session(prompt, suppress=False, schema=Submission)
    if package is None:
        raise RuntimeError("agent returned no work result")
    return package


def _recovery_prompt(previous_prompt: str, soft_minutes: int) -> str:
    return f"""{previous_prompt}

Your previous turn ended before Humanize received a usable Submission.
Continue working in the same private lane on the same Sealed Exchange milestone
(Sealed Exchange 阶段性里程碑). Do not inspect the peer lane. Submit when the milestone is complete
or when you receive `{TIME_BOUNDARY_MESSAGE}` at the {soft_minutes}-minute boundary."""


def _work_until_submission(
    session: Session,
    prompt: str,
    soft_minutes: float,
    grace_minutes: float = 60,
    *,
    label: str = "agent",
    round_number: int = 0,
    remind_at: float | None = None,
    snapshot_at: float | None = None,
    activate_after: threading.Event | None = None,
    activation_text: str | None = None,
    deadline_provider: Callable[[], dict[str, float]] | None = None,
    stop: threading.Event | None = None,
) -> LaneOutcome:
    if stop is not None and stop.is_set():
        raise _LaneStopped
    # A resumed, already-expired epoch must snapshot before starting an RPC.
    # Otherwise the reminder can close a session before its first call is active.
    if (
        activate_after is None
        and snapshot_at is not None
        and snapshot_at <= time.time()
    ):
        print(
            f"sealed_exchange: {label} resumed expired round {round_number}; "
            "creating system snapshot before model call",
            flush=True,
        )
        return LaneOutcome(
            _system_submission(label, round_number), system_snapshot=True
        )
    current = prompt
    retries = 0
    finished = threading.Event()
    time_boundary = threading.Event()
    forced_snapshot = threading.Event()
    cancelled = threading.Event()
    call_active = threading.Event()
    round_started = threading.Event()
    if activate_after is None:
        round_started.set()
    elif activate_after.is_set():
        current = activation_text or prompt
        round_started.set()

    def remind() -> None:
        if activate_after is not None and not activate_after.is_set():
            while not finished.is_set() and not activate_after.wait(0.1):
                pass
            if finished.is_set():
                return
            if stop is not None and stop.is_set():
                cancelled.set()
                _close(session)
                return
            if not round_started.is_set() and call_active.is_set() and activation_text:
                try:
                    _interject(session, activation_text)
                    round_started.set()
                    print(
                        f"sealed_exchange: round {round_number} started inside live session",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 -- next call can carry it
                    print(
                        "sealed_exchange: live round-start injection was not consumed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
        if deadline_provider is not None:
            deadlines = deadline_provider()
            remind_deadline = float(deadlines["remind_at"])
            snapshot_deadline = float(deadlines["snapshot_at"])
        else:
            now = time.time()
            remind_deadline = (
                remind_at if remind_at is not None else now + soft_minutes * 60
            )
            snapshot_deadline = (
                snapshot_at
                if snapshot_at is not None
                else remind_deadline + grace_minutes * 60
            )
        if finished.wait(max(0.0, remind_deadline - time.time())):
            return
        time_boundary.set()
        try:
            _interject(session, TIME_BOUNDARY_MESSAGE)
            print(
                f"sealed_exchange: {soft_minutes:g}m exchange reminder delivered",
                flush=True,
            )
        except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
            print(
                "sealed_exchange: timed exchange reminder could not be delivered: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        if finished.wait(max(0.0, snapshot_deadline - time.time())):
            return
        forced_snapshot.set()
        print(
            f"sealed_exchange: {label} grace expired; closing live session for "
            f"round {round_number} system snapshot",
            flush=True,
        )
        _close(session)

    reminder = threading.Thread(
        target=remind,
        name="sealed-exchange-time-boundary",
        daemon=True,
    )
    reminder.start()

    try:
        while True:
            if stop is not None and stop.is_set():
                raise _LaneStopped
            try:
                call_active.set()
                try:
                    package = _take_turn(session, current)
                finally:
                    call_active.clear()
            except Exception as exc:
                if cancelled.is_set() or (stop is not None and stop.is_set()):
                    raise _LaneStopped from exc
                if forced_snapshot.is_set():
                    return LaneOutcome(
                        _system_submission(label, round_number),
                        system_snapshot=True,
                        session_closed=True,
                    )
                if not isinstance(exc, (subprocess.CalledProcessError, ValueError)):
                    raise
                if retries >= TURN_RETRIES:
                    raise
                retries += 1
                print(
                    "sealed_exchange: agent turn failed; retrying once: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                current = _recovery_prompt(current, soft_minutes)
                continue

            retries = 0
            if cancelled.is_set() or (stop is not None and stop.is_set()):
                raise _LaneStopped
            if activate_after is not None and not round_started.is_set():
                if activate_after.is_set():
                    current = activation_text or prompt
                    round_started.set()
                else:
                    current = prompt
                continue
            if package.submitted:
                return LaneOutcome(package)
            if forced_snapshot.is_set():
                return LaneOutcome(
                    _system_submission(label, round_number),
                    system_snapshot=True,
                    session_closed=True,
                )
            current = (
                _timed_exchange_prompt()
                if time_boundary.is_set()
                else _continue_prompt(soft_minutes)
            )
    finally:
        finished.set()
        reminder.join(timeout=0.1)


def _deepseek_splice(session: Session, text: str) -> None:
    """Insert a prompt into a busy DSH session using the SDK inbox."""

    harness = getattr(session, "_harness", None)
    session_id = getattr(session, "named", None)
    client = getattr(harness, "client", None)
    prompt = getattr(client, "session_prompt", None)
    if not isinstance(session_id, str) or not session_id or not callable(prompt):
        raise RuntimeError("DeepSeek session is not running")
    prompt(session_id, [{"type": "text", "text": text}])


def _interject(session: Session, text: str) -> None:
    """Steer a live session, including Humanize's current DSH adapter."""

    try:
        session.interject(text)
    except NotImplementedError:
        _deepseek_splice(session, text)


def _waiting_prompt(label: str, peer: str, round_number: int) -> str:
    return f"""Your round-{round_number} Sealed Exchange package has been sealed. The peer has
not submitted yet, so no peer work is visible. Do not wait and do not inspect controller or peer
directories. Continue useful private work in this workspace: finish experiments, improve the
implementation, validate ideas, and leave durable notes. You may work ahead, but do not create or
return another formal Sealed Exchange package before the round-{round_number} exchange occurs.
If this call finishes before the exchange, return a structured Submission with
`submitted=false`; the controller will keep the private lane running.

If a user message announces that the exchange is ready, it immediately starts the next logical
turn. Read `.sealed_exchange/inbox/round-{round_number}/from-{peer}/`, incorporate useful evidence,
and continue directly into the next milestone without returning merely to acknowledge the
exchange. Otherwise, return after a useful chunk of private work; the controller will immediately
let {label} continue privately."""


def _exchange_notice(peer: str, round_number: int) -> str:
    next_round = round_number + 1
    return f"""SEALED EXCHANGE READY for round {round_number}. The peer submitted independently,
and its sealed package is now available at
`.sealed_exchange/inbox/round-{round_number}/from-{peer}/`.

This message is the start of logical turn and Sealed Exchange round {next_round}. Interrupt the
old private direction at a safe operation boundary, read the peer package now, and continue
directly with the next milestone in this same session and workspace. Do not return merely to
acknowledge this message. The round-{next_round} soft timer starts when this message is sent;
eventually return a formal Submission for round {next_round} when its milestone is complete or
when the timed exchange reminder arrives."""


class _RoundExchange:
    """Seal independently, then atomically reveal a matched pair."""

    def __init__(
        self,
        *,
        run_root: Path,
        round_number: int,
        workspaces: dict[str, Path],
        kept: dict[str, Any],
        soft_minutes: int,
        grace_minutes: int,
        max_rounds: int,
        stop: threading.Event | None = None,
        budget_reached: Callable[[], bool] | None = None,
    ) -> None:
        self.run_root = run_root
        self.round_number = round_number
        self.workspaces = workspaces
        self.kept = kept
        self.soft_minutes = soft_minutes
        self.grace_minutes = grace_minutes
        self.max_rounds = max_rounds
        self.stop = stop
        self.budget_reached = budget_reached
        self.packages: dict[str, Submission] = {}
        self.outboxes: dict[str, Path] = {}
        self.exchange_ready = threading.Event()
        self.finished = False
        self._lock = threading.Lock()

    def submit(self, label: str, outcome: LaneOutcome) -> None:
        """Freeze one package immediately and reveal only when its peer is frozen."""

        with self._lock:
            if label in self.packages:
                raise ValueError(f"round {self.round_number}: {label} already sealed")
        package = outcome.package
        outbox = _seal(
            run_root=self.run_root,
            workspace=self.workspaces[label],
            label=label,
            round_number=self.round_number,
            package=package,
            system_snapshot=outcome.system_snapshot,
        )
        with self._lock:
            self.packages[label] = package
            self.outboxes[label] = outbox
            print(f"round {self.round_number} · {label} package sealed", flush=True)
            self._exchange_if_ready()

    def _exchange_if_ready(self) -> None:
        """Commit a matched pair; caller holds the exchange lock."""
        if len(self.packages) == 2 and not self.exchange_ready.is_set():
            inbox_1 = _deliver(
                self.outboxes["agent-2"],
                self.workspaces["agent-1"],
                self.round_number,
                "agent-2",
            )
            inbox_2 = _deliver(
                self.outboxes["agent-1"],
                self.workspaces["agent-2"],
                self.round_number,
                "agent-1",
            )
            self.kept["round"] = self.round_number
            self.kept["exchanges"].append(
                {
                    "round": self.round_number,
                    "agent_1": self.packages["agent-1"].model_dump(mode="json"),
                    "agent_2": self.packages["agent-2"].model_dump(mode="json"),
                    "system_snapshots": {
                        "agent_1": (
                            self.outboxes["agent-1"] / "SYSTEM-SNAPSHOT.md"
                        ).is_file(),
                        "agent_2": (
                            self.outboxes["agent-2"] / "SYSTEM-SNAPSHOT.md"
                        ).is_file(),
                    },
                }
            )
            self.finished = all(package.done for package in self.packages.values())
            over_budget = bool(self.budget_reached and self.budget_reached())
            should_stop = (
                self.finished or over_budget or self.round_number >= self.max_rounds
            )
            if should_stop and self.stop is not None:
                self.stop.set()
            if not should_stop:
                _epoch_deadlines(
                    self.kept,
                    self.round_number + 1,
                    self.soft_minutes,
                    self.grace_minutes,
                )
            # Setting this event is the logical next-turn boundary. Any lane already
            # inside a private continuation receives the next-round prompt in-place.
            self.exchange_ready.set()
            print(f"round {self.round_number} · exchanged", flush=True)
            print(f"agent-1 inbox: {inbox_1}", flush=True)
            print(f"agent-2 inbox: {inbox_2}", flush=True)

    def restore_pending(self) -> None:
        """Restore only an explicit, byte-validated current-round descriptor."""
        path = self.run_root / "controller" / "resume-pending.json"
        if not path.is_file():
            return
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("run_root") != str(
            self.run_root.resolve()
        ):
            raise ValueError("pending recovery descriptor run mismatch")
        number = record.get("round")
        if not isinstance(number, int):
            raise ValueError("invalid pending recovery round")
        if number < self.round_number:
            return  # This explicit recovery already completed; never replay it.
        if number != self.round_number:
            raise ValueError("pending recovery descriptor round mismatch")
        entries = record.get("packages")
        if (
            not isinstance(entries, dict)
            or not entries
            or not set(entries) <= {"agent-1", "agent-2"}
        ):
            raise ValueError("invalid pending recovery packages")
        directory = self.run_root / "controller" / "submissions" / f"round-{number}"
        if {p.name for p in directory.iterdir()} != set(entries):
            raise ValueError(
                "pending package set changed; take a fresh recovery backup"
            )
        packages, outboxes = {}, {}
        for label, entry in entries.items():
            outbox = directory / label
            package = Submission.model_validate(entry["submission"])
            if not package.submitted:
                raise ValueError("pending package is not a submission")
            if _package_manifest(outbox) != entry["files"]:
                raise ValueError("pending package hash mismatch")
            if _submission_markdown(label, number, package) != (
                outbox / "submission.md"
            ).read_text(encoding="utf-8"):
                raise ValueError(
                    "pending Submission metadata differs from sealed bytes"
                )
            if (
                bool(entry["system_snapshot"])
                != (outbox / "SYSTEM-SNAPSHOT.md").is_file()
            ):
                raise ValueError("pending system-snapshot provenance mismatch")
            packages[label], outboxes[label] = package, outbox
        with self._lock:
            if self.packages:
                raise ValueError("cannot restore an already-live exchange")
            self.packages.update(packages)
            self.outboxes.update(outboxes)
            for label in packages:
                print(
                    f"round {number} · {label} preserved package restored", flush=True
                )
            self._exchange_if_ready()


def _run_lane(
    *,
    agent: Agent,
    label: str,
    peer: str,
    workspace: Path,
    objective: str,
    exchanges: dict[int, _RoundExchange],
    start_round: int,
    held: Config,
    kept: dict[str, Any],
    stop: threading.Event,
) -> LaneOutcome:
    """Own one session continuously while logical turns advance at exchange events."""

    session = agent.new(cwd=workspace)
    last = LaneOutcome(_system_submission(label, start_round))
    needs_initial_context = label in exchanges[start_round].packages
    try:
        for round_number in range(start_round, held.max_rounds + 1):
            exchange = exchanges[round_number]
            if label in exchange.packages:
                # The sealed bytes belong to this round. Do not package newer private
                # work over them; the next iteration uses its normal waiting prompt.
                last = LaneOutcome(
                    exchange.packages[label],
                    system_snapshot=(
                        exchange.outboxes[label] / "SYSTEM-SNAPSHOT.md"
                    ).is_file(),
                )
                if round_number >= held.max_rounds:
                    exchange.exchange_ready.wait()
                    return last
                continue

            def deadlines_for_lane(
                round_number: int = round_number,
            ) -> dict[str, float]:
                return _epoch_deadlines(
                    kept,
                    round_number,
                    held.soft_minutes,
                    held.grace_minutes,
                )[label]

            if round_number == start_round:
                deadlines = deadlines_for_lane()
                prompt = _prompt(
                    objective=objective,
                    label=label,
                    peer=peer,
                    round_number=round_number,
                    soft_minutes=held.soft_minutes,
                )
                activate_after = None
                activation_text = None
                remind_at = deadlines["remind_at"]
                snapshot_at = deadlines["snapshot_at"]
                deadline_provider = None
            else:
                previous = exchanges[round_number - 1]
                prompt = _waiting_prompt(label, peer, round_number - 1)
                activate_after = previous.exchange_ready
                activation_text = _exchange_notice(peer, round_number - 1)
                remind_at = None
                snapshot_at = None
                deadline_provider = deadlines_for_lane

                if round_number == start_round + 1 and needs_initial_context:
                    # Restarted sessions have no prior Task prompt. Keep the original
                    # objective even if the exchange is ready before their first call.
                    context = (
                        f"You are {label}, one of two independent agents working on the same task in\n"
                        "separate workspace copies. Keep all work inside your current workspace and do not inspect the\n"
                        "other agent's workspace or the flow controller's directories.\n\n"
                        f"Task:\n{objective}\n\n"
                    )
                    prompt = context + prompt
                    activation_text = context + activation_text

            last = _work_until_submission(
                session,
                prompt,
                held.soft_minutes,
                held.grace_minutes,
                label=label,
                round_number=round_number,
                remind_at=remind_at,
                snapshot_at=snapshot_at,
                activate_after=activate_after,
                activation_text=activation_text,
                deadline_provider=deadline_provider,
                stop=stop,
            )
            if (
                round_number == start_round
                and last.system_snapshot
                and not last.session_closed
            ):
                # The synchronous expired-start branch did not call this session.
                needs_initial_context = True
            exchange.submit(label, last)
            if last.session_closed:
                _close(session)
                session = agent.new(cwd=workspace)

            if round_number >= held.max_rounds:
                exchange.exchange_ready.wait()
                return last
            time.sleep(held.rest_seconds)
        return last
    except _LaneStopped:
        return last
    finally:
        _close(session)


def _close(session: Session) -> None:
    try:
        session.close()
    except Exception:
        pass


def _epoch_deadlines(
    kept: dict[str, Any],
    round_number: int,
    soft_minutes: int,
    grace_minutes: int,
) -> dict[str, dict[str, float]]:
    """Create or resume durable per-lane reminder and snapshot deadlines."""

    with _DEADLINE_LOCK:
        epochs = kept.setdefault("epoch_deadlines", {})
        key = str(round_number)
        raw = epochs.get(key)
        if isinstance(raw, dict) and all(
            isinstance(raw.get(label), dict) for label in ("agent-1", "agent-2")
        ):
            return raw
        remind_at = time.time() + soft_minutes * 60
        snapshot_at = remind_at + grace_minutes * 60
        deadlines = {
            label: {"remind_at": remind_at, "snapshot_at": snapshot_at}
            for label in ("agent-1", "agent-2")
        }
        epochs[key] = deadlines
        return deadlines


@flow(resumable=True)
def run(
    agents: Agents,
    task: str,
    config: Config | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Run private work in parallel and exchange only after both agents submit."""
    held = config or Config()
    kept = state if state is not None else {}
    source = Path.cwd().absolute()
    objective, run_root, workspace_1, workspace_2 = _open_run(source, task, kept)
    before = float(kept.get("output", 0.0))
    start_round = int(kept.get("round", 0)) + 1
    print(f"sealed_exchange · state {run_root}")
    if start_round > held.max_rounds:
        print(f"stopping: reached {held.max_rounds} rounds")
        print(f"results: {run_root}")
        kept.clear()
        return

    workspaces = {"agent-1": workspace_1, "agent-2": workspace_2}
    stop = threading.Event()
    budget_lock = threading.Lock()

    def budget_reached() -> bool:
        with budget_lock:
            spent = (
                before + agents.agent_1.spent().output + agents.agent_2.spent().output
            )
            kept["output"] = spent
            return bool(held.budget and spent >= held.budget * MILLION)

    exchanges = {
        round_number: _RoundExchange(
            run_root=run_root,
            round_number=round_number,
            workspaces=workspaces,
            kept=kept,
            soft_minutes=held.soft_minutes,
            grace_minutes=held.grace_minutes,
            max_rounds=held.max_rounds,
            stop=stop,
            budget_reached=budget_reached,
        )
        for round_number in range(start_round, held.max_rounds + 1)
    }
    exchanges[start_round].restore_pending()
    # Round one begins at controller start. Later rounds begin atomically when the
    # preceding matched packages are delivered and exchange_ready is set.
    _epoch_deadlines(kept, start_round, held.soft_minutes, held.grace_minutes)
    print(f"round {start_round} · both agents working privately")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(
            _run_lane,
            agent=agents.agent_1,
            label="agent-1",
            peer="agent-2",
            workspace=workspace_1,
            objective=objective,
            exchanges=exchanges,
            start_round=start_round,
            held=held,
            kept=kept,
            stop=stop,
        )
        future_2 = executor.submit(
            _run_lane,
            agent=agents.agent_2,
            label="agent-2",
            peer="agent-1",
            workspace=workspace_2,
            objective=objective,
            exchanges=exchanges,
            start_round=start_round,
            held=held,
            kept=kept,
            stop=stop,
        )
        future_1.result()
        future_2.result()

    spent = before + agents.agent_1.spent().output + agents.agent_2.spent().output
    completed_round = int(kept.get("round", start_round))
    completed = exchanges[completed_round]
    if completed.finished:
        reason = "both agents report the task complete"
    elif held.budget and spent >= held.budget * MILLION:
        reason = (
            f"{spent / MILLION:.2f}M output tokens reached the {held.budget:g}M budget"
        )
    else:
        reason = f"reached {held.max_rounds} rounds"
    print(f"stopping: {reason}")
    print(f"results: {run_root}")
    kept.clear()


__all__ = ["Agents", "Config", "Submission", "run"]
