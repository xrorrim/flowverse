"""Two agents alternate fresh sessions on one workspace with a soft handoff timer.

Each turn reads the shared repository rather than the previous agent's conversation. At the
configured boundary the live session is asked to finish only its current operation, write a
handoff, and return. The timer never kills the operation; returning closes that session and
starts the other agent in a fresh one.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Annotated, Any

from hmz.flows import Agent, AgentDefaults, Allowance, Session, flow
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Cadence for one agent's turn."""

    model_config = {"extra": "forbid"}

    soft_hours: float = Field(
        default=6.0,
        ge=0,
        description=(
            "Hours before the live agent is asked to finish its in-flight operation, "
            "write a handoff, and return. Zero disables the reminder."
        ),
    )


def _handoff_reminder(hours: float) -> str:
    return f"""Soft work-time reminder: you have been working for about {hours:g} hours in this
turn. Do not start another validation or experiment. Finish only the operation already in
flight, preserve the verified results, write a clear handoff for the next agent, and then return
to end this turn. This is a soft boundary: do not abandon or kill the current operation midway."""


def _remind(session: Session, hours: float) -> None:
    if not session.steers:
        raise RuntimeError(
            "the selected backend cannot receive a live handoff reminder"
        )
    session.interject(_handoff_reminder(hours))


def _run_turn(agent: Agent, task: str, soft_hours: float) -> None:
    """Run one fresh turn and inject at most one soft handoff reminder."""

    session = agent.new()
    finished = threading.Event()
    reminder: threading.Thread | None = None

    if soft_hours:

        def remind() -> None:
            if finished.wait(soft_hours * 60 * 60):
                return
            try:
                _remind(session, soft_hours)
                print(
                    f"flame_chase: soft {soft_hours:g}h handoff reminder delivered",
                    flush=True,
                )
            except (NotImplementedError, RuntimeError, OSError) as exc:
                print(
                    "flame_chase: soft handoff reminder could not be delivered: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        reminder = threading.Thread(
            target=remind,
            name="flame-chase-soft-handoff",
            daemon=True,
        )
        reminder.start()

    try:
        session(task, suppress=True)
    finally:
        finished.set()
        session.close()
        if reminder is not None:
            reminder.join(timeout=0.1)


@flow(budget=Allowance(tokens=33.550336), resumable=True)
def run(
    agents: tuple[
        Annotated[Agent, AgentDefaults(goals=False)],
        Annotated[Agent, AgentDefaults(goals=False)],
    ],
    task: str,
    config: Config | None = None,
    state: dict[str, Any] | None = None,
) -> None:
    """Alternate the two agents until the run allowance stops the flow."""

    held = config or Config()
    kept = state if state is not None else {}
    at = int(kept.get("turn", 0)) % len(agents)
    while True:
        _run_turn(agents[at], task, held.soft_hours)
        at = (at + 1) % len(agents)
        written: dict[str, Any] = {"turn": at}
        if at == 0:
            written["rounds"] = int(kept.get("rounds", 0)) + 1
        kept.update(written)
        time.sleep(5)
