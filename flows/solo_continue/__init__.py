"""One agent reuses one native session and keeps advancing it with ``continue``."""

from __future__ import annotations

import time
from typing import Annotated, Any

from hmz.flows import Agent, AgentDefaults, Allowance, flow


@flow(budget=Allowance(tokens=10.0), resumable=True)
def run(
    agents: tuple[Annotated[Agent, AgentDefaults(goals=False)]],
    task: str,
    state: dict[str, Any] | None = None,
) -> None:
    """Keep one conversation alive for every logical round in this process."""

    (agent,) = agents
    kept = state if state is not None else {}
    session = agent.new()
    prompt = task
    try:
        while True:
            kept["rounds"] = int(kept.get("rounds", 0)) + 1
            answered = session(prompt, suppress=True)
            if answered:
                prompt = "continue"
            time.sleep(5)
    finally:
        session.close()
