from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import flame_chase
import pytest
import sealed_exchange
import solo_continue


class TimedSession:
    shapes = True
    takes_tools = True
    steers = True

    def __init__(self, *, return_after_reminder: bool = True) -> None:
        self.return_after_reminder = return_after_reminder
        self.interjections: list[str] = []
        self.prompts: list[str] = []
        self.closed = threading.Event()

    def __call__(
        self,
        prompt: str,
        *,
        suppress: bool = False,
        schema: type[Any] | None = None,
    ) -> Any:
        self.prompts.append(prompt)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            if self.return_after_reminder and self.interjections:
                if schema is None:
                    return "done"
                return schema(
                    submitted=True,
                    summary="timed milestone",
                    files=[],
                    tests=["checked"],
                    next_step="continue",
                    done=False,
                )
            if self.closed.is_set():
                raise RuntimeError("session closed")
            time.sleep(0.001)
        raise TimeoutError("test session did not receive its boundary")

    def interject(self, text: str) -> None:
        self.interjections.append(text)

    def close(self) -> None:
        self.closed.set()


class TimedAgent:
    def __init__(self) -> None:
        self.sessions: list[TimedSession] = []

    def new(self, cwd: str | Path | None = None) -> TimedSession:
        session = TimedSession()
        self.sessions.append(session)
        return session


def test_flame_chase_defaults_to_six_hour_soft_handoffs() -> None:
    assert flame_chase.Config().soft_hours == 6


def test_flame_chase_reminds_without_killing_and_each_turn_is_fresh() -> None:
    agent = TimedAgent()

    flame_chase._run_turn(agent, "work", soft_hours=0.000001)
    flame_chase._run_turn(agent, "work", soft_hours=0.000001)

    assert len(agent.sessions) == 2
    for session in agent.sessions:
        assert session.closed.is_set()
        assert len(session.interjections) == 1
        reminder = " ".join(session.interjections[0].split())
        assert "Do not start another validation" in reminder
        assert "Finish only the operation already in flight" in reminder


def test_sealed_exchange_defaults_to_four_hours_plus_one_hour_grace() -> None:
    config = sealed_exchange.Config()
    assert config.soft_minutes == 240
    assert config.grace_minutes == 60


def test_sealed_soft_boundary_returns_a_normal_package_without_closing() -> None:
    session = TimedSession()

    outcome = sealed_exchange._work_until_submission(
        session,
        "work privately",
        soft_minutes=0.000001,
        grace_minutes=1,
        label="agent-1",
        round_number=1,
    )

    assert outcome.package.submitted
    assert not outcome.system_snapshot
    assert not outcome.session_closed
    assert not session.closed.is_set()
    assert session.interjections == [sealed_exchange.TIME_BOUNDARY_MESSAGE]


def test_sealed_grace_expiry_closes_and_returns_a_system_snapshot() -> None:
    session = TimedSession(return_after_reminder=False)

    outcome = sealed_exchange._work_until_submission(
        session,
        "work privately",
        soft_minutes=0.000001,
        grace_minutes=0.000001,
        label="agent-2",
        round_number=3,
    )

    assert outcome.package.submitted
    assert outcome.system_snapshot
    assert outcome.session_closed
    assert session.closed.is_set()


class StopSolo(Exception):
    pass


class SoloSession:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False

    def __call__(self, prompt: str, *, suppress: bool = False) -> str:
        if len(self.prompts) == 3:
            raise StopSolo
        self.prompts.append(prompt)
        return "worked"

    def close(self) -> None:
        self.closed = True


class SoloAgent:
    def __init__(self) -> None:
        self.sessions: list[SoloSession] = []

    def new(self, cwd: str | Path | None = None) -> SoloSession:
        session = SoloSession()
        self.sessions.append(session)
        return session


def test_solo_continue_reuses_one_session(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = SoloAgent()
    state: dict[str, Any] = {}
    monkeypatch.setattr(solo_continue.time, "sleep", lambda _seconds: None)

    with pytest.raises(StopSolo):
        solo_continue.run((agent,), "start here", state)

    (session,) = agent.sessions
    assert session.prompts == ["start here", "continue", "continue"]
    assert session.closed
    assert state == {"rounds": 4}
