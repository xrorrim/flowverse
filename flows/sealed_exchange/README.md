# Sealed Exchange

Two agents start from separate copies of the same source tree and work concurrently. Packages
remain private until both sides submit. The first side to submit keeps working privately while it
waits; neither side can inspect the other's lane.

A package is triggered by a prompt-defined milestone or the configured timer. The defaults are a
240-minute soft reminder and a 60-minute grace period. A normal package keeps the same native
session across the exchange. When grace expires, the controller closes that session, submits a
clearly marked workspace snapshot, and starts a new session for the next round.

```yaml
soft_minutes: 240
grace_minutes: 60
max_rounds: 6
budget: 0.2
rest_seconds: 2
```
