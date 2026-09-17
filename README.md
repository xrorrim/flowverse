# xrorrim flowverse

This fork tracks Humanize's official flows and the experiment flows used by our long-running
agent ablations.

## Maintained flows

- `flame_chase`: two agents alternate fresh sessions on a shared workspace. A live turn gets
  one soft handoff reminder after six hours by default; it finishes its in-flight operation,
  writes a handoff, and returns without a forced cutoff.
- `sealed_exchange`: two isolated lanes work concurrently. A verified milestone or the
  four-hour timer can produce a package; after one additional hour the controller closes an
  unreturned session and exchanges a clearly marked workspace snapshot. Normal exchanges keep
  the native session; forced snapshots create a new one.
- `solo_continue`: one agent keeps one native session within the controller process. It receives
  the full task once and `continue` on later logical rounds.

The flow source contains the complete configuration schema. Runtime allowances remain Humanize
run settings and can be overridden when starting a flow.
