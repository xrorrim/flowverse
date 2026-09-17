# Flame Chase

Two agents alternate on the same workspace. Every turn opens a fresh native session. After
`soft_hours` (six by default), the flow asks the live agent to finish only its current operation,
save verified results, write a handoff, and return. There is no hard cutoff in this flow.

```yaml
soft_hours: 6
```
