# Solo Continue

One agent opens one native session for the lifetime of the controller process. The first logical
round receives the full task; subsequent rounds receive `continue`. Humanize's resumable state
retains the logical round across controller restarts, but a restarted process necessarily opens a
new native session.
