---
description: Orient the session to the project's current goal, phase, and immediate priorities
argument-hint: "[optional: a focus area to center this session on]"
---

You are orienting yourself (and the user) at the start of work on the
**agentic-crypto-trader** project. Produce a tight, current snapshot — not a recap of
everything. (CLAUDE.md is already in context; this command surfaces *what matters now*.)

## Steps

1. Re-check `CLAUDE.md` for the phased plan, four-surface stack, and key dates.
2. Read the vault MOC `.obsidian-vault/BNB Hackathon/Index.md` and
   `.obsidian-vault/BNB Hackathon/Project Overview.md` for current state and timeline.
3. Skim the `BNB Hackathon/` topic notes to gauge which are developed vs still stubs.

## Then report, briefly

- **North star** — the one outcome that matters right now (one sentence).
- **Current phase** — which phase of the plan we're in, and the gate ahead (the **June 16
  Track 1 PoC**, **June 22 registration**, and the Track 2 fallback).
- **Immediate priorities** — the 2–4 things that actually move the project this session.
- **Active blockers** — anything gating progress (autonomous self-custody signing; hosting &
  keys; on-chain data reach) — see [[Security and Encryption]] / [[Tech Stack]].
- **Suggested next action** — one concrete move, and which agent should own it.
- **Tip** — if the next action has a verifiable end state, suggest framing it as a built-in
  `/goal` (e.g. `/goal <condition>`) so the work auto-continues until done.

## If an argument was provided

`$ARGUMENTS`

If non-empty, treat it as the focus for this session: reconcile it with the phased plan, say
plainly whether it fits the current phase or jumps ahead, and frame priorities around it.
Surface any change to the plan rather than silently rewriting it.

Keep it scannable. The point is to start aligned, fast.
