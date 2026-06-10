"""The launch tier of the RL experiment loop — build a sweep, gate its smoke, kill by PID.

The PURE builders + parsers behind the 🟡 rl_train / rl_kill MCP tools. The runbook discipline
(vault "Remote Capabilities") is encoded structurally here so an autonomous loop can't re-learn
it the hard way:

  - `reward_config` is a FREEFORM knob dict validated against `REWARD_KEYS` — an unknown key is
    REFUSED, not silently ignored, so a typo'd shaping param fails loud instead of training the
    wrong thing.
  - the sweep is built **argv-direct** (one `train_event.py` per seed, sequenced — never parallel),
    superseding the hardcoded presets in scripts/run_eventrung_sweep.sh.
  - the smoke runs with publishing SUPPRESSED (empty `APENTIC_PUBLISH_TARGET`) so a gate check
    never pollutes the CDN/ledger.
  - the kill targets SPECIFIC PIDs and NEVER a process group — `kill -- -<PGID>` once took
    tailscaled down with the job (shared process group). The `[x]` bracket trick keeps the matcher
    from ever counting/killing itself.

Nothing here runs a command; the server tools feed these strings to `remote.run_ssh` / the SSH
executor. That keeps every builder/parser unit-testable offline.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

# Freeform reward-shaping knobs -> (train_event.py flag, kind). `bool` flags are store_true:
# emitted only when truthy. Anything NOT here is refused by `build_reward_args` (typo guard).
REWARD_KEYS: dict[str, tuple[str, type]] = {
    "reward_mode": ("--reward-mode", str),     # absolute | relative | residual
    "r4_beta": ("--r4-beta", float),           # residual R4 foregone-opportunity penalty weight
    "res_gamma": ("--res-gamma", float),
    "norm_reward": ("--norm-reward", bool),     # VecNormalize norm_reward (store_true)
    "dd_lambda": ("--dd-lambda", float),
    "dd_soft": ("--dd-soft", float),
    "ent_coef": ("--ent-coef", float),
    "lr": ("--lr", float),
    "lr_end": ("--lr-end", float),
    "episode_bars": ("--episode-bars", int),
    "max_entry_frac": ("--max-entry-frac", float),
    "stop_k": ("--stop-k", float),
    "cooldown": ("--cooldown", int),
}

DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_SMOKE_STEPS = 100_000
DEFAULT_N_ENVS = 8


def build_reward_args(reward_config: dict[str, Any]) -> list[str]:
    """Translate the freeform knob dict to train_event.py argv. Refuses unknown keys (typo guard)."""
    args: list[str] = []
    for k, v in reward_config.items():
        if k not in REWARD_KEYS:
            raise ValueError(f"unknown reward_config key {k!r}; allowed: {sorted(REWARD_KEYS)}")
        flag, kind = REWARD_KEYS[k]
        if kind is bool:
            if v:                       # store_true: present ⇒ on, absent ⇒ off
                args.append(flag)
        else:
            args += [flag, repr(v) if kind is float else str(v)]
    return args


def auto_prefix(reward_config: dict[str, Any], split: str) -> str:
    """A descriptive run-id stem from the shaping (e.g. ``ppo-event-residual`` / ``…-test``)."""
    mode = reward_config.get("reward_mode", "absolute")
    return f"ppo-event-{mode}" + ("-test" if split == "test" else "")


def _per_seed_cmd(python: str, reward_config: dict, split: str, timesteps: int, n_envs: int,
                  prefix: str, logdir: str) -> str:
    """One seed's command with `$s` left as a shell var the sweep loop substitutes."""
    base = [python, "scripts/train_event.py", "--timesteps", str(timesteps),
            "--n-envs", str(n_envs), "--eval-split", split, *build_reward_args(reward_config),
            "--seed", "$s", "--run-id", f"{prefix}-s$s"]
    inner = " ".join(shlex.quote(a) if a != "$s" and not a.endswith("-s$s") else a for a in base)
    return f"{inner} > {logdir}/{prefix}-s$s.log 2>&1"


def build_sweep_command(*, python: str, workdir: str, reward_config: dict, seeds: list,
                        split: str, timesteps: int = DEFAULT_TIMESTEPS, n_envs: int = DEFAULT_N_ENVS,
                        prefix: str) -> str:
    """The detached remote bash that **sequences** seeds (never parallel) and echoes the driver PID.

    Mirrors scripts/run_eventrung_sweep.sh's structure (mkdir logs, per-seed run-id, own log) but
    argv-direct from `reward_config`. `nohup … < /dev/null & echo $!` returns the driver bash PID.
    """
    logdir = f"runs-rl/{prefix}-logs"
    seed_str = " ".join(str(s) for s in seeds)
    loop = (f"for s in {seed_str}; do "
            f"{_per_seed_cmd(python, reward_config, split, timesteps, n_envs, prefix, logdir)}; done")
    return (f"cd {shlex.quote(workdir)} && mkdir -p runs-rl {logdir} && "
            f"nohup bash -c {shlex.quote(loop)} > runs-rl/{prefix}.log 2>&1 < /dev/null & echo $!")


def build_smoke_command(*, python: str, workdir: str, reward_config: dict, split: str,
                        smoke_steps: int = DEFAULT_SMOKE_STEPS, n_envs: int = DEFAULT_N_ENVS,
                        prefix: str) -> str:
    """A single FOREGROUND smoke run with publishing suppressed; returns the last lines for the gate.

    `APENTIC_PUBLISH_TARGET=` (empty) makes train_event.py skip publishing, so the gate never
    writes a bundle to the CDN. `tail -6` keeps the reply tiny (the [eval]/[verdict]/[train_event]
    lines sit at the end).
    """
    cmd = [python, "scripts/train_event.py", "--timesteps", str(smoke_steps), "--n-envs", str(n_envs),
           "--eval-split", split, *build_reward_args(reward_config),
           "--seed", "0", "--run-id", f"{prefix}-smoke"]
    inner = " ".join(shlex.quote(a) for a in cmd)
    return (f"cd {shlex.quote(workdir)} && mkdir -p runs-rl && "
            f"APENTIC_PUBLISH_TARGET= {inner} 2>&1 | tail -6")


_EVAL_RE = re.compile(r"\[eval\] events=(\d+) action mean=(-?[\d.]+) min=(-?[\d.]+) max=(-?[\d.]+)")
_DONE_RE = re.compile(r"\[train_event\][^:]*: return ([-+][\d.]+)%.*?trades (\d+)", re.DOTALL)


def parse_smoke(stdout: str, *, span_min: float = 0.5, mean_cap: float = 0.95) -> dict:
    """Smoke-gate: parse the smoke's tail and judge ALIVE (trades>0) + STRADDLE (action not pinned).

    `straddle` guards the [-1,1] boundary collapse that produced 0-trade duds: the action range must
    span (`max−min > span_min`) and its mean must sit inside the boundary (`|mean| < mean_cap`).
    Returns the parsed fields + `alive` / `straddle` / `passed`. A run that games reward but is dead
    or pinned does NOT pass — we won't burn a 4-seed sweep on a dud.
    """
    out: dict[str, Any] = {"alive": False, "straddle": False, "passed": False}
    ev = _EVAL_RE.search(stdout)
    if ev:
        out.update(events=int(ev.group(1)), action_mean=float(ev.group(2)),
                   action_min=float(ev.group(3)), action_max=float(ev.group(4)))
    dn = _DONE_RE.search(stdout)
    if dn:
        out.update(return_pct=float(dn.group(1)) / 100.0, trades=int(dn.group(2)))
    if "trades" in out and "action_mean" in out:
        out["alive"] = out["trades"] > 0
        out["straddle"] = ((out["action_max"] - out["action_min"]) > span_min
                           and abs(out["action_mean"]) < mean_cap)
        out["passed"] = out["alive"] and out["straddle"]
    else:
        out["error"] = "could not parse smoke output (no [eval]/[train_event] lines)"
    return out


def build_preflight_command(*, workdir: str, sha: str | None) -> str:
    """Sync to `sha` (if given) + confirm HEAD and that the gitignored market data is present.

    Tiny reply: ``HEAD=<sha> data=<n>`` where n>0 means data/ohlcv/hour_1 has files (market data
    lives only on the box). The driver checks HEAD == requested sha before launching.
    """
    sync = (f"git fetch --quiet && git checkout --quiet {shlex.quote(sha)} && " if sha else "")
    return (f"cd {shlex.quote(workdir)} && {sync}"
            "printf 'HEAD=%s data=%s\\n' \"$(git rev-parse --short HEAD)\" "
            "\"$(ls data/ohlcv/hour_1 2>/dev/null | wc -l)\"")


def parse_preflight(line: str) -> dict:
    """Parse ``HEAD=<sha> data=<n>`` into {head, data_files}."""
    out: dict[str, Any] = {}
    for tok in line.split():
        if tok.startswith("HEAD="):
            out["head"] = tok[5:]
        elif tok.startswith("data="):
            try:
                out["data_files"] = int(tok[5:])
            except ValueError:
                out["data_files"] = 0
    return out


# Kill the driver bash + train python by SPECIFIC PID, never the process group (the group-kill
# that dropped tailscaled). The `[t]`/`[r]` bracket trick keeps the matcher from killing itself;
# neither pattern can match `tailscaled`. Emits the killed PIDs (tiny).
_KILL_PATTERN = "[t]rain_event.py|[r]un_eventrung_sweep"
_KILL_ONELINER = (
    f"pids=$(pgrep -f '{_KILL_PATTERN}'); "
    "for p in $pids; do kill \"$p\" 2>/dev/null; done; "
    "echo \"killed=$(echo $pids | tr '\\n' ' ')\""
)


def build_kill_command() -> str:
    return _KILL_ONELINER


def parse_kill(stdout: str) -> dict:
    """Parse ``killed=<pids>`` into {killed_pids: [...]}."""
    pids: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("killed="):
            pids = line[len("killed="):].split()
    return {"killed_pids": pids, "n_killed": len(pids)}


def verify_launch(status: dict, n_envs: int) -> dict:
    """Judge whether exactly ONE clean run is on the box after a launch — the stack detector.

    `status` is a `remote.sweep_status()` reading taken ~60–90 s post-launch (after the torch
    import + volume-panel build). A single n_envs sweep loads the box ~n_envs and shows ~2 matched
    procs (driver bash + train main; the SubprocVecEnv workers don't carry the script name). Load
    well above n_envs, or >3 matched procs, means a SECOND sweep stacked — the Vmmem-throttle
    scenario — and the caller must abort/kill. Low load while `running` is just warm-up, not a fail.
    """
    running = bool(status.get("running"))
    load = status.get("load") or 0.0
    trainers = int(status.get("trainers") or 0)
    stacked = load > 1.7 * n_envs or trainers > 3
    clean = running and not stacked
    reason = ("STACKED run detected — abort/kill" if stacked
              else "not running (launch failed or finished early)" if not running
              else "one clean run")
    return {"clean": clean, "stacked": stacked, "running": running,
            "load": load, "trainers": trainers, "reason": reason}
