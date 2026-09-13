# ---------------------------------------------------------*\
# Title: Full Combinatorial Sweep Runner for FaultBench-Industrial
# ---------------------------------------------------------*/
#
# Runs main.py's evaluation logic (fault x recovery, but also target x
# duration x fault-specific mode) across the FULL combinatorial grid
# instead of one interactive run at a time.
#
# WHY THIS DOESN'T JUST IMPORT main.py:
# main.py's config-selection block calls input() at module import time
# (not behind `if __name__ == "__main__":`), so importing it - or piping
# answers into it via subprocess - can't be parameterized past what it
# already asks (fault type / transient-vs-permanent / duration /
# recovery mode). It never asks about `target` or the fault-specific
# `mode` keys (actuator degradation mode, comms_mode, byzantine_mode) -
# those only exist as hardcoded keys in DEFAULT_FAULT_CONFIG. So this
# script re-implements run_trained_modular_agents()'s logic directly
# against the same underlying modules, parameterized over every axis.
#
# AXES SWEPT (see README block at bottom / --dry-run for the full table):
#   fault_type   x  duration_type  x  target  x  fault_specific_mode  x  recovery_mode
#
# ---------------------------------------------------------------------------
# FLAG REFERENCE
# ---------------------------------------------------------------------------
# --faults STR                 comma list, default = all 5
#                               (sensor_noise,actuator_degradation,agent_dropout,
#                               comms_loss,byzantine)
# --recoveries STR             comma list, default = all 6
#                               (none,rule_based,rule_based_detect,
#                               fault_tolerant_marl,supervisor,llm_replanning)
# --targets STR                comma list, default = all 3 (sort,press,both)
# --durations STR              comma list, default = both (transient,permanent)
# --include-noop-comms-target  off by default. Also runs comms_loss with
#                               target=sort/both, which are documented no-ops
#                               (no sort->press channel exists to sever) -
#                               off by default to skip 8 wasted runs.
# --skip-llm                   off by default. Shortcut that drops
#                               llm_replanning from --recoveries, so you
#                               don't burn API calls on a first pass.
# --steps INT                  default 200. Eval episode length, matches
#                               main.py's STEPS_TEST.
# --seed INT                   default 42. Base/master seed. Each of the
#                               --seeds realizations below derives its OWN
#                               episode/material seed AND fault-injection
#                               seed from (this value, realization index) -
#                               deterministic and reproducible given the
#                               same --seed, but genuinely different each
#                               realization. Matches main.py's SEED only
#                               for realization 0's episode seed by
#                               coincidence of the derivation, not by
#                               design - don't rely on that.
# --seeds INT                   default 8. Number of independent seed
#                               realizations to run for every combo. A
#                               single realization (the old default
#                               behavior) cannot distinguish a mechanism's
#                               real effect from this one random draw's
#                               noise - see the null-result / benchmark
#                               scoring discussion. Each realization's rows
#                               get a __seedN suffix on run_id so they
#                               don't collide in --log-path and can be
#                               grouped/averaged later (see
#                               aggregate_benchmark_scores.py).
# --transient-duration INT     default 20. Fixed step-count for transient
#                               faults, matches main.py's
#                               DEFAULT_TRANSIENT_DURATION.
# --save-plots                 off by default. Turns on test_env's
#                               matplotlib save (312 plot files if left on
#                               for the full grid).
# --plot-dir STR                default "./img/figures/sweep/". Where those
#                               plots go if --save-plots is enabled.
# --log-path STR                default "./log/recovery_metrics_sweep.jsonl".
#                               Where every run's metrics get appended
#                               (kept separate from the existing
#                               log/recovery_metrics.jsonl).
# --resume                     off by default. Skips any (combo, seed
#                               realization) pair whose run_id is already
#                               present in --log-path, so a killed/
#                               interrupted multi-seed sweep can pick back
#                               up mid-way through any realization.
# --limit N                    default none. Caps the combo grid to the
#                               first N combos PER seed realization (so
#                               total runs = N x --seeds), for smoke-
#                               testing before committing to the full grid.
# --dry-run                    off by default. Prints every combo's run_id
#                               and the total count, runs nothing.
# --include-baseline          # + one clean no-fault/no-recovery run
#
# Duration is one of the five swept axes, not an afterthought: every combo
# is fault x target x duration_type x fault-specific mode x recovery, and
# duration_type in {transient, permanent} is what supplies the "x2" in
# every fault-type count (e.g. actuator_degradation: 3 targets x 2
# durations x 3 modes = 18). Transient runs set duration=<transient
# value> / duration_range=None; permanent runs set both duration=None and
# duration_range=None, matching main.py's own "persists for the rest of
# the episode" encoding.
# ---------------------------------------------------------------------------

import argparse
import itertools
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

# This file lives in evaluation/, one level below the repo root. Anchor
# BOTH imports (sys.path) AND every relative path this file and everything
# it imports uses (./models, ./log, ./img, config.yml, find_latest_model's
# "./models" scan, ...) to the repo root via chdir - not just sys.path.
# Without the chdir, running this as `python evaluation/run_full_sweep.py`
# from inside evaluation/ itself (or from an IDE that defaults cwd to the
# script's own directory) silently looks for ./models/... under
# evaluation/models/ instead of the repo root and fails to find any
# checkpoint.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

# --- repo modules (same imports main.py makes) -----------------------------
from src.envs_train.env_combined import Env_Combined
from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault
from src.recovery.rule_based import RULE_BASED_REGISTRY
from src.recovery.rule_based_detect import RULE_BASED_DETECT_REGISTRY
from src.recovery.supervisor import SUPERVISOR_REGISTRY
from src.recovery.llm_replanning import LLM_REGISTRY

from src.testing import test_env
from src.training import find_latest_model
from utils.metrics import compute_recovery_metrics, print_recovery_metrics

FAULT_REGISTRY = {
    "sensor_noise": Env_SensorNoiseFault,
    "actuator_degradation": Env_ActuatorDegradationFault,
    "agent_dropout": Env_AgentDropoutFault,
    "comms_loss": Env_CommsLossFault,
    "byzantine": Env_ByzantineFault,
}

RECOVERY_REGISTRY = {
    "rule_based": RULE_BASED_REGISTRY,
    "rule_based_detect": RULE_BASED_DETECT_REGISTRY,
    "supervisor": SUPERVISOR_REGISTRY,
    "llm_replanning": LLM_REGISTRY,
}

ALL_FAULTS = ["sensor_noise", "actuator_degradation", "agent_dropout", "comms_loss", "byzantine"]
ALL_RECOVERIES = ["none", "rule_based", "rule_based_detect", "fault_tolerant_marl", "supervisor", "llm_replanning"]
ALL_TARGETS = ["sort", "press", "both"]
ALL_DURATIONS = ["transient", "permanent"]

# fault-specific categorical modes; None = fault type has no extra mode axis
FAULT_MODE_AXIS = {
    "sensor_noise": {"key": None, "values": [None]},
    "actuator_degradation": {"key": "mode", "values": ["stuck", "restrict", "slip"]},
    "agent_dropout": {"key": None, "values": [None]},
    "comms_loss": {"key": "comms_mode", "values": ["stale", "blackout"]},
    "byzantine": {"key": "byzantine_mode", "values": ["fixed_malicious", "worst_action", "random_malicious"]},
}

# Base config mirroring main.py's DEFAULT_FAULT_CONFIG (non-swept keys keep
# main.py's defaults; swept keys - target/duration/mode - are overridden
# per-combo below, and "seed" is always overridden per-realization by
# make_fault_config() - never left at a fixed value across a --seeds>1
# sweep, since that RNG governs injection timing within the window plus
# the slip/random_malicious coin-flips, independently of the episode's
# own material-generation seed).
BASE_FAULT_CONFIG = {
    "injection_step_range": (40, 60),
    "noise_std": 0.1,
    "stuck_press_action": 0,
    "stuck_sort_mode": None,
    "restrict_sort_mode": 1,
    "restrict_press_id": 2,
    "degradation_prob": 0.3,
    "dropout_sort_mode": 0,
    "dropout_press_action": 0,
    "malicious_sort_mode": None,
    "malicious_press_action": None,
}


def build_combos(faults, recoveries, targets, durations, include_noop_comms_target, transient_duration):
    """Yields dicts describing every (fault, target, duration, mode, recovery) combo."""
    for fault in faults:
        mode_key = FAULT_MODE_AXIS[fault]["key"]
        mode_values = FAULT_MODE_AXIS[fault]["values"]

        fault_targets = targets
        if fault == "comms_loss" and not include_noop_comms_target:
            # target="sort" (and "both", which behaves identically to "press"
            # for this fault type - see env_fault_comms_loss.py) collapse to
            # just "press" being meaningful; drop the redundant/no-op ones.
            fault_targets = [t for t in targets if t == "press"]
            if not fault_targets:
                continue

        for target, duration_type, mode_value in itertools.product(fault_targets, durations, mode_values):
            for recovery in recoveries:
                yield {
                    "fault": fault,
                    "target": target,
                    "duration_type": duration_type,
                    "mode_key": mode_key,
                    "mode_value": mode_value,
                    "recovery": recovery,
                    "transient_duration": transient_duration,
                }


def combo_run_id(c, seed_index=None):
    mode_part = f"__{c['mode_key']}-{c['mode_value']}" if c["mode_key"] else ""
    seed_part = f"__seed{seed_index}" if seed_index is not None else ""
    return (
        f"{c['fault']}__target-{c['target']}__dur-{c['duration_type']}"
        f"{mode_part}__recovery-{c['recovery']}{seed_part}"
    )


def derive_trial_seeds(base_seed, seed_index):
    """Deterministically derive one realization's (episode_seed, fault_seed)
    from the sweep's base --seed and a 0-based realization index. Fully
    reproducible given the same --seed, but each realization gets a
    genuinely different material-generation world AND a genuinely
    different fault-injection RNG stream - both axes matter independently
    (a single shared seed across realizations would just re-run the exact
    same world/fault-timing combo --seeds times, which defeats the point).
    Uses a tuple seed (numpy hashes it via SeedSequence) rather than
    base_seed + seed_index so different --seed values can't accidentally
    collide on the same derived seed for different realization indices."""
    trial_rng = np.random.default_rng((int(base_seed), int(seed_index)))
    episode_seed = int(trial_rng.integers(0, 2**31 - 1))
    fault_seed = int(trial_rng.integers(0, 2**31 - 1))
    return episode_seed, fault_seed


def make_fault_config(c, fault_seed):
    cfg = dict(BASE_FAULT_CONFIG)
    cfg["seed"] = fault_seed
    cfg["target"] = c["target"]
    if c["duration_type"] == "transient":
        cfg["duration"] = c["transient_duration"]
        cfg["duration_range"] = None
    else:
        cfg["duration"] = None
        cfg["duration_range"] = None
    if c["mode_key"]:
        cfg[c["mode_key"]] = c["mode_value"]
    return cfg


def load_existing_run_ids(log_path):
    if not os.path.exists(log_path):
        return set()
    seen = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line).get("run_id"))
            except json.JSONDecodeError:
                continue
    return seen


def append_result(log_path, record):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


class ModelCache:
    """Loads each PPO model at most once and reuses it across combos.
    Also caches one no-fault CONTROL episode's reward_data per agent
    variant (vanilla vs fault-tolerant) - this feeds compute_recovery_metrics'
    null-correction (see utils/metrics.py's control_reward_data param),
    which nets out the natural non-stationary-reward artifact that a single
    fixed-seed episode shows even with zero fault effect. One control run
    per agent variant is enough since the episode is fully deterministic
    given the same seed/agents/steps - it's computed once, on first use,
    and reused across every combo that shares that agent variant.

    For rule_based / rule_based_detect / supervisor / llm_replanning, a
    plain Env_Combined control run isn't the right reference: those
    recovery mechanisms run their own detection/intervention machinery
    every step regardless of whether a fault is active, so they can carry
    their own baseline behavior (e.g. a detector's residual false-positive
    rate) that plain Env_Combined never exercises. wrapped_control_reward_data()
    instead runs the SAME recovery-wrapped environment class with the fault
    injection pushed past the episode end, so that mechanism-specific
    baseline gets captured and netted out too, not just generic reward
    non-stationarity. One canonical fault_mode ("sensor_noise") is enough
    per recovery_mode: with the fault disabled, fault_context is inert for
    the whole episode, so the mixin's detection/intervention behavior is
    identical regardless of which of the 5 fault base classes it's mixed
    into."""

    def __init__(self):
        from stable_baselines3 import PPO
        self.PPO = PPO
        self._cache = {}
        self._supervisor = None
        self._supervisor_loaded = False
        self._control_reward_data = {}
        self._wrapped_control_reward_data = {}

    def sort_press(self, fault_tolerant):
        key = "ft" if fault_tolerant else "vanilla"
        if key not in self._cache:
            if fault_tolerant:
                sort_path = "./models/PPO_Sorting_FaultTolerant_NoMask_10000000.zip"
                press_path = "./models/PPO_Pressing_FaultTolerant_NoMask_10000000.zip"
            else:
                sort_path = "./models/PPO_Sorting_NoMask_10000000.zip"
                press_path = "./models/PPO_Pressing_NoMask_10000000.zip"
            if not (os.path.exists(sort_path) and os.path.exists(press_path)):
                self._cache[key] = None
            else:
                self._cache[key] = (self.PPO.load(sort_path), self.PPO.load(press_path))
        return self._cache[key]

    def supervisor(self):
        if not self._supervisor_loaded:
            self._supervisor_loaded = True
            path = find_latest_model("PPO_Supervisor_NoMask")
            self._supervisor = self.PPO.load(path) if path else None
        return self._supervisor

    def control_reward_data(self, fault_tolerant, steps_test, seed):
        # Keyed on seed too, not just agent variant - a --seeds>1 sweep
        # calls this once per realization with a DIFFERENT episode seed,
        # and each realization needs its OWN matching control episode, not
        # realization 0's cached one reused for every subsequent seed.
        key = ("ft" if fault_tolerant else "vanilla", seed)
        if key not in self._control_reward_data:
            sp = self.sort_press(fault_tolerant)
            if sp is None:
                self._control_reward_data[key] = None
            else:
                sort_model, press_model = sp
                control_env = Env_Combined(max_steps=steps_test, seed=seed)
                control_env.set_agents(sort_agent=sort_model, press_agent=press_model)
                control_env.reset(seed=seed)
                for _ in range(steps_test):
                    _, _, terminated, truncated, _ = control_env.step()
                    if terminated or truncated:
                        break
                self._control_reward_data[key] = control_env.reward_data
        return self._control_reward_data[key]

    def wrapped_control_reward_data(self, recovery_mode, steps_test, seed):
        key = (recovery_mode, seed)  # see control_reward_data()'s note on why seed is part of the key
        if key in self._wrapped_control_reward_data:
            return self._wrapped_control_reward_data[key]

        result = None
        sp = self.sort_press(fault_tolerant=False)
        if sp is not None and not (
            recovery_mode == "llm_replanning" and not os.environ.get("OPENAI_API_KEY")
        ):
            supervisor_model = self.supervisor() if recovery_mode == "supervisor" else None
            if recovery_mode != "supervisor" or supervisor_model is not None:
                sort_model, press_model = sp
                env_class = RECOVERY_REGISTRY[recovery_mode]["sensor_noise"]
                no_fault_config = dict(BASE_FAULT_CONFIG)
                no_fault_config["seed"] = seed  # never actually consumed since the fault never fires, but avoid relying on that
                # Push injection past the episode end so the fault never
                # activates - fault_context stays inert for every step,
                # isolating the recovery mixin's own baseline behavior.
                no_fault_config["injection_step_range"] = (steps_test + 1000, steps_test + 1001)

                control_env = env_class(max_steps=steps_test, seed=seed, fault_config=no_fault_config)
                control_env.set_agents(sort_agent=sort_model, press_agent=press_model)
                if recovery_mode == "supervisor":
                    control_env.set_supervisor(supervisor_model)
                control_env.reset(seed=seed)
                for _ in range(steps_test):
                    _, _, terminated, truncated, _ = control_env.step()
                    if terminated or truncated:
                        break
                result = control_env.reward_data

        self._wrapped_control_reward_data[key] = result
        return result


def run_one(c, run_id, models, steps_test, episode_seed, fault_seed, save_plots, plot_dir, tag_prefix):
    fault_mode, recovery_mode = c["fault"], c["recovery"]
    fault_config = make_fault_config(c, fault_seed)

    if recovery_mode == "llm_replanning" and not os.environ.get("OPENAI_API_KEY"):
        return {"run_id": run_id, "status": "skipped", "reason": "OPENAI_API_KEY not set"}

    fault_tolerant = recovery_mode == "fault_tolerant_marl"
    sp = models.sort_press(fault_tolerant)
    if sp is None:
        return {"run_id": run_id, "status": "skipped", "reason": "required model weights not found"}
    sort_model, press_model = sp

    if recovery_mode in ("none", "fault_tolerant_marl"):
        env_class = FAULT_REGISTRY[fault_mode]
    else:
        env_class = RECOVERY_REGISTRY[recovery_mode][fault_mode]

    env = env_class(max_steps=steps_test, seed=episode_seed, fault_config=fault_config)
    env.set_agents(sort_agent=sort_model, press_agent=press_model)

    if recovery_mode == "supervisor":
        supervisor_model = models.supervisor()
        if supervisor_model is None:
            return {"run_id": run_id, "status": "skipped", "reason": "no trained supervisor model found"}
        env.set_supervisor(supervisor_model)

    eval_tag = f"{tag_prefix}_{run_id}"
    try:
        test_env(
            env=env, tag=eval_tag, save=save_plots, show=False,
            title=f"[fault:{fault_mode} target:{c['target']} dur:{c['duration_type']}"
                  f"{' ' + str(c['mode_value']) if c['mode_value'] else ''} recovery:{recovery_mode}]",
            steps=steps_test, dir=plot_dir, seed=episode_seed,
        )
    except Exception as e:
        return {"run_id": run_id, "status": "error", "reason": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}

    if recovery_mode in ("none", "fault_tolerant_marl"):
        control_reward_data = models.control_reward_data(fault_tolerant, steps_test, episode_seed)
    else:
        control_reward_data = models.wrapped_control_reward_data(recovery_mode, steps_test, episode_seed)

    metrics = compute_recovery_metrics(
        env, fault_mode=fault_mode, recovery_mode=recovery_mode, seed=episode_seed,
        control_reward_data=control_reward_data,
    )
    metrics["run_id"] = run_id
    metrics["status"] = "ok"
    metrics["target"] = c["target"]
    metrics["duration_type"] = c["duration_type"]
    metrics["episode_seed"] = episode_seed
    metrics["fault_seed"] = fault_seed
    if c["mode_key"]:
        metrics[c["mode_key"]] = c["mode_value"]
    return metrics


def run_baseline(run_id, models, steps_test, episode_seed, save_plots, plot_dir, tag_prefix):
    """
    Clean-episode reference run: plain Env_Combined, no fault injected, no
    recovery mechanism attached (recovery_mode="none" is already covered by
    the main grid for every FAULT-ACTIVE combo - this is the separate
    "nothing was ever broken" baseline main.py's fault_mode="none" branch
    produces, which the grid never touches since it only iterates over the
    5 real fault types).
    """
    sp = models.sort_press(fault_tolerant=False)
    if sp is None:
        return {"run_id": run_id, "status": "skipped", "reason": "vanilla model weights not found"}
    sort_model, press_model = sp

    env = Env_Combined(max_steps=steps_test, seed=episode_seed)
    env.set_agents(sort_agent=sort_model, press_agent=press_model)

    eval_tag = f"{tag_prefix}_{run_id}"
    try:
        test_env(
            env=env, tag=eval_tag, save=save_plots, show=False,
            title="[baseline: no fault, no recovery]",
            steps=steps_test, dir=plot_dir, seed=episode_seed,
        )
    except Exception as e:
        return {"run_id": run_id, "status": "error", "reason": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}

    # fault_log/detection_log don't exist on plain Env_Combined - compute_recovery_metrics
    # falls back to [] for both via getattr(), so this still returns a clean
    # total_reward / episode_safety_violations record with an empty fault_events list.
    metrics = compute_recovery_metrics(env, fault_mode="none", recovery_mode="none", seed=episode_seed)
    metrics["run_id"] = run_id
    metrics["status"] = "ok"
    metrics["target"] = None
    metrics["duration_type"] = None
    metrics["episode_seed"] = episode_seed
    return metrics


def main():
    p = argparse.ArgumentParser(description="Run the full FaultBench-Industrial combinatorial sweep.")
    p.add_argument("--faults", default=",".join(ALL_FAULTS))
    p.add_argument("--recoveries", default=",".join(ALL_RECOVERIES))
    p.add_argument("--targets", default=",".join(ALL_TARGETS))
    p.add_argument("--durations", default=",".join(ALL_DURATIONS))
    p.add_argument("--include-noop-comms-target", action="store_true",
                    help="Also run comms_loss with target=sort/both, which are documented no-ops.")
    p.add_argument("--include-baseline", action="store_true",
                    help="Also run one clean episode with no fault and no recovery (the "
                         "\"healthy system\" reference run) - logged separately, doesn't "
                         "count toward --limit.")
    p.add_argument("--skip-llm", action="store_true", help="Shortcut for --recoveries without llm_replanning.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42, help="Base/master seed. See --seeds.")
    p.add_argument("--seeds", type=int, default=8,
                    help="Number of independent seed realizations to run for every combo. "
                         "Each realization derives its own episode/material seed AND "
                         "fault-injection seed from (--seed, realization index) - see "
                         "derive_trial_seeds(). Pass --seeds 1 to reproduce the old "
                         "single-shot behavior (not recommended - see the benchmark-score "
                         "discussion on why n=1 can't distinguish a real effect from noise).")
    p.add_argument("--transient-duration", type=int, default=20)
    p.add_argument("--save-plots", action="store_true", help="Also save the per-run reward plot (slower, lots of files).")
    p.add_argument("--plot-dir", default="./img/figures/sweep/")
    p.add_argument("--log-path", default="./log/recovery_metrics_sweep.jsonl")
    p.add_argument("--resume", action="store_true",
                    help="Skip (combo, seed realization) pairs whose run_id already appears in --log-path.")
    p.add_argument("--limit", type=int, default=None,
                    help="Only run the first N combos per seed realization (smoke test).")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit without running anything.")
    args = p.parse_args()

    faults = [f.strip() for f in args.faults.split(",") if f.strip()]
    recoveries = [r.strip() for r in args.recoveries.split(",") if r.strip()]
    if args.skip_llm:
        recoveries = [r for r in recoveries if r != "llm_replanning"]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    durations = [d.strip() for d in args.durations.split(",") if d.strip()]

    for f in faults:
        assert f in ALL_FAULTS, f"Unknown fault '{f}'"
    for r in recoveries:
        assert r in ALL_RECOVERIES, f"Unknown recovery '{r}'"

    combos = list(build_combos(faults, recoveries, targets, durations,
                                args.include_noop_comms_target, args.transient_duration))

    if args.limit is not None:
        combos = combos[: args.limit]

    total_runs = len(combos) * args.seeds + (args.seeds if args.include_baseline else 0)

    print(f"\n{'=' * 70}\nFaultBench-Industrial full sweep\n{'=' * 70}")
    print(f"Fault types : {faults}")
    print(f"Recoveries  : {recoveries}")
    print(f"Targets     : {targets}")
    print(f"Durations   : {durations}")
    print(f"Combos      : {len(combos)}  x  seeds: {args.seeds}  = {total_runs} run(s) total"
          f"{' (incl. 1 baseline per seed)' if args.include_baseline else ''}")
    print(f"{'=' * 70}\n")

    if args.dry_run:
        for seed_index in range(args.seeds):
            episode_seed, fault_seed = derive_trial_seeds(args.seed, seed_index)
            if args.include_baseline:
                print(f"  baseline__no_fault__no_recovery__seed{seed_index}"
                      f"  (episode_seed={episode_seed})")
            for c in combos:
                print("  " + combo_run_id(c, seed_index)
                      + f"  (episode_seed={episode_seed}, fault_seed={fault_seed})")
        print(f"\n[dry-run] {total_runs} run(s) would run. Nothing executed.")
        return

    done = load_existing_run_ids(args.log_path) if args.resume else set()

    models = ModelCache()
    tag_prefix = f"Sweep_{datetime.now().strftime('%d-%m-%Y_%H-%M')}"

    n_ok = n_skip = n_err = n_resumed = 0
    t0 = time.time()

    for seed_index in range(args.seeds):
        episode_seed, fault_seed = derive_trial_seeds(args.seed, seed_index)
        print(f"\n{'#' * 70}\n# Seed realization {seed_index + 1}/{args.seeds}"
              f"  (episode_seed={episode_seed}, fault_seed={fault_seed})\n{'#' * 70}")

        if args.include_baseline:
            baseline_run_id = f"baseline__no_fault__no_recovery__seed{seed_index}"
            if args.resume and baseline_run_id in done:
                n_resumed += 1
                print(f"\n[resume] {baseline_run_id} already in log, skipping.")
            else:
                print(f"\n[baseline] {baseline_run_id}")
                result = run_baseline(baseline_run_id, models, args.steps, episode_seed,
                                       args.save_plots, args.plot_dir, tag_prefix)
                append_result(args.log_path, result)
                status = result.get("status")
                if status == "ok":
                    n_ok += 1
                    print_recovery_metrics(result)
                elif status == "skipped":
                    n_skip += 1
                    print(f"  ⏭️  skipped: {result.get('reason')}")
                else:
                    n_err += 1
                    print(f"  ❌ error: {result.get('reason')}")

        for i, c in enumerate(combos, 1):
            run_id = combo_run_id(c, seed_index)
            if args.resume and run_id in done:
                n_resumed += 1
                continue

            print(f"\n[seed {seed_index + 1}/{args.seeds}][{i}/{len(combos)}] {run_id}")
            result = run_one(c, run_id, models, args.steps, episode_seed, fault_seed,
                              args.save_plots, args.plot_dir, tag_prefix)
            append_result(args.log_path, result)

            status = result.get("status")
            if status == "ok":
                n_ok += 1
                print_recovery_metrics(result)
            elif status == "skipped":
                n_skip += 1
                print(f"  ⏭️  skipped: {result.get('reason')}")
            else:
                n_err += 1
                print(f"  ❌ error: {result.get('reason')}")

    elapsed = time.time() - t0
    total_run = n_ok + n_skip + n_err
    print(f"\n{'=' * 70}")
    print(f"Sweep complete in {elapsed / 60:.1f} min — {total_run} run(s) attempted across "
          f"{args.seeds} seed realization(s) "
          f"(ok: {n_ok}, skipped: {n_skip}, errors: {n_err}, already-done/resumed: {n_resumed})")
    print(f"Results appended to: {args.log_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()