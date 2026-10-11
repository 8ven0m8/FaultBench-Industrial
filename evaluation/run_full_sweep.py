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
# --save-plots                 off by default. Saves one reward/fill
#                               plot per run as SVG to --plot-dir, named
#                               "Sweep_<date>_<run_id>_env_simulation.svg"
#                               (~4,170 files, roughly 2 GB, for the full
#                               grid incl. training-seed replicates and
#                               --include-baseline). Without it, no per-run
#                               plots are written at all.
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
# --train-seeds STR             default "auto". Which TRAINING-seed replicates
#                               of fault_tolerant_marl / supervisor to
#                               evaluate (see train_seeds.py). "auto" = every
#                               "_ts<k>" checkpoint found in ./models (falls
#                               back to the legacy single-seed checkpoint if
#                               there are none); "legacy" = only the legacy
#                               checkpoint; or a comma list like 1,2,3. Each
#                               replicate is a separate run (run_id gets a
#                               __ts<k> part), so those two mechanisms run
#                               (#replicates) x --seeds times per combo.
# --allow-llm-base-url          off by default. llm_replanning refuses to run
#                               while MARL_LLM_BASE_URL is set (i.e. against a
#                               proxy/auto-router instead of OpenAI itself),
#                               because the paper needs a known model. Pass
#                               this to run against a proxy anyway.
#
# Every llm_replanning record also carries its per-call llm_decision_log
# (action, reasoning, latency, error, served model, tokens) plus an
# llm_summary, and each LLM control episode is logged as its own
# status="control" record so its API calls are accounted for too.
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
from src.model_paths import (
    VANILLA_SORT_PATH, VANILLA_PRESS_PATH, find_ft_pair, find_supervisor,
    available_ft_train_seeds, available_supervisor_train_seeds,
)
from src.recovery.llm_agent import llm_endpoint_description
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

# Mechanisms that are themselves trained policies, evaluated once per
# training-seed replicate (see train_seeds.py / --train-seeds).
SEEDED_RECOVERIES = ("fault_tolerant_marl", "supervisor")

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


def combo_run_id(c, seed_index=None, train_seed=None):
    mode_part = f"__{c['mode_key']}-{c['mode_value']}" if c["mode_key"] else ""
    ts_part = f"__ts{train_seed}" if train_seed is not None else ""
    seed_part = f"__seed{seed_index}" if seed_index is not None else ""
    return (
        f"{c['fault']}__target-{c['target']}__dur-{c['duration_type']}"
        f"{mode_part}__recovery-{c['recovery']}{ts_part}{seed_part}"
    )


def resolve_train_seeds(spec):
    """--train-seeds -> {recovery_mode: [train_seed, ...]}. None in a list
    means the legacy single-seed checkpoint."""
    spec = spec.strip().lower()
    if spec == "auto":
        return {
            "fault_tolerant_marl": available_ft_train_seeds() or [None],
            "supervisor": available_supervisor_train_seeds() or [None],
        }
    if spec == "legacy":
        return {"fault_tolerant_marl": [None], "supervisor": [None]}
    seeds = [int(x) for x in spec.split(",") if x.strip()]
    return {"fault_tolerant_marl": seeds, "supervisor": seeds}


def summarize_llm_log(log):
    """Per-run rollup of an llm_decision_log, so the paper can report the
    real success rate, latency, served model and token cost."""
    log = log or []
    errors = [d for d in log if d.get("error")]
    ok_latencies = [d["latency_ms"] for d in log if not d.get("error") and d.get("latency_ms") is not None]
    actions = {}
    for d in log:
        actions[d.get("action")] = actions.get(d.get("action"), 0) + 1
    return {
        "n_calls": len(log),
        "n_errors": len(errors),
        "error_rate": round(len(errors) / len(log), 4) if log else None,
        "mean_latency_ms": round(sum(ok_latencies) / len(ok_latencies), 1) if ok_latencies else None,
        "models": sorted({d["model"] for d in log if d.get("model")}),
        "input_tokens": sum(d.get("input_tokens", 0) or 0 for d in log),
        "output_tokens": sum(d.get("output_tokens", 0) or 0 for d in log),
        "actions": actions,
        "errors": sorted({d["error"] for d in errors})[:5],
    }


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
        self._paths = {}
        self._supervisors = {}
        self._supervisor_paths = {}
        self._control_reward_data = {}
        self._wrapped_control_reward_data = {}
        # Extra JSONL records produced as a side effect (currently: the LLM
        # control episode's own API calls). main() flushes these to the log.
        self.pending_records = []

    def sort_press(self, fault_tolerant, train_seed=None):
        """train_seed only matters for fault_tolerant=True; the vanilla pair
        is a single checkpoint shared by every other mechanism."""
        key = ("ft", train_seed) if fault_tolerant else "vanilla"
        if key not in self._cache:
            if fault_tolerant:
                paths = find_ft_pair(train_seed)
            else:
                paths = (VANILLA_SORT_PATH, VANILLA_PRESS_PATH)
                if not all(os.path.exists(p) for p in paths):
                    paths = None
            self._paths[key] = paths
            self._cache[key] = (self.PPO.load(paths[0]), self.PPO.load(paths[1])) if paths else None
        return self._cache[key]

    def sort_press_paths(self, fault_tolerant, train_seed=None):
        key = ("ft", train_seed) if fault_tolerant else "vanilla"
        self.sort_press(fault_tolerant, train_seed)
        return self._paths.get(key)

    def supervisor(self, train_seed=None):
        if train_seed not in self._supervisors:
            path = find_supervisor(train_seed)
            self._supervisor_paths[train_seed] = path
            self._supervisors[train_seed] = self.PPO.load(path) if path else None
        return self._supervisors[train_seed]

    def supervisor_path(self, train_seed=None):
        self.supervisor(train_seed)
        return self._supervisor_paths.get(train_seed)

    def control_reward_data(self, fault_tolerant, steps_test, seed, train_seed=None):
        # Keyed on seed too, not just agent variant - a --seeds>1 sweep
        # calls this once per realization with a DIFFERENT episode seed,
        # and each realization needs its OWN matching control episode, not
        # realization 0's cached one reused for every subsequent seed. Also
        # keyed on train_seed: each fault-tolerant replicate is a different
        # policy and needs its own fault-free reference.
        key = (("ft", train_seed) if fault_tolerant else "vanilla", seed)
        if key not in self._control_reward_data:
            sp = self.sort_press(fault_tolerant, train_seed)
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

    def wrapped_control_reward_data(self, recovery_mode, steps_test, seed, train_seed=None):
        # see control_reward_data()'s note on why seed (and the supervisor's
        # train_seed) are part of the key
        key = (recovery_mode, train_seed, seed)
        if key in self._wrapped_control_reward_data:
            return self._wrapped_control_reward_data[key]

        result = None
        sp = self.sort_press(fault_tolerant=False)
        if sp is not None and not (
            recovery_mode == "llm_replanning" and not os.environ.get("OPENAI_API_KEY")
        ):
            supervisor_model = self.supervisor(train_seed) if recovery_mode == "supervisor" else None
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

                if recovery_mode == "llm_replanning":
                    # The control episode's detector can still fire (false
                    # alarms) and call the API - log those calls too, so
                    # cost and success-rate numbers cover every call made.
                    llm_log = list(getattr(control_env, "llm_decision_log", []) or [])
                    record = {
                        "run_id": f"control__llm_replanning__episode_seed{seed}",
                        "status": "control",
                        "recovery_mode": "llm_replanning",
                        "episode_seed": seed,
                        "llm_decision_log": llm_log,
                        "llm_summary": summarize_llm_log(llm_log),
                    }
                    record.update(llm_endpoint_description())
                    self.pending_records.append(record)

        self._wrapped_control_reward_data[key] = result
        return result


def run_one(c, run_id, models, steps_test, episode_seed, fault_seed, save_plots, plot_dir, tag_prefix,
            train_seed=None):
    fault_mode, recovery_mode = c["fault"], c["recovery"]
    fault_config = make_fault_config(c, fault_seed)

    if recovery_mode == "llm_replanning" and not os.environ.get("OPENAI_API_KEY"):
        return {"run_id": run_id, "status": "skipped", "reason": "OPENAI_API_KEY not set"}

    fault_tolerant = recovery_mode == "fault_tolerant_marl"
    ft_seed = train_seed if fault_tolerant else None
    sp = models.sort_press(fault_tolerant, ft_seed)
    if sp is None:
        return {"run_id": run_id, "status": "skipped",
                "reason": f"required model weights not found (train_seed={ft_seed})"}
    sort_model, press_model = sp

    if recovery_mode in ("none", "fault_tolerant_marl"):
        env_class = FAULT_REGISTRY[fault_mode]
    else:
        env_class = RECOVERY_REGISTRY[recovery_mode][fault_mode]

    env = env_class(max_steps=steps_test, seed=episode_seed, fault_config=fault_config)
    env.set_agents(sort_agent=sort_model, press_agent=press_model)

    sup_seed = train_seed if recovery_mode == "supervisor" else None
    if recovery_mode == "supervisor":
        supervisor_model = models.supervisor(sup_seed)
        if supervisor_model is None:
            return {"run_id": run_id, "status": "skipped",
                    "reason": f"no trained supervisor model found (train_seed={sup_seed})"}
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
        control_reward_data = models.control_reward_data(fault_tolerant, steps_test, episode_seed, ft_seed)
    else:
        control_reward_data = models.wrapped_control_reward_data(recovery_mode, steps_test, episode_seed, sup_seed)

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
    metrics["train_seed"] = train_seed if recovery_mode in SEEDED_RECOVERIES else None
    paths = models.sort_press_paths(fault_tolerant, ft_seed)
    metrics["sort_model_path"], metrics["press_model_path"] = paths if paths else (None, None)
    metrics["supervisor_model_path"] = models.supervisor_path(sup_seed) if recovery_mode == "supervisor" else None
    if recovery_mode == "llm_replanning":
        llm_log = list(getattr(env, "llm_decision_log", []) or [])
        metrics["llm_decision_log"] = llm_log
        metrics["llm_summary"] = summarize_llm_log(llm_log)
        metrics.update(llm_endpoint_description())
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
    p.add_argument("--save-plots", action="store_true",
                    help="Save one SVG plot per run to --plot-dir (off by default; ~2 GB for the full grid).")
    p.add_argument("--plot-dir", default="./img/figures/sweep/")
    p.add_argument("--log-path", default="./log/recovery_metrics_sweep.jsonl")
    p.add_argument("--resume", action="store_true",
                    help="Skip (combo, seed realization) pairs whose run_id already appears in --log-path.")
    p.add_argument("--limit", type=int, default=None,
                    help="Only run the first N combos per seed realization (smoke test).")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit without running anything.")
    p.add_argument("--train-seeds", default="auto",
                    help="Training-seed replicates of fault_tolerant_marl/supervisor to evaluate: "
                         "'auto' (every _ts<k> checkpoint found, else the legacy one), 'legacy', "
                         "or a comma list like 1,2,3. See train_seeds.py.")
    p.add_argument("--allow-llm-base-url", action="store_true",
                    help="Allow llm_replanning to run while MARL_LLM_BASE_URL is set (proxy / "
                         "auto-router). Off by default because the model identity is then unknown.")
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

    train_seeds = resolve_train_seeds(args.train_seeds)
    # (combo, train_seed) pairs: trained-policy mechanisms run once per
    # replicate, every other mechanism once (train_seed=None).
    runs = [(c, ts) for c in combos for ts in train_seeds.get(c["recovery"], [None])]

    total_runs = len(runs) * args.seeds + (args.seeds if args.include_baseline else 0)

    print(f"\n{'=' * 70}\nFaultBench-Industrial full sweep\n{'=' * 70}")
    print(f"Fault types : {faults}")
    print(f"Recoveries  : {recoveries}")
    print(f"Targets     : {targets}")
    print(f"Durations   : {durations}")
    for mech in SEEDED_RECOVERIES:
        if mech in recoveries:
            seeds_label = ["legacy" if t is None else t for t in train_seeds[mech]]
            print(f"Train seeds : {mech} -> {seeds_label}")
    print(f"Combos      : {len(combos)} ({len(runs)} incl. training-seed replicates)  x  "
          f"seeds: {args.seeds}  = {total_runs} run(s) total"
          f"{' (incl. 1 baseline per seed)' if args.include_baseline else ''}")

    if "llm_replanning" in recoveries:
        llm_cfg = llm_endpoint_description()
        print(f"LLM         : requested model={llm_cfg['requested_model']}, endpoint={llm_cfg['endpoint']}")
        if llm_cfg["endpoint"] != "OpenAI default" and not args.allow_llm_base_url and not args.dry_run:
            print("\n❌ MARL_LLM_BASE_URL is set, so llm_replanning would go through a proxy/auto-router "
                  "and the paper couldn't name the model. Unset it (or comment it out in .env) and set "
                  "MARL_LLM_MODEL=gpt-4o-mini, or pass --allow-llm-base-url to run anyway.")
            sys.exit(1)
    print(f"{'=' * 70}\n")

    if args.dry_run:
        for seed_index in range(args.seeds):
            episode_seed, fault_seed = derive_trial_seeds(args.seed, seed_index)
            if args.include_baseline:
                print(f"  baseline__no_fault__no_recovery__seed{seed_index}"
                      f"  (episode_seed={episode_seed})")
            for c, ts in runs:
                print("  " + combo_run_id(c, seed_index, ts)
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

        for i, (c, ts) in enumerate(runs, 1):
            run_id = combo_run_id(c, seed_index, ts)
            if args.resume and run_id in done:
                n_resumed += 1
                continue

            print(f"\n[seed {seed_index + 1}/{args.seeds}][{i}/{len(runs)}] {run_id}")
            result = run_one(c, run_id, models, args.steps, episode_seed, fault_seed,
                              args.save_plots, args.plot_dir, tag_prefix, train_seed=ts)
            result.setdefault("train_seed", ts)
            append_result(args.log_path, result)
            for extra in models.pending_records:
                append_result(args.log_path, extra)
            models.pending_records.clear()

            status = result.get("status")
            if status == "ok":
                n_ok += 1
                print_recovery_metrics(result)
                llm = result.get("llm_summary")
                if llm:
                    print(f"  LLM calls: {llm['n_calls']} ({llm['n_errors']} errors), "
                          f"mean latency {llm['mean_latency_ms']} ms, models {llm['models']}, "
                          f"tokens in/out {llm['input_tokens']}/{llm['output_tokens']}")
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