"""
Full pre-flight sanity battery -- run this BEFORE trusting the seeded matrix
for the paper. Covers all 5 fault types x all 6 recovery mechanisms
(none, rule_based, rule_based_detect, fault_tolerant_marl, supervisor,
llm_replanning) x both duration modes (permanent, transient).

Mirrors main.py's actual dispatch logic in run_trained_modular_agents() --
NOT imported directly from main.py, since main.py runs interactive input()
prompts at module level and would block/misbehave on import. The registry
wiring below is copied from main.py's own imports/dicts; if you add a new
fault type or recovery mechanism there, mirror it here too.

Usage: python evaluation/verify_workings.py (run from the repo root)
Tip: set RUN_LLM_CHECKS = False while iterating on everything else, then
flip it back on for a final pass -- llm_replanning makes real API calls.
"""
import os
import sys

# This file lives in evaluation/, one level below the repo root. Anchor
# BOTH imports (sys.path) AND every relative path this file and everything
# it imports uses (./models, config.yml, find_latest_model's "./models"
# scan, .env lookup, ...) to the repo root via chdir - not just sys.path -
# so this works whether invoked from the repo root, from inside
# evaluation/, or from an IDE that defaults cwd to the script's own
# directory. Done before any other import, including load_dotenv() below.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

import numpy as np
from stable_baselines3 import PPO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.training import find_latest_model
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
from utils.metrics import compute_recovery_metrics

# ---------------------------------------------------------------------------
# Config -- tune these before running
# ---------------------------------------------------------------------------
STEPS = 200
N_SEEDS_STANDARD = 3   # seeds for non-LLM mechanisms in the full matrix pass
N_SEEDS_LLM = 2        # keep this low by default -- llm_replanning costs real API calls
RUN_LLM_CHECKS = True  # flip False to skip llm_replanning entirely while iterating
RUN_FULL_MATRIX_SMOKE_TEST = True  # Phase 5 below; the expensive/complete pass

FAULT_CLASSES = {
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
RECOVERY_MODES = ["none", "rule_based", "rule_based_detect", "fault_tolerant_marl", "supervisor", "llm_replanning"]

DURATION_MODES = {
    "permanent": {"duration": None, "duration_range": None},
    "transient": {"duration": None, "duration_range": (10, 40)},  # matches TRANSIENT_DURATION_RANGE used elsewhere
}
BASE_FAULT_CONFIG = {"injection_step_range": (40, 60), "target": "both"}

VANILLA_SORT_PATH = "./models/PPO_Sorting_NoMask_10000000.zip"
VANILLA_PRESS_PATH = "./models/PPO_Pressing_NoMask_10000000.zip"
FT_SORT_PATH = "./models/PPO_Sorting_FaultTolerant_NoMask_10000000.zip"
FT_PRESS_PATH = "./models/PPO_Pressing_FaultTolerant_NoMask_10000000.zip"

failures = []
warnings_ = []


def flag(msg, hard=True):
    (failures if hard else warnings_).append(msg)


# ---------------------------------------------------------------------------
# Phase 0 -- artifact inventory. Run this first: no point discovering a
# missing model 40 minutes into a matrix run.
# ---------------------------------------------------------------------------
print("=" * 90)
print("PHASE 0 -- artifact inventory")
print("=" * 90)

have_vanilla = os.path.exists(VANILLA_SORT_PATH) and os.path.exists(VANILLA_PRESS_PATH)
have_ft = os.path.exists(FT_SORT_PATH) and os.path.exists(FT_PRESS_PATH)
supervisor_path = find_latest_model("PPO_Supervisor_NoMask")
have_supervisor = supervisor_path is not None
have_openai_key = bool(os.environ.get("OPENAI_API_KEY"))

print(f"  vanilla sort+press models   : {'FOUND' if have_vanilla else 'MISSING'}")
print(f"  fault-tolerant sort+press   : {'FOUND' if have_ft else 'MISSING'}")
print(f"  supervisor model            : {'FOUND @ ' + supervisor_path if have_supervisor else 'MISSING'}")
print(f"  OPENAI_API_KEY              : {'SET' if have_openai_key else 'NOT SET'}")

if not have_vanilla:
    flag("vanilla sort/press models missing -- almost nothing below can run without these")
if not have_ft:
    warnings_.append("fault_tolerant_marl models missing -- that mechanism will be skipped below")
if not have_supervisor:
    warnings_.append("supervisor model missing -- that mechanism will be skipped below")
if not have_openai_key:
    warnings_.append("OPENAI_API_KEY not set -- llm_replanning will be skipped below")
    RUN_LLM_CHECKS = False

if not have_vanilla:
    print("\nFATAL: cannot proceed without vanilla models. Fix and re-run.")
    raise SystemExit(1)

vanilla_sort_model = PPO.load(VANILLA_SORT_PATH)
vanilla_press_model = PPO.load(VANILLA_PRESS_PATH)
ft_sort_model = PPO.load(FT_SORT_PATH) if have_ft else None
ft_press_model = PPO.load(FT_PRESS_PATH) if have_ft else None
supervisor_model = PPO.load(supervisor_path) if have_supervisor else None

# Sanity: fault_tolerant weights should NOT be byte-identical to vanilla --
# catches a copy/train-forgot-to-overwrite bug where FT "recovery" is a no-op
# because it's secretly loading the same weights as vanilla.
if have_ft:
    import torch
    vanilla_params = torch.cat([p.flatten() for p in vanilla_sort_model.policy.parameters()])
    ft_params = torch.cat([p.flatten() for p in ft_sort_model.policy.parameters()])
    identical = vanilla_params.shape == ft_params.shape and torch.allclose(vanilla_params, ft_params)
    print(f"  FT sort weights vs vanilla  : {'IDENTICAL <-- SUSPICIOUS' if identical else 'different (expected)'}")
    if identical:
        flag("fault_tolerant sort model has IDENTICAL weights to vanilla -- likely loading the wrong checkpoint")

# ---------------------------------------------------------------------------
# Phase 1 -- registry completeness. Every fault type must have a combo class
# for every recovery mechanism that claims to support it. Pure wiring check,
# no simulation, catches a missing entry before you ever try to run it.
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("PHASE 1 -- registry completeness (every fault type x every recovery mechanism)")
print("=" * 90)
for recovery_name, registry in RECOVERY_REGISTRY.items():
    missing = [f for f in FAULT_CLASSES if f not in registry]
    status = "OK" if not missing else f"FAIL <-- missing: {missing}"
    print(f"  {recovery_name:20s} {status}")
    if missing:
        flag(f"{recovery_name}: missing combo class for fault types {missing}")


def make_fault_config(duration_mode, seed):
    cfg = dict(BASE_FAULT_CONFIG)
    cfg.update(DURATION_MODES[duration_mode])
    cfg["seed"] = seed
    return cfg


def get_env_and_models(fault_name, recovery_mode):
    """Mirrors run_trained_modular_agents()'s dispatch logic in main.py.
    Returns (env_class, sort_model, press_model, needs_supervisor, available, reason)."""
    if recovery_mode == "fault_tolerant_marl":
        if not have_ft:
            return None, None, None, False, False, "FT models not trained yet"
        return FAULT_CLASSES[fault_name], ft_sort_model, ft_press_model, False, True, None
    if recovery_mode == "none":
        return FAULT_CLASSES[fault_name], vanilla_sort_model, vanilla_press_model, False, True, None
    if recovery_mode == "supervisor":
        if not have_supervisor:
            return None, None, None, False, False, "supervisor model not trained yet"
        return SUPERVISOR_REGISTRY[fault_name], vanilla_sort_model, vanilla_press_model, True, True, None
    if recovery_mode == "llm_replanning":
        if not have_openai_key:
            return None, None, None, False, False, "OPENAI_API_KEY not set"
        return LLM_REGISTRY[fault_name], vanilla_sort_model, vanilla_press_model, False, True, None
    # rule_based / rule_based_detect
    return RECOVERY_REGISTRY[recovery_mode][fault_name], vanilla_sort_model, vanilla_press_model, False, True, None


def run_episode(fault_name, recovery_mode, duration_mode, seed):
    env_class, sort_model, press_model, needs_supervisor, available, reason = get_env_and_models(fault_name, recovery_mode)
    if not available:
        return None, reason
    fault_config = make_fault_config(duration_mode, seed)
    env = env_class(max_steps=STEPS, seed=seed, fault_config=fault_config)
    env.set_agents(sort_agent=sort_model, press_agent=press_model)
    if needs_supervisor:
        env.set_supervisor(supervisor_model)
    env.reset(seed=seed)
    try:
        for _ in range(STEPS):
            _, _, term, trunc, _ = env.step()
            if term or trunc:
                break
    except Exception as e:
        return None, f"episode crashed: {type(e).__name__}: {e}"
    return env, None


# ---------------------------------------------------------------------------
# Phase 2 -- fault injection correctness: onset range + duration actually
# enforced, for BOTH permanent and transient, across all 5 fault types.
# Uses recovery_mode="none" as the vehicle since injection RNG lives in the
# fault env's own __init__, independent of whichever recovery mixin sits on
# top of it -- no need to repeat this per recovery mechanism.
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("PHASE 2 -- fault injection correctness (onset range + duration enforcement)")
print("=" * 90)
for duration_mode in DURATION_MODES:
    print(f"\n  -- {duration_mode} --")
    for fault_name, cls in FAULT_CLASSES.items():
        onsets, durations_ok, active_trace_ok = [], True, True
        for seed in range(N_SEEDS_STANDARD):
            # Re-run manually here (rather than via run_episode()) so we can
            # inspect info["fault_active"] returned from EVERY step -- this
            # is a real, guaranteed-present signal (see env_fault_comms_loss.py's
            # step() return), unlike env.fault_context, which is a LOCAL
            # variable inside step() and is not readable from outside at all
            # (confirmed empirically -- see verify_oracle_invariance.py's Check D).
            env_class, sort_model, press_model, needs_supervisor, available, reason = get_env_and_models(fault_name, "none")
            if not available:
                flag(f"{fault_name}/{duration_mode}: episode unavailable -- {reason}")
                continue
            fault_config = make_fault_config(duration_mode, seed)
            env = env_class(max_steps=STEPS, seed=seed, fault_config=fault_config)
            env.set_agents(sort_agent=sort_model, press_agent=press_model)
            env.reset(seed=seed)
            last_active_step = None
            for step_i in range(STEPS):
                _, _, term, trunc, info = env.step()
                if info.get("fault_active"):
                    last_active_step = step_i
                if term or trunc:
                    break
            for ev in env.fault_log:
                onsets.append(ev["onset_step"])
                onset, dur = ev["onset_step"], ev.get("duration")
                if duration_mode == "transient" and dur is not None and last_active_step is not None:
                    if last_active_step > onset + dur + 2:
                        active_trace_ok = False
        onset_ok = all(40 <= o <= 60 for o in onsets) if onsets else False
        status_bits = []
        status_bits.append("onset OK" if onset_ok else "onset FAIL")
        if duration_mode == "transient":
            status_bits.append("duration-enforced OK" if active_trace_ok else "duration NOT enforced (still active past onset+duration)")
        print(f"    {fault_name:22s} onsets={onsets}  {' | '.join(status_bits)}")
        if not onset_ok:
            flag(f"{fault_name}/{duration_mode}: onset step outside configured 40-60 range")
        if duration_mode == "transient" and not active_trace_ok:
            flag(f"{fault_name}/transient: fault_context still shows active well past onset+duration -- duration may not be enforced")

# ---------------------------------------------------------------------------
# Phase 3 -- oracle (rule_based) invariants: false-positive rate ~0, and
# corrected degradation-area (lower = better) should not be worse than
# doing nothing. Both duration modes.
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("PHASE 3 -- oracle (rule_based) invariants")
print("=" * 90)
for duration_mode in DURATION_MODES:
    print(f"\n  -- {duration_mode} --")
    for fault_name in FAULT_CLASSES:
        fps, none_degradation, oracle_degradation = [], [], []
        for seed in range(N_SEEDS_STANDARD):
            env_o, err_o = run_episode(fault_name, "rule_based", duration_mode, seed)
            if env_o is None:
                flag(f"{fault_name}/{duration_mode}/rule_based: episode failed -- {err_o}")
                continue
            m_o = compute_recovery_metrics(env_o, fault_mode=fault_name, recovery_mode="rule_based", seed=seed)
            fp = m_o.get("intervention_precision", {}).get("overall_false_positive_rate")
            if fp is not None:
                fps.append(fp)
            for ev in m_o.get("fault_events", []):
                if ev.get("total_degradation_area") is not None:
                    oracle_degradation.append(ev["total_degradation_area"])

            env_n, err_n = run_episode(fault_name, "none", duration_mode, seed)
            if env_n is None:
                flag(f"{fault_name}/{duration_mode}/none: episode failed -- {err_n}")
                continue
            m_n = compute_recovery_metrics(env_n, fault_mode=fault_name, recovery_mode="none", seed=seed)
            for ev in m_n.get("fault_events", []):
                if ev.get("total_degradation_area") is not None:
                    none_degradation.append(ev["total_degradation_area"])

        bad_fp = [fp for fp in fps if fp > 1e-6]
        mean_none = np.mean(none_degradation) if none_degradation else float("nan")
        mean_oracle = np.mean(oracle_degradation) if oracle_degradation else float("nan")
        # Lower degradation-area is better - oracle shouldn't cause MORE
        # damage than doing nothing, within a small tolerance (matches
        # null_result_epsilon's convention elsewhere in the codebase).
        beats_none = mean_oracle <= mean_none + 2.0
        print(f"    {fault_name:22s} fp={fps}  none_degradation={mean_none:.2f}  "
              f"oracle_degradation={mean_oracle:.2f}  {'OK' if not bad_fp and beats_none else 'CHECK'}")
        if bad_fp:
            flag(f"{fault_name}/{duration_mode}: oracle has nonzero false-positive rate {bad_fp}")
        if not beats_none:
            # NOTE: this can be a genuine finding (a fixed heuristic applied on
            # every fault-active step isn't free -- it can cost more than it
            # saves against a mild fault) rather than a bug. Flag as a WARNING,
            # not a hard failure, and investigate before writing it up: check
            # whether the corrective action (e.g. check_container_level()) is
            # producing off-bale-multiple presses during the override window.
            flag(f"{fault_name}/{duration_mode}: oracle rule_based causes MORE degradation than "
                 f"no-recovery (none={mean_none:.2f} vs oracle={mean_oracle:.2f}) -- verify this is "
                 f"a real characteristic of the heuristic, not a bug, before reporting it", hard=False)

# ---------------------------------------------------------------------------
# Phase 4 -- rule_based_detect: informational precision/recall per fault type,
# permanent duration only (fast eyeball check that thresholds are still sane
# after any future retuning).
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("PHASE 4 -- rule_based_detect precision/recall (informational, permanent faults)")
print("=" * 90)
for fault_name in FAULT_CLASSES:
    recalls, precisions = [], []
    for seed in range(N_SEEDS_STANDARD):
        env, err = run_episode(fault_name, "rule_based_detect", "permanent", seed)
        if env is None:
            flag(f"{fault_name}/rule_based_detect: episode failed -- {err}")
            continue
        m = compute_recovery_metrics(env, fault_mode=fault_name, recovery_mode="rule_based_detect", seed=seed)
        r = m.get("intervention_recall", {}).get("overall_recall")
        p = m.get("intervention_precision", {})
        if r is not None:
            recalls.append(r)
    print(f"  {fault_name:22s} overall_recall={recalls}")
print("  (no pass/fail here -- just confirm these numbers still look like Fix 1's discussion:")
print("   near-zero recall is EXPECTED for fault types that don't really touch a given channel)")

# ---------------------------------------------------------------------------
# Phase 5 -- full matrix smoke test: every fault x every recovery mode x both
# duration modes actually runs without crashing, with sane (non-NaN) output.
# This is the "compare them all" pass -- prints a degradation-area grid
# (lower = better; raw, not null-corrected, since this smoke test doesn't
# thread a control run through) you can eyeball before committing to the
# full multi-seed run.
# ---------------------------------------------------------------------------
if RUN_FULL_MATRIX_SMOKE_TEST:
    print()
    print("=" * 90)
    print("PHASE 5 -- full matrix smoke test (all faults x all recovery modes x both durations)")
    print("=" * 90)
    for duration_mode in DURATION_MODES:
        print(f"\n  -- {duration_mode} --")
        header = f"    {'fault':22s}" + "".join(f"{r:>16s}" for r in RECOVERY_MODES)
        print(header)
        for fault_name in FAULT_CLASSES:
            row = f"    {fault_name:22s}"
            for recovery_mode in RECOVERY_MODES:
                if recovery_mode == "llm_replanning" and not RUN_LLM_CHECKS:
                    row += f"{'skipped':>16s}"
                    continue
                n_seeds = N_SEEDS_LLM if recovery_mode == "llm_replanning" else N_SEEDS_STANDARD
                scores = []
                for seed in range(n_seeds):
                    env, err = run_episode(fault_name, recovery_mode, duration_mode, seed)
                    if env is None:
                        if err and "not trained yet" in err or err and "not set" in err:
                            row += f"{'n/a':>16s}"
                        else:
                            flag(f"{fault_name}/{duration_mode}/{recovery_mode}: episode failed -- {err}")
                            row += f"{'CRASH':>16s}"
                        scores = None
                        break
                    m = compute_recovery_metrics(env, fault_mode=fault_name, recovery_mode=recovery_mode, seed=seed)
                    for ev in m.get("fault_events", []):
                        da = ev.get("total_degradation_area")
                        if da is None or not np.isfinite(da):
                            flag(f"{fault_name}/{duration_mode}/{recovery_mode}/seed{seed}: total_degradation_area is {da}")
                        else:
                            scores.append(da)
                    # Episode-length check
                    total_len = len(env.reward_data.get("Reward", []))
                    if total_len < STEPS:
                        warnings_.append(f"{fault_name}/{duration_mode}/{recovery_mode}/seed{seed}: "
                                          f"episode ended early at step {total_len}/{STEPS} (overflow/termination?)")
                if scores:
                    row += f"{np.mean(scores):>16.2f}"
                elif scores is not None:
                    row += f"{'--':>16s}"
            print(row)

# ---------------------------------------------------------------------------
# Phase 6 -- llm_replanning specific checks: error/fallback rate and latency,
# so you know whether the mechanism is actually working before trusting its
# numbers in the matrix (kept to N_SEEDS_LLM seeds to bound API cost).
# ---------------------------------------------------------------------------
if RUN_LLM_CHECKS:
    print()
    print("=" * 90)
    print("PHASE 6 -- llm_replanning error/fallback rate + latency (permanent faults)")
    print("=" * 90)
    for fault_name in FAULT_CLASSES:
        all_errors, all_latencies, total_calls = 0, [], 0
        for seed in range(N_SEEDS_LLM):
            env, err = run_episode(fault_name, "llm_replanning", "permanent", seed)
            if env is None:
                flag(f"{fault_name}/llm_replanning: episode failed -- {err}")
                continue
            log = getattr(env, "llm_decision_log", [])
            total_calls += len(log)
            all_errors += sum(1 for d in log if d.get("error"))
            all_latencies += [d["latency_ms"] for d in log if d.get("latency_ms") is not None]
        error_rate = all_errors / total_calls if total_calls else float("nan")
        mean_latency = np.mean(all_latencies) if all_latencies else float("nan")
        status = "OK" if total_calls and error_rate < 0.1 else "CHECK"
        print(f"  {fault_name:22s} calls={total_calls}  error_rate={error_rate:.2%}  "
              f"mean_latency={mean_latency:.0f}ms  {status}")
        if total_calls == 0:
            flag(f"{fault_name}/llm_replanning: zero LLM calls logged -- mechanism may not be invoking the API at all")
        elif error_rate >= 0.1:
            flag(f"{fault_name}/llm_replanning: {error_rate:.0%} of calls fell back on error -- check API key/base_url/model config")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 90)
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S) -- fix before trusting the full matrix:")
    for f in failures:
        print(f"  [FAIL] {f}")
else:
    print("RESULT: no hard failures.")
if warnings_:
    print(f"\n{len(warnings_)} WARNING(S) -- not necessarily bugs, but worth a manual look:")
    for w in warnings_:
        print(f"  [WARN] {w}")
if not failures and not warnings_:
    print("Clean run. Safe to proceed to the full seeded matrix.")
print("=" * 90)