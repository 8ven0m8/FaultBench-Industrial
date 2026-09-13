"""
Quick post-retrain sanity check for the fault-tolerant (FT) modular agents --
run this AFTER retraining (python main.py -> train -> fault_tolerant) and
BEFORE spending time on the full sweep.

All numbers below are raw total_reward (sort_reward + press_reward summed
over the episode) from directly running the env -- don't cross-reference
these against aggregate_benchmark_scores.py's output (degradation-area,
safety-violation excess, etc.), they're a different scale entirely.

Checks the specific gaps found in the pre-retrain models:
  1. Clean, fault-free episode: vanilla scored 107.95 vs FT's 81.96 (a
     ~26-point gap with ZERO fault present -- the main thing [64,64] +
     FAULT_PROB=0.40 is meant to close).
  2. actuator_degradation "restrict" mode, target=sort, permanent: FT
     scored 48.33 vs vanilla's 88.76 (a mode never seen in training before
     -- now added to ACTUATOR_MODE_CHOICES).
  3. agent_dropout, target=sort, permanent: FT scored 48.33 vs vanilla's
     88.76 too (same underlying cause as #1 -- the sort agent's own action
     is fully overridden by dropout, so any FT-vs-vanilla gap here comes
     from the OTHER (press) channel's general capability, not fault
     handling). Vanilla and FT each happen to land on the exact same
     number for "restrict" and "dropout" respectively -- confirmed this
     isn't a bug: under this seed, both policies' post-onset sort
     predictions settle onto a single repeated value, which makes
     "restrict" (force away from that value) collapse onto the same
     applied action sequence as "dropout" (fixed at a different value
     that the policy never chooses anyway). Coincidental to this seed, not
     a metrics artifact.

This does not replace the full sweep -- it's a fast, 4-episode check to
tell you whether retraining actually helped before you spend the time
running run_full_sweep.py's full grid.

Usage: python evaluation/check_ft_baseline.py (run from the repo root)
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# This file lives in evaluation/, one level below the repo root. Anchor
# both imports AND the relative "./models/..." paths below to the repo
# root via chdir - not just sys.path - so this works whether invoked from
# the repo root, from inside evaluation/, or from an IDE that defaults cwd
# to the script's own directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from stable_baselines3 import PPO

from src.envs_train.env_combined import Env_Combined
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault

STEPS = 200
SEED = 42

BASE_FAULT_CONFIG = {
    "injection_step_range": (40, 60),
    "seed": 123,
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


def load_pair(fault_tolerant):
    suffix = "FaultTolerant_" if fault_tolerant else ""
    sort_path = f"./models/PPO_Sorting_{suffix}NoMask_10000000.zip"
    press_path = f"./models/PPO_Pressing_{suffix}NoMask_10000000.zip"
    return PPO.load(sort_path), PPO.load(press_path)


def run_clean(sort_agent, press_agent, seed=SEED, steps=STEPS):
    env = Env_Combined(max_steps=steps, seed=seed)
    env.set_agents(sort_agent=sort_agent, press_agent=press_agent)
    env.reset(seed=seed)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step()
        if terminated or truncated:
            break
    return sum(env.reward_data["Reward"][i][0] + env.reward_data["Reward"][i][1]
               for i in range(len(env.reward_data["Reward"])))


def run_fault(env_class, sort_agent, press_agent, extra_cfg, seed=SEED, steps=STEPS):
    cfg = dict(BASE_FAULT_CONFIG)
    cfg.update(extra_cfg)
    env = env_class(max_steps=steps, seed=seed, fault_config=cfg)
    env.set_agents(sort_agent=sort_agent, press_agent=press_agent)
    env.reset(seed=seed)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step()
        if terminated or truncated:
            break
    return sum(env.reward_data["Reward"][i][0] + env.reward_data["Reward"][i][1]
               for i in range(len(env.reward_data["Reward"])))


def report(label, vanilla_score, ft_score, prev_vanilla, prev_ft):
    gap = vanilla_score - ft_score
    prev_gap = prev_vanilla - prev_ft
    # 1.0-point margin, not a tiny epsilon: float summation order can jitter
    # these sums by a few hundredths between runs even for the IDENTICAL
    # checkpoint, and this is a directional smoke test, not a precise diff.
    if gap < prev_gap - 1.0:
        verdict = "IMPROVED"
    elif gap > prev_gap + 1.0:
        verdict = "WORSE"
    else:
        verdict = "~UNCHANGED"
    print(f"\n{label}")
    print(f"  vanilla : {vanilla_score:8.2f}   (was {prev_vanilla:.2f})")
    print(f"  FT      : {ft_score:8.2f}   (was {prev_ft:.2f})")
    print(f"  gap     : {gap:8.2f}   (was {prev_gap:.2f})  -> {verdict}")


def main():
    print("Loading vanilla and fault-tolerant checkpoints...")
    sort_v, press_v = load_pair(fault_tolerant=False)
    sort_ft, press_ft = load_pair(fault_tolerant=True)

    clean_v = run_clean(sort_v, press_v)
    clean_ft = run_clean(sort_ft, press_ft)
    report("1) Clean, fault-free episode (no fault at all)",
           clean_v, clean_ft, prev_vanilla=107.9515, prev_ft=81.9553)

    restrict_v = run_fault(Env_ActuatorDegradationFault, sort_v, press_v,
                            {"mode": "restrict", "target": "sort", "duration": None, "duration_range": None})
    restrict_ft = run_fault(Env_ActuatorDegradationFault, sort_ft, press_ft,
                             {"mode": "restrict", "target": "sort", "duration": None, "duration_range": None})
    report("2) actuator_degradation 'restrict', target=sort, permanent (never trained on before)",
           restrict_v, restrict_ft, prev_vanilla=88.7587, prev_ft=48.3318)

    dropout_v = run_fault(Env_AgentDropoutFault, sort_v, press_v,
                           {"target": "sort", "duration": None, "duration_range": None})
    dropout_ft = run_fault(Env_AgentDropoutFault, sort_ft, press_ft,
                            {"target": "sort", "duration": None, "duration_range": None})
    report("3) agent_dropout, target=sort, permanent (sort action fully overridden either way -\n"
           "   any gap here is from the PRESS channel's general capability, not fault handling)",
           dropout_v, dropout_ft, prev_vanilla=88.7587, prev_ft=48.3318)

    print("\n" + "=" * 70)
    print("If gap #1 shrank a lot, [64,64] + FAULT_PROB=0.40 is working as intended.")
    print("If gap #2 shrank MORE than gap #3, the 'restrict' curriculum fix is")
    print("specifically responsible (not just general capacity).")
    print("These are single-seed checks -- treat as a directional smoke test,")
    print("not a substitute for the full multi-seed sweep + aggregate comparison.")
    print("=" * 70)


if __name__ == "__main__":
    main()
