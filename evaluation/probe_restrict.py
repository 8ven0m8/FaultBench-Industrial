"""
Probe: why fault_tolerant_marl does not recover from actuator "restrict".

Under restrict, one press (press 2 in the sweep) is disabled: any command
routed to it is downgraded to a no-op. The press policy only observes
container levels, sorter contents and the two press timers, so a press
that never starts a job keeps a timer of 0 and looks idle - exactly like a
healthy free press. The PPO policies are feed-forward (MlpPolicy, no frame
stacking), so they cannot notice that their commands have no effect.

This script re-runs the press-policy under permanent restrict on the press
and both targets, with the same 8 evaluation seeds and fault config as
run_full_sweep.py (--seed 42), and reports, before and after onset:
  - the share of press commands sent to the disabled press
  - the share of steps in which each press timer is 0 (looks idle)
for the vanilla pair (used by `none`) and each fault-tolerant replicate.

Writes nothing; prints a table. Usage (from anywhere):
    python evaluation/probe_restrict.py
"""

import collections
import os
import sys

import numpy as np

# Same convention as run_full_sweep.py: resolve imports and relative model
# paths against the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from stable_baselines3 import PPO
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.model_paths import VANILLA_SORT_PATH, VANILLA_PRESS_PATH, find_ft_pair

BASE_SEED = 42          # run_full_sweep.py default --seed
N_SEEDS = 8             # run_full_sweep.py default --seeds
STEPS = 200
BROKEN_PRESS = 2        # run_full_sweep.py BASE_FAULT_CONFIG["restrict_press_id"]
TARGETS = ("press", "both")
TRAIN_SEEDS = (1, 2, 3)


def derive_trial_seeds(base_seed, seed_index):
    """Identical to run_full_sweep.derive_trial_seeds (not imported, since
    that module parses CLI args and changes global state on import)."""
    trial_rng = np.random.default_rng((int(base_seed), int(seed_index)))
    episode_seed = int(trial_rng.integers(0, 2**31 - 1))
    fault_seed = int(trial_rng.integers(0, 2**31 - 1))
    return episode_seed, fault_seed


class PressRecorder:
    """Wraps the press model and logs (intended action, press1 timer,
    press2 timer) for every step. Timers are obs[14] and obs[15] of the
    16-dim press observation (see Env_Super.get_press_obs)."""

    def __init__(self, model):
        self.model = model
        self.log = []

    def predict(self, obs, deterministic=True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        self.log.append((int(action), float(obs[14]), float(obs[15])))
        return action, state


def press_id(action):
    # Discrete(11): 0 = no-op, 1-5 = press 1, 6-10 = press 2
    return 0 if action == 0 else (1 if action <= 5 else 2)


def probe(sort_model, press_model):
    counts = collections.Counter()
    for target in TARGETS:
        for i in range(N_SEEDS):
            episode_seed, fault_seed = derive_trial_seeds(BASE_SEED, i)
            fault_config = {
                "injection_step_range": (40, 60), "seed": fault_seed, "target": target,
                "duration": None, "duration_range": None, "mode": "restrict",
                "restrict_sort_mode": 1, "restrict_press_id": BROKEN_PRESS,
                "stuck_press_action": 0, "stuck_sort_mode": None, "degradation_prob": 0.3,
            }
            env = Env_ActuatorDegradationFault(max_steps=STEPS, seed=episode_seed, fault_config=fault_config)
            recorder = PressRecorder(press_model)
            env.set_agents(sort_agent=sort_model, press_agent=recorder)
            env.reset(seed=episode_seed)
            for _ in range(STEPS):
                _, _, done, _, _ = env.step(action=None, mode="model")
                if done:
                    break

            onset = env._injection_step
            for t, (action, timer1, timer2) in enumerate(recorder.log):
                phase = "post" if t >= onset else "pre"
                counts[(phase, "steps")] += 1
                counts[(phase, f"cmd{press_id(action)}")] += 1
                counts[(phase, "p1_idle")] += timer1 == 0
                counts[(phase, "p2_idle")] += timer2 == 0
    return counts


def main():
    pairs = {"vanilla (none)": (VANILLA_SORT_PATH, VANILLA_PRESS_PATH)}
    for ts in TRAIN_SEEDS:
        pair = find_ft_pair(ts)
        if pair:
            pairs[f"FT ts{ts}"] = pair

    print(f"Permanent restrict, press {BROKEN_PRESS} disabled, targets {TARGETS}, {N_SEEDS} seeds\n")
    for name, (sort_path, press_path) in pairs.items():
        counts = probe(PPO.load(sort_path), PPO.load(press_path))
        for phase in ("pre", "post"):
            n = counts[(phase, "steps")]
            commands = counts[(phase, "cmd1")] + counts[(phase, "cmd2")]
            print(f"{name:15s} {phase:4s}: "
                  f"to disabled press {counts[(phase, f'cmd{BROKEN_PRESS}')] / max(commands, 1):6.1%} "
                  f"of {commands:4d} press commands | "
                  f"timer==0: press1 {counts[(phase, 'p1_idle')] / n:6.1%}, "
                  f"press2 {counts[(phase, 'p2_idle')] / n:6.1%}")


if __name__ == "__main__":
    main()
