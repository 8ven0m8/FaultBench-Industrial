# ---------------------------------------------------------*\
# Title: Multi-Training-Seed Driver (fault_tolerant_marl + supervisor)
# ---------------------------------------------------------*/
#
# The 8-seed sweep only randomizes EVALUATION seeds (episode world + fault
# injection). fault_tolerant_marl and supervisor are themselves trained
# policies, so a single training run can't separate the paradigm's effect
# from that one run's training variance. This script trains independent
# replicates, one per training seed, each saved under its own
# "_ts<seed>" checkpoint name (see src/model_paths.py). run_full_sweep.py
# then evaluates every replicate it finds, and
# aggregate_benchmark_scores.py reports the across-training-seed spread.
#
# Why a separate script instead of main.py: main.py asks its questions via
# input() at import time and has no notion of a training seed. This calls
# the same training functions main.py does, non-interactively.
#
# Replicates are independent, so different seeds (and the two variants)
# can be run in parallel in separate terminals - each replicate writes its
# own best-model directory and its own checkpoint name.
#
# Usage (from the repo root):
#   python train_seeds.py --variant fault_tolerant --train-seeds 1,2,3
#   python train_seeds.py --variant supervisor     --train-seeds 1,2,3
#
# The vanilla Sorting/Pressing pair is NOT retrained here - every other
# mechanism (none, rule_based, rule_based_detect, supervisor, llm_replanning)
# runs on that single existing pair, and the supervisor is trained on top of it.

import argparse
import os
import sys
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

# Same budgets main.py uses.
TOTAL_TIMESTEPS = 10_000_000
SUPERVISOR_TOTAL_TIMESTEPS = 5_000_000
STEPS_TRAIN = 200
STEPS_TEST = 200
SEED = 42  # only used for the post-training test plots, same as main.py


def main():
    p = argparse.ArgumentParser(description="Train independent training-seed replicates.")
    p.add_argument("--variant", required=True, choices=["fault_tolerant", "supervisor"])
    p.add_argument("--train-seeds", required=True,
                   help="Comma list of integer training seeds, e.g. 1,2,3.")
    p.add_argument("--timesteps", type=int, default=None,
                   help="Override the training budget (default: 10M for fault_tolerant, "
                        "5M for supervisor). Only for smoke tests - the sweep's results "
                        "assume the default budgets.")
    args = p.parse_args()

    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    for train_seed in train_seeds:
        tag = f"Gold_{datetime.now().strftime('%d-%m-%Y_%H-%M')}_NoMask"
        print(f"\n{'=' * 70}\nTraining {args.variant} replicate, train_seed={train_seed}\n{'=' * 70}")

        if args.variant == "fault_tolerant":
            from src.training_fault_tolerant import train_fault_tolerant_modular_agents
            train_fault_tolerant_modular_agents(
                total_timesteps=args.timesteps or TOTAL_TIMESTEPS,
                steps_train=STEPS_TRAIN, steps_test=STEPS_TEST,
                seed=SEED, tag=tag, train_seed=train_seed,
            )
        else:
            from src.training_supervisor import train_supervisor
            model = train_supervisor(
                total_timesteps=args.timesteps or SUPERVISOR_TOTAL_TIMESTEPS,
                steps_train=STEPS_TRAIN, steps_test=STEPS_TEST,
                seed=SEED, tag=tag, train_seed=train_seed,
            )
            if model is None:
                sys.exit(1)  # vanilla pair missing - train_supervisor already printed why


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------*/
