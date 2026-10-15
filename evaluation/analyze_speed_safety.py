"""
RQ2: is there a trade-off between recovery speed and recovery safety?

Time-to-recovery is not used to compare mechanisms (it only exists for runs
that recover, and that subset differs between mechanisms). This script uses
a CENSORED time to recovery instead: steps from fault onset to the first
confirmed recovery, and for runs that never recover, the full remaining
window (200 - onset). Every run gets a value, so there is no recovered-only
selection bias.

Safety excess is computed exactly as in aggregate_benchmark_scores.py
(weighted violations minus the matched control's, floored at 0).

Reports:
  [1] run level, within each mechanism: Spearman(censored time, safety excess)
  [2] run level, within each fault (all mechanisms pooled)
  [3] cell level (fault x mechanism means): recovery rate, mean censored
      time, mean safety excess, and their Spearman correlation
  [4] reaction latency (onset -> first intervention on the targeted
      channel) for the three detecting mechanisms

A positive correlation in [1]/[2] means slower recovery comes with MORE
safety violations, i.e. no speed-safety trade-off.

Writes nothing; prints tables. Usage (from anywhere):
    python evaluation/analyze_speed_safety.py [--log log/recovery_metrics_sweep.jsonl]
"""

import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

# Same convention as run_full_sweep.py: resolve imports and relative paths
# against the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "evaluation"))
os.chdir(_REPO_ROOT)

from aggregate_benchmark_scores import _weighted_violations

EPISODE_STEPS = 200
FAULTS = ["sensor_noise", "actuator_degradation", "agent_dropout", "comms_loss", "byzantine"]
MECHANISMS = ["none", "rule_based", "rule_based_detect", "fault_tolerant_marl", "supervisor", "llm_replanning"]
DETECTING = ["rule_based_detect", "supervisor", "llm_replanning"]


def load_events(log_path):
    events = []
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("status") != "ok" or record.get("fault_mode") not in FAULTS:
                continue
            for event in record.get("fault_events", []):
                if event.get("total_degradation_area") is None:
                    continue  # same skip rule as aggregate_benchmark_scores.py
                window = EPISODE_STEPS - int(event["onset_step"])
                recovered = bool(event.get("recovered")) and event.get("time_to_first_recovery") is not None
                events.append({
                    "fault": record["fault_mode"],
                    "mechanism": record["recovery_mode"],
                    "censored_time": event["time_to_first_recovery"] if recovered else window,
                    "recovered": recovered,
                    "null": bool(event.get("likely_null_result")),
                    "safety_excess": max(0.0, _weighted_violations(event.get("safety"))
                                         - _weighted_violations(event.get("control_safety"))),
                    "reaction_latency": event.get("reaction_latency"),
                })
    return events


def spearman(rows):
    rho, p = spearmanr([r["censored_time"] for r in rows], [r["safety_excess"] for r in rows])
    return rho, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="log/recovery_metrics_sweep.jsonl")
    args = parser.parse_args()

    events = load_events(args.log)
    print(f"fault events: {len(events)}")

    print("\n[1] Run level, within each mechanism: Spearman(censored recovery time, safety excess)")
    for m in MECHANISMS:
        rows = [e for e in events if e["mechanism"] == m]
        non_null = [e for e in rows if not e["null"]]
        rho, p = spearman(rows)
        rho_nn, p_nn = spearman(non_null)
        print(f"  {m:20s} n={len(rows):4d} rho={rho:+.2f} (p={p:.1e}) | "
              f"non-null only: n={len(non_null):4d} rho={rho_nn:+.2f} (p={p_nn:.1e})")

    print("\n[2] Run level, within each fault (all mechanisms pooled)")
    for f in FAULTS:
        rows = [e for e in events if e["fault"] == f]
        rho, p = spearman(rows)
        print(f"  {f:22s} n={len(rows):4d} rho={rho:+.2f} (p={p:.1e})")

    print("\n[3] Cell level (fault x mechanism means)")
    cells = collections.defaultdict(list)
    for e in events:
        cells[(e["fault"], e["mechanism"])].append(e)
    mean_time, mean_safety, recovery_rate = [], [], []
    for f in FAULTS:
        for m in MECHANISMS:
            rows = cells[(f, m)]
            recovery_rate.append(np.mean([e["recovered"] for e in rows]))
            mean_time.append(np.mean([e["censored_time"] for e in rows]))
            mean_safety.append(np.mean([e["safety_excess"] for e in rows]))
            print(f"  {f:22s} {m:20s} recovered {recovery_rate[-1]:6.1%}  "
                  f"mean censored time {mean_time[-1]:6.1f}  safety excess {mean_safety[-1]:7.1f}")
    rho, p = spearmanr(mean_time, mean_safety)
    print(f"  {len(mean_time)} cells: Spearman(mean censored time, safety excess) rho={rho:+.2f} (p={p:.1e})")
    rho, p = spearmanr(recovery_rate, mean_safety)
    print(f"  {len(mean_time)} cells: Spearman(recovery rate, safety excess)      rho={rho:+.2f} (p={p:.1e})")

    print("\n[4] Reaction latency (onset -> first intervention on the targeted channel)")
    for m in DETECTING:
        for f in FAULTS:
            rows = cells[(f, m)]
            lat = [e["reaction_latency"] for e in rows if e["reaction_latency"] is not None]
            if lat:
                print(f"  {m:18s} {f:22s} reacted {len(lat):3d}/{len(rows):3d}  "
                      f"median {np.median(lat):5.1f} steps  share at 0: {np.mean([x == 0 for x in lat]):.0%}")
            else:
                print(f"  {m:18s} {f:22s} reacted   0/{len(rows):3d}")


if __name__ == "__main__":
    main()
