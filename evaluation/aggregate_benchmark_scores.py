"""
Multi-seed aggregation for the recovery-mechanism sweep.

Why this exists: a single run of run_full_sweep.py (--seeds 1) is one
dice-roll per combo - a mechanism can look great because it genuinely
recovered well, or because that one fault instance happened to be too weak
to measure, or from pure seed luck. There's no way to tell those apart from
one run. This script instead groups every combo's N seed realizations
together and reports MEANS with confidence intervals - the same reason a
GPU benchmark reports average FPS over many frames/runs instead of one
frame's timing.

There used to also be a composite 0-100 "benchmark_score" reported here
(and in utils/metrics.py). Removed: it blended safety/deficit/relapse/ttr
into one number using fixed, somewhat arbitrary weights, saturated at 100
on a large fraction of runs (losing discriminative power exactly where it
mattered), and collapsed the speed-vs-safety trade-off the research
questions are actually about into a single figure. Compare mechanisms on
the four primitive metrics below directly instead.

Input: a JSONL file where each line is one run's metrics, as produced by
run_full_sweep.py --seeds N (each realization's run_id carries a __seedK
suffix; see combo_run_id() there).

Output:
  1. aggregate_detailed.csv  - one row per (fault_mode, target, duration_type,
     extra_mode_key/value, recovery_mode), aggregated across seeds.
  2. aggregate_summary.csv   - one row per (fault_mode, recovery_mode), pooled
     across every target/duration/mode combo for that fault type (the
     "headline" table).
  3. aggregate_faceted__<fault>__<metric>.png   - one chart per
     (fault_mode, metric) with target (sort/press/both) as rows and
     duration_type (transient/permanent) as columns, recovery_mode bars
     within each cell. Up to 6 metrics x 5 fault types = up to 30 files
     (fewer for metrics with no data for a given fault type, e.g.
     intervention_recall needs at least one detector-based run).
  4. effect_sizes_detailed.csv / effect_sizes_summary.csv - Glass's delta
     for every recovery_mode against the recovery_mode == "none" (no
     recovery) control, at the same two granularities as 1/2 above. See
     compute_effect_sizes() for the definition and why it's Glass's delta
     specifically (not Cohen's d/Hedges' g).

There used to also be a pooled "headline" chart per fault_mode (one bar
per recovery_mode, averaged across every target/duration/mode combo).
Removed: pooling across target/duration hid real, large effects (e.g.
sensor_noise/supervisor averaged ~0.5 corrected degradation-area on
target=sort but ~50x that on target=press - a single pooled bar blends
those into one misleading middling number). aggregate_summary.csv still
has the pooled numbers if you want them for a quick reference, but the
faceted charts are the ones safe to draw conclusions from visually.

Metrics reported, per group (direction noted since it's not all
lower-is-better - see plot_faceted_by_target_duration()'s docstring):
  n                        : number of seed realizations found for this group
  null_result_rate         : fraction of those realizations where the fault's
                              effect was statistically indistinguishable from
                              the no-fault control episode (likely_null_result).
                              Diagnostic on fault-suite severity, not filtered
                              out of the other stats (with enough seeds, null
                              flukes wash out in the average on their own).
  degradation_area_mean/ci : [LOWER better] mean/95%-CI of
                              total_degradation_area_corrected.
  safety_excess_mean/ci    : [LOWER better] mean/95%-CI of a weighted
                              excess-safety-violation count (5*catastrophic +
                              2*severe + 1*near_miss, fault run minus the
                              matched no-fault control run, floored at 0) -
                              see WEIGHTS below.
  fraction_time_healthy_mean/ci : [HIGHER better] mean/95%-CI of the fraction
                              of the fault-active window spent in a healthy
                              state, per event - computed for EVERY event
                              (including null results, where it's correctly
                              ~1.0), unlike the old time-to-recovery metric,
                              so it doesn't suffer the same selection-bias
                              problem (see below).
  relapse_count_mean/ci    : [LOWER better] mean/95%-CI of how many times an
                              event's reward dropped back out of "healthy"
                              after first recovering - a mechanism that
                              recovers once and then flickers scores worse
                              here even if its degradation-area looks fine.
  false_recovery_rate_mean/ci : [LOWER better] mean/95%-CI of the run-level
                              overall false-recovery rate (from
                              utils/metrics.py's _intervention_scoring,
                              "none"/oracle rule_based runs have no detector
                              so this is None for them - rule_based being an
                              oracle rather than "no detector" is why it
                              still shows 0.0, not None: see
                              RuleBasedRecoveryMixin's docstring).
  intervention_recall_mean/ci : [HIGHER better] mean/95%-CI of the run-level
                              overall recall - of the steps where a fault was
                              genuinely active, what fraction did the
                              mechanism actually flag? Complements
                              false_recovery_rate (that's the false-positive
                              side; this is the false-negative side). None
                              for "none"/fault_tolerant_marl (no detector).

There used to also be a time-to-recovery metric here. Removed: it's only
meaningful over events that both recovered AND weren't a null result, and
that qualifying subset's SIZE and COMPOSITION differs wildly by mechanism -
a mechanism that fully neutralizes a fault (turning it into a null result)
gets that event excluded entirely, while a mechanism that never touches the
fault has a bigger, easier-composed pool to average over. Comparing a
17-sample average against a 5-sample average of a harder subset produced
misleading "faster recovery" numbers for mechanisms that were doing less,
not more. See utils/metrics.py's time_to_first_recovery field if you want
it for inspecting one specific run - just don't aggregate it across
mechanisms as a comparison metric.

Usage (run from the repo root):
    python evaluation/aggregate_benchmark_scores.py [path_to_jsonl] [--include-baseline]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This file lives in evaluation/, one level below the repo root. Anchor
# the default relative paths below (./log/..., ./img/...) to the repo
# root via chdir, so this works whether invoked from the repo root, from
# inside evaluation/, or from an IDE that defaults cwd to the script's own
# directory.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same 5/2/1 severity weighting the old benchmark_score used for its safety
# term - kept here as a standalone, reported metric rather than buried
# inside a composite score.
WEIGHTS = {"catastrophic_count": 5, "severe_count": 2, "near_miss_count": 1}

PREFERRED_RECOVERY_ORDER = [
    "none", "rule_based", "rule_based_detect", "supervisor", "fault_tolerant_marl", "llm_replanning",
]


def _ordered_recovery_modes(modes) -> list:
    modes = list(modes)
    return [m for m in PREFERRED_RECOVERY_ORDER if m in modes] + [
        m for m in modes if m not in PREFERRED_RECOVERY_ORDER
    ]


def _weighted_violations(safety: dict) -> float:
    if not safety:
        return 0.0
    return sum(WEIGHTS[k] * safety.get(k, 0) for k in WEIGHTS)


def _extra_mode(record: dict):
    """Whichever fault-specific categorical mode key run_full_sweep.py stored
    on this record ("mode" for actuator_degradation, "comms_mode" for
    comms_loss, "byzantine_mode" for byzantine), or (None, None) for fault
    types with no extra axis (sensor_noise, agent_dropout)."""
    for key in ("mode", "comms_mode", "byzantine_mode"):
        if key in record:
            return key, record[key]
    return None, None


def load_rows(jsonl_path: Path, exclude_baseline: bool = True) -> pd.DataFrame:
    """Flatten the sweep JSONL into one row per fault event, keeping enough
    fields to both group by combo identity (ignoring the seed) and compute
    every aggregated metric described in the module docstring."""
    rows = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_num}: {e}", file=sys.stderr)
                continue

            fault_mode = record.get("fault_mode")
            recovery_mode = record.get("recovery_mode")
            if record.get("status") != "ok":
                continue
            if exclude_baseline and fault_mode == "none":
                continue

            extra_mode_key, extra_mode_value = _extra_mode(record)
            overall_fpr = ((record.get("false_recovery_rate") or {}).get("overall"))
            overall_recall = ((record.get("intervention_recall") or {}).get("overall_recall"))

            for event in record.get("fault_events", []):
                if event.get("total_degradation_area") is None:
                    continue  # onset never reached / empty series - see _score_fault_event

                fault_safety = _weighted_violations(event.get("safety"))
                control_safety = _weighted_violations(event.get("control_safety"))
                safety_excess = max(0.0, fault_safety - control_safety)

                likely_null = bool(event.get("likely_null_result"))

                rows.append({
                    "fault_mode": fault_mode,
                    "recovery_mode": recovery_mode,
                    "target": event.get("target") or record.get("target"),
                    "duration_type": record.get("duration_type"),
                    "extra_mode_key": extra_mode_key,
                    "extra_mode_value": extra_mode_value,
                    "run_id": record.get("run_id"),
                    "episode_seed": record.get("episode_seed"),
                    "likely_null_result": likely_null,
                    "degradation_area_corrected": event.get("total_degradation_area_corrected"),
                    "safety_excess": safety_excess,
                    "false_recovery_rate": overall_fpr,
                    "intervention_recall": overall_recall,
                    "fraction_time_healthy": event.get("fraction_time_healthy"),
                    "relapse_count": event.get("relapse_count"),
                })

    return pd.DataFrame(rows)


def _mean_ci(series: pd.Series):
    """Mean and a 95% CI half-width (normal approximation: 1.96 * sem).
    With small n (few seeds) this is an approximation, not exact - report n
    alongside it and treat CIs as indicative, not a rigorous test. Returns
    (mean, ci_halfwidth, n) with NaN-safe handling for empty/all-NaN input."""
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = clean.mean()
    if n < 2:
        return mean, float("nan"), n
    sem = clean.std(ddof=1) / math.sqrt(n)
    return mean, 1.96 * sem, n


def aggregate(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    records = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["n"] = len(group)
        row["null_result_rate"] = round(group["likely_null_result"].mean(), 4)

        for metric, out_prefix in [
            ("degradation_area_corrected", "degradation_area"),
            ("safety_excess", "safety_excess"),
            ("fraction_time_healthy", "fraction_time_healthy"),
            ("relapse_count", "relapse_count"),
        ]:
            mean, ci, n_used = _mean_ci(group[metric])
            row[f"{out_prefix}_mean"] = round(mean, 4) if not math.isnan(mean) else None
            row[f"{out_prefix}_ci95"] = round(ci, 4) if not math.isnan(ci) else None

        for metric, out_prefix in [
            ("false_recovery_rate", "false_recovery_rate"),
            ("intervention_recall", "intervention_recall"),
        ]:
            mean, ci, n_used = _mean_ci(group[metric])
            row[f"{out_prefix}_mean"] = round(mean, 4) if not math.isnan(mean) else None
            row[f"{out_prefix}_ci95"] = round(ci, 4) if not math.isnan(ci) else None

        records.append(row)

    return pd.DataFrame(records)


def compute_effect_sizes(df: pd.DataFrame, cell_cols: list, control_mode: str = "none") -> pd.DataFrame:
    """Glass's delta for every recovery_mode against the recovery_mode ==
    control_mode ("none", i.e. no recovery mechanism at all - the fault
    still happens) within each cell defined by cell_cols minus
    "recovery_mode" itself:

        delta = (treatment_mean - control_mean) / control_std

    This is specifically Glass's delta rather than Cohen's d/Hedges' g
    because the denominator uses ONLY the control group's std, not a
    pooled std across both groups. That's the right choice here: a
    recovery mechanism that actually works can itself change the metric's
    variance (e.g. squashing it toward a tight near-zero band), so
    "control std" is the more stable, meaningful yardstick, and it's what
    the DT-MARL supply-chain paper (cited in research-proposal.txt as the
    reporting-bar reference) uses for the same reason.

    Scoped to the four metrics where "none" always has a value
    (degradation_area_corrected, safety_excess, fraction_time_healthy,
    relapse_count). false_recovery_rate/intervention_recall are
    detector-quality metrics - "none" runs no detector at all, so there is
    no meaningful "no-recovery" reference value for those two.

    A cell with fewer than 2 control samples, or a control std of exactly
    0, can't produce a delta - those cells get None rather than a
    divide-by-zero or single-point "variance".

    |delta| conventional magnitude bands (Cohen, 1988): ~0.2 small, ~0.5
    medium, ~0.8 large. These are rules of thumb for interpretation, not a
    significance test - report them alongside n, not instead of it.
    """
    metrics = ["degradation_area_corrected", "safety_excess", "fraction_time_healthy", "relapse_count"]
    group_keys = [c for c in cell_cols if c != "recovery_mode"]
    records = []
    for keys, cell in df.groupby(group_keys, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        control = cell[cell["recovery_mode"] == control_mode]
        if control.empty:
            continue
        for recovery_mode, treatment in cell.groupby("recovery_mode"):
            if recovery_mode == control_mode:
                continue
            row = dict(zip(group_keys, keys))
            row["recovery_mode"] = recovery_mode
            row["n_control"] = len(control)
            row["n_treatment"] = len(treatment)
            for metric in metrics:
                c = control[metric].dropna()
                t = treatment[metric].dropna()
                control_std = c.std(ddof=1) if len(c) >= 2 else float("nan")
                if len(t) < 1 or math.isnan(control_std) or control_std == 0:
                    row[f"glass_delta_{metric}"] = None
                else:
                    row[f"glass_delta_{metric}"] = round(float((t.mean() - c.mean()) / control_std), 3)
            records.append(row)

    return pd.DataFrame(records)


def plot_faceted_by_target_duration(df: pd.DataFrame, outdir: Path) -> list:
    """One chart per (fault_mode, metric): a grid of subplots with target
    (sort/press/both) as rows and duration_type (transient/permanent) as
    columns, recovery_mode bars within each cell. Pools across the
    fault-specific "mode" axis (e.g. actuator_degradation's
    stuck/restrict/slip) to keep the grid readable; target and duration
    are the two axes that turned out to produce the largest swings (e.g.
    sensor_noise/supervisor: target=sort averaged ~0.5 corrected
    degradation-area, target=press averaged ~50x that - invisible in a
    pooled average).

    Every subplot title says which direction is good for that metric -
    they're not all "lower is better": fraction_time_healthy and
    intervention_recall are the two where a TALLER bar is the good
    outcome, everything else is shorter-is-better. Mixing that up is an
    easy mistake since every other chart in this project used to be
    higher-is-better (the old benchmark_score)."""
    saved = []
    grouped = aggregate(df, ["fault_mode", "target", "duration_type", "recovery_mode"])

    # (mean_col, ci_col, label, higher_is_better)
    metrics = [
        ("degradation_area_mean", "degradation_area_ci95", "Corrected degradation-area", False),
        ("safety_excess_mean", "safety_excess_ci95", "Weighted safety-violation excess", False),
        ("fraction_time_healthy_mean", "fraction_time_healthy_ci95", "Fraction of window spent healthy", True),
        ("relapse_count_mean", "relapse_count_ci95", "Relapse count", False),
        ("false_recovery_rate_mean", "false_recovery_rate_ci95", "False-recovery rate", False),
        ("intervention_recall_mean", "intervention_recall_ci95", "Intervention recall", True),
    ]

    for fault_mode in sorted(grouped["fault_mode"].unique()):
        fm_df = grouped[grouped["fault_mode"] == fault_mode]
        targets = sorted(fm_df["target"].dropna().unique())
        durations = sorted(fm_df["duration_type"].dropna().unique())
        if not targets or not durations:
            continue

        for mean_col, ci_col, label, higher_is_better in metrics:
            if fm_df[mean_col].dropna().empty:
                continue  # e.g. intervention_recall for a fault type with no detector-based runs found

            color = "steelblue" if higher_is_better else "seagreen"
            direction = "HIGHER is better" if higher_is_better else "LOWER is better"

            fig, axes = plt.subplots(
                len(targets), len(durations),
                figsize=(5.5 * len(durations), 4 * len(targets)),
                squeeze=False,
            )
            for i, target in enumerate(targets):
                for j, duration in enumerate(durations):
                    ax = axes[i][j]
                    cell = fm_df[(fm_df["target"] == target) & (fm_df["duration_type"] == duration)]
                    if cell.empty:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"target={target}, {duration}", fontsize=9)
                        continue

                    order = _ordered_recovery_modes(cell["recovery_mode"].unique())
                    cell = cell.set_index("recovery_mode").reindex(order).reset_index()

                    x = np.arange(len(cell))
                    means = cell[mean_col].astype(float)
                    cis = cell[ci_col].astype(float).fillna(0)
                    ax.bar(x, means, yerr=cis, capsize=3, color=color, edgecolor="black", zorder=2)
                    ax.set_xticks(x)
                    ax.set_xticklabels(cell["recovery_mode"], rotation=35, ha="right", fontsize=7)
                    ax.set_title(f"target={target}, {duration} (n={cell['n'].iloc[0]})", fontsize=9)
                    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
                    ax.set_axisbelow(True)

            fig.suptitle(
                f"{label} by target x duration — fault: {fault_mode}  [{direction}]\n"
                f"(pooled across fault-specific modes, e.g. stuck/restrict/slip, where applicable)",
                fontsize=11,
            )
            fig.tight_layout()

            metric_slug = mean_col.replace("_mean", "")
            out_path = outdir / f"aggregate_faceted__{fault_mode}__{metric_slug}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved.append(out_path)
            print(f"Saved {out_path}")

    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl_path", nargs="?", default="./log/recovery_metrics_sweep.jsonl")
    parser.add_argument("--include-baseline", action="store_true",
                         help="Include fault_mode == 'none' baseline runs (no fault_events, excluded by default).")
    parser.add_argument("--outdir", default="./img/figures/benchmarks")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_rows(jsonl_path, exclude_baseline=not args.include_baseline)
    if df.empty:
        print("No fault-event data found — check the input file.", file=sys.stderr)
        sys.exit(1)

    detail = aggregate(df, ["fault_mode", "target", "duration_type", "extra_mode_key", "extra_mode_value", "recovery_mode"])
    summary = aggregate(df, ["fault_mode", "recovery_mode"])

    pd.set_option("display.width", 160)
    print("\n=== Headline summary: mean ± 95% CI by fault_mode x recovery_mode (pooled across all combos) ===")
    print(summary.to_string(index=False))

    detail_path = outdir / "aggregate_detailed.csv"
    summary_path = outdir / "aggregate_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved detailed table to {detail_path}")
    print(f"Saved summary table to {summary_path}")

    effect_detail = compute_effect_sizes(
        df, ["fault_mode", "target", "duration_type", "extra_mode_key", "extra_mode_value", "recovery_mode"]
    )
    effect_summary = compute_effect_sizes(df, ["fault_mode", "recovery_mode"])

    print("\n=== Glass's delta vs. recovery_mode=='none', pooled across all combos "
          "(negative = improvement on lower-is-better metrics) ===")
    print(effect_summary.to_string(index=False))

    effect_detail_path = outdir / "effect_sizes_detailed.csv"
    effect_summary_path = outdir / "effect_sizes_summary.csv"
    effect_detail.to_csv(effect_detail_path, index=False)
    effect_summary.to_csv(effect_summary_path, index=False)
    print(f"\nSaved detailed effect sizes to {effect_detail_path}")
    print(f"Saved summary effect sizes to {effect_summary_path}")

    plot_faceted_by_target_duration(df, outdir)


if __name__ == "__main__":
    main()
