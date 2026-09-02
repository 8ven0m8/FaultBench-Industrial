# ---------------------------------------------------------*\
# Title: Recovery Metrics
# ---------------------------------------------------------*/
#
# Post-hoc scoring for the recovery metrics used by the project:
#   -> Time-to-recovery (now time-to-first-recovery)
#   -> Degradation-area (now total-degradation-area / cumulative-reward-deficit)
#   -> Intervention precision / recall / F1 (formerly false-recovery rate)
#   -> Safety-violation count (fill-level + quality/purity tiers)
#   -> Per-channel reward scoring
#   -> Composite benchmark score
#
# IMPORTANT SEMANTIC DISTINCTION
# --------------------------------
# The fault injector and the recovery mechanism answer different questions:
#
#   fault onset       = when the fault becomes active
#   detection         = when the mechanism flags/recognises the fault
#   recovery          = when observed performance has demonstrably returned
#                        to the pre-fault baseline and remains there
#
# This module measures PERFORMANCE recovery. It does not assume that the
# underlying fault has physically disappeared. That distinction is important
# for persistent faults such as sensor_noise with duration=None.
#
# In particular, recovery is NOT allowed to be declared at the first fault
# step merely because a short post-onset reward window happens to be healthy.
# Recovery is confirmed only after:
#   1. a complete healthy run is available, and
#   2. the required number of consecutive healthy steps satisfy the recovery
#      threshold.
#
# The reported first_recovery_step is the END of the final confirming window.
# This makes time_to_first_recovery an actual latency to confirmed recovery
# rather than the start of a window that has not yet been observed in full.
#
# The safety metric is deliberately independent of reward recovery. Safety is
# evaluated over the complete fault-active interval, not stopped when reward
# happens to cross the recovery threshold.
#
# Expected environment logs:
#   env.reward_data     - per-step reward/fill histories
#   env.fault_log       - [{onset_step, type, target, duration?}, ...]
#   env.detection_log   - per-step detection/intervention state for recovery
#                         mechanisms that expose it
#
# Usage (see main.py):
#   from utils.metrics import (
#       compute_recovery_metrics,
#       print_recovery_metrics,
#       save_recovery_metrics,
#   )
#   metrics = compute_recovery_metrics(...)
#   print_recovery_metrics(metrics)
#   save_recovery_metrics(metrics)

import json
import math
import os
import time


DEFAULT_LOG_PATH = "log/recovery_metrics.jsonl"


# ---------------------------------------------------------*/
# Reward series
# ---------------------------------------------------------
def _channel_reward_series(reward_data):
    """Return per-channel reward series: (sort_series, press_series)."""
    rewards = reward_data.get("Reward", []) or []
    sort_series = []
    press_series = []
    for item in rewards:
        try:
            r_sort, r_press = item
            sort_series.append(float(r_sort))
            press_series.append(float(r_press))
        except (TypeError, ValueError, IndexError):
            # Ignore malformed reward entries rather than crashing the
            # evaluator. A malformed entry cannot contribute a meaningful
            # recovery score.
            continue
    return sort_series, press_series


def _total_reward_series(reward_data):
    """Return total reward per environment step."""
    sort_series, press_series = _channel_reward_series(reward_data)
    return [s + p for s, p in zip(sort_series, press_series)]


# ---------------------------------------------------------*/
# Single-channel recovery scoring
# ---------------------------------------------------------
def _score_single_channel(
    series,
    onset_idx,
    fault_window_end,
    baseline,
    tolerance_frac,
    required_healthy_steps,
):
    """Score performance recovery for a single reward channel.

    Tracks every sustained recovery and relapse in the window, and computes
    window-complete (non-truncated) degradation area.
    """
    n = len(series)
    if n == 0 or onset_idx >= n:
        return {
            "baseline_reward": 0.0,
            "recovery_threshold": 0.0,
            "degradation_detected": False,
            "degradation_start_step": None,
            "recovered": False,
            "first_recovery_idx": None,
            "first_recovery_step": None,
            "time_to_first_recovery": None,
            "sustained_recovery": False,
            "recovery_events": [],
            "relapse_count": 0,
            "time_in_recovered_state": 0,
            "fraction_time_healthy": None,
            "total_degradation_area": 0.0,
            "cumulative_reward_deficit": 0.0,
        }

    MIN_ABS_TOLERANCE = 0.06  # floor so a small baseline doesn't make tolerance vanish
    tolerance = max(abs(baseline) * tolerance_frac, MIN_ABS_TOLERANCE)
    threshold = baseline - tolerance

    # A performance degradation must be observed before recovery is eligible.
    degradation_start = None
    for i in range(onset_idx, fault_window_end):
        if _is_finite_number(series[i]) and float(series[i]) < threshold:
            degradation_start = i
            break

    # If never degraded, the window is trivially healthy.
    if degradation_start is None:
        window_len = fault_window_end - onset_idx
        return {
            "baseline_reward": round(baseline, 4),
            "recovery_threshold": round(threshold, 4),
            "degradation_detected": False,
            "degradation_start_step": None,
            "recovered": False,
            "first_recovery_idx": None,
            "first_recovery_step": None,
            "time_to_first_recovery": None,
            "sustained_recovery": False,
            "recovery_events": [],
            "relapse_count": 0,
            "time_in_recovered_state": window_len if window_len > 0 else 0,
            "fraction_time_healthy": 1.0 if window_len > 0 else None,
            "total_degradation_area": 0.0,
            "cumulative_reward_deficit": 0.0,
        }

    # Full pass: track every sustained recovery and relapse in the window.
    recovery_events = []
    current_recovery = None
    healthy_run = 0
    first_recovery_idx = None
    time_in_recovered_state = 0

    for i in range(degradation_start, fault_window_end):
        val = float(series[i]) if _is_finite_number(series[i]) else None
        if val is None:
            healthy_run = 0
            if current_recovery is not None:
                current_recovery["relapsed_at"] = i
                recovery_events.append(current_recovery)
                current_recovery = None
            continue

        if val >= threshold:
            healthy_run += 1
            time_in_recovered_state += 1
            if healthy_run >= required_healthy_steps:
                recovery_end = i
                if current_recovery is None:
                    current_recovery = {"recovered_at": recovery_end, "relapsed_at": None}
                    if first_recovery_idx is None:
                        first_recovery_idx = recovery_end
        else:
            healthy_run = 0
            if current_recovery is not None:
                current_recovery["relapsed_at"] = i
                recovery_events.append(current_recovery)
                current_recovery = None

    if current_recovery is not None:
        recovery_events.append(current_recovery)

    sustained_recovery = bool(recovery_events and recovery_events[-1]["relapsed_at"] is None)
    relapse_count = (
        len(recovery_events) - 1
        if (recovery_events and recovery_events[-1]["relapsed_at"] is None)
        else len(recovery_events)
    )

    window_len = fault_window_end - onset_idx
    fraction_time_healthy = (time_in_recovered_state / window_len) if window_len > 0 else 0.0

    total_degradation_area = sum(
        max(0.0, baseline - float(series[i]))
        for i in range(onset_idx, fault_window_end)
        if _is_finite_number(series[i])
    )

    return {
        "baseline_reward": round(baseline, 4),
        "recovery_threshold": round(threshold, 4),
        "degradation_detected": True,
        "degradation_start_step": degradation_start,
        "recovered": first_recovery_idx is not None,
        "first_recovery_idx": first_recovery_idx,
        "first_recovery_step": first_recovery_idx,
        "time_to_first_recovery": (first_recovery_idx - onset_idx) if first_recovery_idx is not None else None,
        "sustained_recovery": sustained_recovery,
        "recovery_events": recovery_events,
        "relapse_count": relapse_count,
        "time_in_recovered_state": time_in_recovered_state,
        "fraction_time_healthy": round(fraction_time_healthy, 4),
        "total_degradation_area": round(total_degradation_area, 4),
        "cumulative_reward_deficit": round(total_degradation_area, 4),
    }


# ---------------------------------------------------------*/
# Time-to-recovery + degradation-area
# ---------------------------------------------------------
def _score_fault_event(
    total_series,
    sort_series,
    press_series,
    onset_step,
    duration,
    tolerance_frac,
    required_healthy_steps=15,
    cold_start_steps=5,
):
    """
    Score performance recovery for one fault event.

    Tracks the *entire* fault-active window, recording every recovery/relapse
    cycle, per-channel deficits, and window-complete (non-truncated) metrics.
    """
    n = len(total_series)
    required_healthy_steps = max(1, int(required_healthy_steps))
    tolerance_frac = max(0.0, float(tolerance_frac))

    if n == 0:
        return {
            "onset_step": int(onset_step),
            "duration": duration,
            "pre_fault_baseline_reward": 0.0,
            "pre_fault_baseline_sort_reward": None,
            "pre_fault_baseline_press_reward": None,
            "recovery_threshold": 0.0,
            "recovery_threshold_sort": None,
            "recovery_threshold_press": None,
            "degradation_detected": False,
            "degradation_start_step": None,
            "recovered": False,
            "first_recovery_idx": None,
            "first_recovery_step": None,
            "time_to_first_recovery": None,
            "sustained_recovery": False,
            "recovery_events": [],
            "relapse_count": 0,
            "time_in_recovered_state": 0,
            "fraction_time_healthy": None,
            "total_degradation_area": 0.0,
            "cumulative_reward_deficit": 0.0,
            "sort_degradation_area": None,
            "press_degradation_area": None,
            "sort_recovered": None,
            "press_recovered": None,
            "sort_first_recovery_step": None,
            "press_first_recovery_step": None,
            "reward_recovery_window_end": 0,
            "fault_active_window_end": 0,
            # Backward compatibility aliases
            "baseline_reward": 0.0,
            "recovery_step": None,
            "time_to_recovery": None,
            "degradation_area": 0.0,
        }

    onset_idx = max(0, min(int(onset_step), n - 1))

    # Recovery/safety/benchmark scoring always looks through the rest of the
    # episode, not just while the fault is actively being injected -- a
    # transient fault that clears at step 55 but only gets rewarded back to
    # baseline at step 62 should still count as recovered. True injection
    # duration remains available separately via onset_step + duration on
    # this same event dict, so nothing is lost by not capping the window here.
    fault_window_end = n


    # Baseline uses only complete pre-fault observations, excluding cold-start.
    def _baseline(series):
        pre_fault = series[:onset_idx]
        if len(pre_fault) > cold_start_steps:
            return sum(pre_fault[cold_start_steps:]) / len(pre_fault[cold_start_steps:])
        elif pre_fault:
            return sum(pre_fault) / len(pre_fault)
        else:
            fallback = series[cold_start_steps:min(cold_start_steps + 10, len(series))]
            return sum(fallback) / len(fallback) if fallback else 0.0

    baseline_total = _baseline(total_series)
    baseline_sort = _baseline(sort_series) if sort_series else 0.0
    baseline_press = _baseline(press_series) if press_series else 0.0

    total_metrics = _score_single_channel(
        total_series, onset_idx, fault_window_end, baseline_total, tolerance_frac, required_healthy_steps
    )
    sort_metrics = _score_single_channel(
        sort_series, onset_idx, fault_window_end, baseline_sort, tolerance_frac, required_healthy_steps
    ) if sort_series else None
    press_metrics = _score_single_channel(
        press_series, onset_idx, fault_window_end, baseline_press, tolerance_frac, required_healthy_steps
    ) if press_series else None

    result = {
        "onset_step": int(onset_step),
        "duration": duration,
        "pre_fault_baseline_reward": total_metrics["baseline_reward"],
        "pre_fault_baseline_sort_reward": sort_metrics["baseline_reward"] if sort_metrics else None,
        "pre_fault_baseline_press_reward": press_metrics["baseline_reward"] if press_metrics else None,
        "recovery_threshold": total_metrics["recovery_threshold"],
        "recovery_threshold_sort": sort_metrics["recovery_threshold"] if sort_metrics else None,
        "recovery_threshold_press": press_metrics["recovery_threshold"] if press_metrics else None,
        "degradation_detected": total_metrics["degradation_detected"],
        "degradation_start_step": total_metrics["degradation_start_step"],
        "recovered": total_metrics["recovered"],
        "first_recovery_idx": total_metrics["first_recovery_idx"],
        "first_recovery_step": total_metrics["first_recovery_step"],
        "time_to_first_recovery": total_metrics["time_to_first_recovery"],
        "sustained_recovery": total_metrics["sustained_recovery"],
        "recovery_events": total_metrics["recovery_events"],
        "relapse_count": total_metrics["relapse_count"],
        "time_in_recovered_state": total_metrics["time_in_recovered_state"],
        "fraction_time_healthy": total_metrics["fraction_time_healthy"],
        "total_degradation_area": total_metrics["total_degradation_area"],
        "cumulative_reward_deficit": total_metrics["cumulative_reward_deficit"],
        "reward_recovery_window_end": (
            total_metrics["first_recovery_idx"] + 1
            if total_metrics["first_recovery_idx"] is not None
            else fault_window_end
        ),
        "fault_active_window_end": fault_window_end,
    }

    if sort_metrics:
        result["sort_degradation_area"] = sort_metrics["total_degradation_area"]
        result["sort_recovered"] = sort_metrics["recovered"]
        result["sort_first_recovery_step"] = sort_metrics["first_recovery_step"]
    else:
        result["sort_degradation_area"] = None
        result["sort_recovered"] = None
        result["sort_first_recovery_step"] = None

    if press_metrics:
        result["press_degradation_area"] = press_metrics["total_degradation_area"]
        result["press_recovered"] = press_metrics["recovered"]
        result["press_first_recovery_step"] = press_metrics["first_recovery_step"]
    else:
        result["press_degradation_area"] = None
        result["press_recovered"] = None
        result["press_first_recovery_step"] = None

    # Backward compatibility aliases
    result["baseline_reward"] = result["pre_fault_baseline_reward"]
    result["recovery_step"] = result["first_recovery_step"]
    result["time_to_recovery"] = result["time_to_first_recovery"]
    result["degradation_area"] = result["total_degradation_area"]

    return result


def _is_finite_number(value):
    """Return True when *value* can be interpreted as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------*/
# Intervention precision / recall / F1
# ---------------------------------------------------------*/
def _intervention_flag(entry, channel):
    """Check intervention flag, supporting both old and new field names."""
    if channel == "sort":
        return entry.get("intervened_sort", False) or entry.get("detected_sort_active", False)
    else:
        return entry.get("intervened_press", False) or entry.get("detected_press_active", False)


def _intervention_scoring(detection_log):
    """
    Compute intervention precision (as false-discovery rates), recall, and F1
    from the uniform detection log.

    Old ``false_recovery_rate`` is retained as a deprecated alias for backward
    compatibility, but the canonical metrics are now ``intervention_precision``,
    ``intervention_recall``, and ``intervention_f1``.
    """
    if not detection_log:
        return {
            "intervention_precision": {
                "sort_false_positive_rate": None,
                "press_false_positive_rate": None,
                "overall_false_positive_rate": None,
                "n_flagged_steps": 0,
            },
            "intervention_recall": {
                "sort_recall": None,
                "press_recall": None,
                "overall_recall": None,
                "n_true_active_steps": 0,
            },
            "intervention_f1": {
                "sort_f1": None,
                "press_f1": None,
                "overall_f1": None,
            },
            # Deprecated alias
            "false_recovery_rate": {
                "sort": None,
                "press": None,
                "overall": None,
                "n_flagged_steps": 0,
            },
        }

    # ---------- Precision (false discovery rates) ----------
    sort_flagged = [e for e in detection_log if _intervention_flag(e, "sort")]
    press_flagged = [e for e in detection_log if _intervention_flag(e, "press")]
    any_flagged = [
        e for e in detection_log
        if _intervention_flag(e, "sort") or _intervention_flag(e, "press")
    ]

    def _fdr(flagged, true_key):
        if not flagged:
            return None
        false_positives = [e for e in flagged if not e.get(true_key, False)]
        return round(len(false_positives) / len(flagged), 4)

    overall_false = [
        e for e in any_flagged
        if not (e.get("true_sort_active", False) or e.get("true_press_active", False))
    ]
    overall_fdr = round(len(overall_false) / len(any_flagged), 4) if any_flagged else None

    precision = {
        "sort_false_positive_rate": _fdr(sort_flagged, "true_sort_active"),
        "press_false_positive_rate": _fdr(press_flagged, "true_press_active"),
        "overall_false_positive_rate": overall_fdr,
        "n_flagged_steps": len(any_flagged),
    }

    # ---------- Recall ----------
    sort_true_active = [e for e in detection_log if e.get("true_sort_active", False)]
    press_true_active = [e for e in detection_log if e.get("true_press_active", False)]
    any_true_active = [
        e for e in detection_log
        if e.get("true_sort_active", False) or e.get("true_press_active", False)
    ]

    sort_tp = sum(
        1 for e in detection_log
        if e.get("true_sort_active", False) and _intervention_flag(e, "sort")
    )
    press_tp = sum(
        1 for e in detection_log
        if e.get("true_press_active", False) and _intervention_flag(e, "press")
    )
    any_tp = sum(
        1 for e in detection_log
        if (e.get("true_sort_active", False) or e.get("true_press_active", False))
        and (_intervention_flag(e, "sort") or _intervention_flag(e, "press"))
    )

    sort_recall = round(sort_tp / len(sort_true_active), 4) if sort_true_active else None
    press_recall = round(press_tp / len(press_true_active), 4) if press_true_active else None
    overall_recall = round(any_tp / len(any_true_active), 4) if any_true_active else None

    recall = {
        "sort_recall": sort_recall,
        "press_recall": press_recall,
        "overall_recall": overall_recall,
        "n_true_active_steps": len(any_true_active),
    }

    # ---------- F1 ----------
    def _f1(prec, rec):
        if prec is None or rec is None:
            return None
        if prec + rec == 0:
            return 0.0
        return round(2 * prec * rec / (prec + rec), 4)

    # Convert FDR to precision = 1 - FDR
    sort_precision = 1.0 - precision["sort_false_positive_rate"] if precision["sort_false_positive_rate"] is not None else None
    press_precision = 1.0 - precision["press_false_positive_rate"] if precision["press_false_positive_rate"] is not None else None
    overall_precision = 1.0 - precision["overall_false_positive_rate"] if precision["overall_false_positive_rate"] is not None else None

    f1 = {
        "sort_f1": _f1(sort_precision, sort_recall),
        "press_f1": _f1(press_precision, press_recall),
        "overall_f1": _f1(overall_precision, overall_recall),
    }

    # Deprecated alias (old field names, old values)
    old_alias = {
        "sort": precision["sort_false_positive_rate"],
        "press": precision["press_false_positive_rate"],
        "overall": precision["overall_false_positive_rate"],
        "n_flagged_steps": precision["n_flagged_steps"],
    }

    return {
        "intervention_precision": precision,
        "intervention_recall": recall,
        "intervention_f1": f1,
        "false_recovery_rate": old_alias,
    }


# ---------------------------------------------------------*/
# Safety-violation count
# ---------------------------------------------------------*/
def _safety_violations(
    reward_data,
    material_names,
    container_max,
    container_global_max,
    window_start,
    window_end,
    near_miss_ratio=0.90,
    severe_ratio=0.95,
    quality_near_miss_threshold=0.90,
    quality_severe_threshold=0.80,
):
    """
    Count near-miss, severe, and catastrophic fill-level violations,
    plus quality/purity near-miss and severe counts.

    The fill-level thresholds mirror the environment's existing container
    safety tiers; the quality tiers use accuracy/purity dips from
    ``reward_data['Accuracy']``.
    """
    per_material = {}
    near_miss_count = 0
    severe_count = 0
    catastrophic_count = 0

    materials = list(material_names or [])
    if "E" not in materials:
        materials.append("E")

    for mat in materials:
        true_vals = reward_data.get(f"{mat}_True", []) or []
        false_vals = reward_data.get(f"{mat}_False", []) or []

        cap = (
            container_max.get(mat, container_global_max)
            if container_max
            else container_global_max
        )
        if not cap:
            continue

        n = min(len(true_vals), len(false_vals))
        lo = max(0, int(window_start))
        hi = min(n, int(window_end)) if window_end is not None else n

        mat_near = 0
        mat_severe = 0
        mat_cat = 0

        for i in range(lo, hi):
            try:
                level = float(true_vals[i]) + float(false_vals[i])
                ratio = level / float(cap)
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            if ratio > 1.0:
                mat_cat += 1
            elif ratio > severe_ratio:
                mat_severe += 1
            elif ratio > near_miss_ratio:
                mat_near += 1

        per_material[mat] = {
            "near_miss": mat_near,
            "severe": mat_severe,
            "catastrophic": mat_cat,
        }
        near_miss_count += mat_near
        severe_count += mat_severe
        catastrophic_count += mat_cat

    # Quality / purity safety tier
    quality_near_miss_count = 0
    quality_severe_count = 0
    accuracy_series = reward_data.get("Accuracy", []) or []

    if accuracy_series:
        n_acc = len(accuracy_series)
        lo = max(0, int(window_start))
        hi = min(n_acc, int(window_end)) if window_end is not None else n_acc
        for i in range(lo, hi):
            try:
                acc = float(accuracy_series[i])
            except (TypeError, ValueError):
                continue
            if acc < quality_severe_threshold:
                quality_severe_count += 1
            elif acc < quality_near_miss_threshold:
                quality_near_miss_count += 1

    return {
        "near_miss_count": near_miss_count,
        "severe_count": severe_count,
        "catastrophic_count": catastrophic_count,
        "quality_near_miss_count": quality_near_miss_count,
        "quality_severe_count": quality_severe_count,
        "per_material": per_material,
    }


def _reaction_timing(detection_log, onset_step):
    """
    Return the first post-onset detection/intervention step and latency.

    This is intentionally separate from performance recovery. The detection
    log used by the project marks whether sorting/pressing fault state was
    flagged at each step.
    """
    if not detection_log:
        return {"reaction_step": None, "reaction_latency": None}

    onset = int(onset_step)
    first_step = None

    for entry in detection_log:
        try:
            step = int(entry.get("step"))
        except (TypeError, ValueError):
            continue

        if step < onset:
            continue

        flagged = (
            _intervention_flag(entry, "sort") or _intervention_flag(entry, "press")
        )
        if flagged:
            first_step = step
            break

    if first_step is None:
        return {"reaction_step": None, "reaction_latency": None}

    return {
        "reaction_step": first_step,
        "reaction_latency": first_step - onset,
    }


# ---------------------------------------------------------*/
# Composite benchmark score
# ---------------------------------------------------------*/
# PAPER-GRADE BENCHMARK SCORE — design rationale and formula
# -----------------------------------------------------------
#
# WHY A SINGLE NUMBER?
# --------------------
# Cross-paradigm comparison (oracle rule-based vs. learned detector vs.
# hierarchical supervisor) needs a common currency.  Raw episode reward is
# incomparable across fault types because different faults inflict different
# reward scales (sensor_noise may dip to ~-0.5/step, agent_dropout to ~-2.0).
# A bounded [0, 100] score lets us report "mean ± std" across seeds and
# fault types in one table.
#
# DESIGN PRINCIPLE
# ----------------
# The score is a *weighted penalty* converted to a bounded reward:
#
#     P  = Σ w_i · pen_i          (pen_i ∈ [0, 1], 0 = perfect)
#     Score = 100 · (1 − clip(P, 0, 1))
#
# Each penalty term is *independently normalised* so that a fault type with
# a large possible deficit does not automatically dominate the composite.
# Normalisation uses the *fault-active window length* (onset → end), not the
# episode length, so transient and permanent faults are comparable.
#
# COMPONENT PENALTIES  (each ∈ [0, 1], 0 = perfect)
# ---------------------------------------------------
#
# 1. SAFETY PENALTY  —  safety_pen
#    Captures container-overflow and quality/purity risk.  Catastrophic
#    events (ratio > 1.0) are weighted highest because they represent the
#    actual operational failure mode; severe and near-miss are scaled down.
#
#        safety_pen = clip( (5·N_cat + 2·N_sev + 1·N_near) / window_len , 0, 1)
#
#    The linear weighting (5 / 2 / 1) is chosen to reflect industrial
#    safety-tier severity.  The cap at 1.0 prevents a single catastrophic
#    episode from washing out all other signal.
#
# 2. REWARD-DEFICIT PENALTY  —  deficit_pen
#    Measures how much reward was lost relative to the best-case baseline.
#    Normalised by the *theoretical maximum deficit* so that a fault that
#    could at worst drive reward to reward_floor does not score worse than
#    a fault with a smaller floor just because the floor is lower.
#
#        max_deficit_per_step = pre_fault_baseline_reward − reward_floor
#        deficit_pen = clip( cumulative_reward_deficit
#                            / (max_deficit_per_step · window_len) , 0, 1)
#
#    reward_floor is the known minimum combined reward per step (e.g. -2
#    for this environment).  If the baseline is below the floor the penalty
#    is clamped to 0.
#
# 3. RELAPSE PENALTY  —  relapse_pen
#    A mechanism that recovers once but relapses repeatedly is worse than
#    one that sustains recovery.  This term directly measures that.
#
#        relapse_pen = 1 − fraction_time_healthy
#
#    fraction_time_healthy = steps_at_or_above_threshold_after_degradation
#                            / window_len
#
#    If no degradation is ever detected, fraction_time_healthy = 1.0 and
#    the penalty is 0 (the episode was trivially healthy).
#
# 4. TIME-TO-RECOVERY PENALTY  —  ttr_pen
#    Latency to the *first* confirmed recovery.  If recovery never occurs
#    the penalty is the maximum 1.0.
#
#        ttr_pen = clip( time_to_first_recovery / window_len , 0, 1)   [if recovered]
#        ttr_pen = 1.0                                                   [otherwise]
#
#    Using first recovery (not sustained recovery) keeps the penalty
#    monotonic: a later first recovery can only increase the penalty, never
#    decrease it.
#
# COMPOSITE FORMULA
# -----------------
# Let the weights be w_safety, w_deficit, w_relapse, w_ttr (summing to 1.0
# for interpretability, though the code does not enforce this).
#
#        P = w_safety · safety_pen
#          + w_deficit · deficit_pen
#          + w_relapse · relapse_pen
#          + w_ttr     · ttr_pen
#
#        Benchmark Score = 100 × (1 − clip(P, 0, 1))
#
# DEFAULT WEIGHTS (industrial-safety-flavoured)
# ---------------------------------------------
#        safety  = 0.40   # highest: physical/operational failure matters most
#        deficit = 0.25   # economic cost of lost throughput
#        relapse = 0.20   # stability of the recovery
#        ttr     = 0.15   # speed of initial response
#
# These weights are a *knob* — they should be stated explicitly in the
# paper, and a sensitivity check (e.g. uniform 0.25 weights, or safety=0.6)
# should be reported as a robustness line in the results table.
#
# WHY PRECISION / RECALL ARE EXCLUDED
# -----------------------------------
# Intervention precision (false-positive rate) and recall are reported
# *alongside* the score, not folded into it.  Including them would double-
# count the same underlying effect: a mechanism with low recall that never
# intervenes will already score badly on safety, deficit, and relapse because
# the fault goes unmitigated.  Precision/recall explain *why* a mechanism
# scored well or badly; the composite score captures the *outcome*.
#
# AGGREGATION
# -----------
# For the actual cross-paradigm comparison, run each
# (fault_type × mechanism) cell over multiple seeds and report
#
#        mean(Score) ± std(Score)
#
# A single seed (e.g. seed=42) is one sample from a noisy stochastic
# process; the composite is only trustworthy when aggregated.
# -----------------------------------------------------------

def _compute_benchmark_score(
    event,
    reward_floor=-2.0,
    weights=None,
):
    """
    Convert per-event outcome metrics into a single bounded benchmark score.

    See the extended comment block above for the full design rationale,
    normalisation strategy, and sensitivity-check guidance.
    """
    if weights is None:
        weights = {"safety": 0.40, "deficit": 0.25, "relapse": 0.20, "ttr": 0.15}

    window_len = event.get("fault_active_window_end", 0) - event.get("onset_step", 0)
    if window_len <= 0:
        return None

    safety = event.get("safety", {})
    cat = safety.get("catastrophic_count", 0)
    sev = safety.get("severe_count", 0)
    near = safety.get("near_miss_count", 0)
    safety_pen = min(1.0, (5 * cat + 2 * sev + 1 * near) / window_len)

    baseline = event.get("pre_fault_baseline_reward", 0.0)
    max_deficit_per_step = max(0.0, baseline - reward_floor)
    total_deficit = event.get("cumulative_reward_deficit", 0.0)
    if max_deficit_per_step > 0 and window_len > 0:
        deficit_pen = min(1.0, total_deficit / (max_deficit_per_step * window_len))
    else:
        deficit_pen = 0.0

    frac_healthy = event.get("fraction_time_healthy")
    if frac_healthy is None:
        frac_healthy = 1.0
    relapse_pen = 1.0 - frac_healthy

    if event.get("recovered", False):
        ttr = event.get("time_to_first_recovery", window_len)
        ttr_pen = min(1.0, ttr / window_len)
    else:
        ttr_pen = 1.0

    P = (
        weights.get("safety", 0.0) * safety_pen
        + weights.get("deficit", 0.0) * deficit_pen
        + weights.get("relapse", 0.0) * relapse_pen
        + weights.get("ttr", 0.0) * ttr_pen
    )

    score = 100 * (1.0 - max(0.0, min(1.0, P)))
    return round(score, 4)
# ---------------------------------------------------------*/
# Public entry point
# ---------------------------------------------------------*/
def compute_recovery_metrics(
    env,
    fault_mode="none",
    recovery_mode="none",
    seed=None,
    recovery_tolerance=0.15,
    rolling_window=5,
    sustain_windows=3,
    required_healthy_steps=None,
    cold_start_steps=5,
    near_miss_ratio=0.90,
    severe_ratio=0.95,
    quality_near_miss_threshold=0.90,
    quality_severe_threshold=0.80,
    reward_floor=-2.0,
    benchmark_weights=None,
):
    """
    Compute all recovery metrics for the completed episode in `env`.

    Parameters
    ----------
    required_healthy_steps : int or None
        Consecutive healthy steps required to declare recovery. If None,
        computed as ``rolling_window * sustain_windows`` for backward
        compatibility. The old two-parameter indirection is deprecated.
    cold_start_steps : int
        Number of initial episode steps excluded from the pre-fault baseline
        (reward is often near-zero during ramp-up).
    reward_floor : float
        Minimum possible reward per step, used to normalize the deficit
        penalty in the benchmark score.
    benchmark_weights : dict or None
        Weights for the composite score. Defaults to
        {"safety": 0.40, "deficit": 0.25, "relapse": 0.20, "ttr": 0.15}.
    """
    reward_data = getattr(env, "reward_data", {}) or {}
    fault_log = getattr(env, "fault_log", []) or []
    detection_log = getattr(env, "detection_log", []) or []
    material_names = getattr(env, "material_names", []) or []
    container_max = getattr(env, "container_max", {}) or {}
    container_global_max = getattr(env, "container_global_max", None)

    total_series = _total_reward_series(reward_data)
    sort_series, press_series = _channel_reward_series(reward_data)

    if required_healthy_steps is None:
        required_healthy_steps = max(1, int(rolling_window)) * max(1, int(sustain_windows))

    per_event = []
    for event in fault_log:
        onset = event.get("onset_step", 0)
        duration = event.get("duration")

        scored = _score_fault_event(
            total_series,
            sort_series,
            press_series,
            onset,
            duration,
            recovery_tolerance,
            required_healthy_steps,
            cold_start_steps,
        )

        # Safety is scored over the complete fault-active window, independently
        # of reward recovery. This is especially important for permanent faults:
        # slow-building consequences can occur long after reward first looks
        # healthy.
        scored["safety"] = _safety_violations(
            reward_data,
            material_names,
            container_max,
            container_global_max,
            scored["onset_step"],
            scored["fault_active_window_end"],
            near_miss_ratio,
            severe_ratio,
            quality_near_miss_threshold,
            quality_severe_threshold,
        )
        scored.update(_reaction_timing(detection_log, scored["onset_step"]))
        scored["fault_type"] = event.get("type")
        scored["benchmark_score"] = _compute_benchmark_score(
            scored, reward_floor, benchmark_weights
        )
        per_event.append(scored)

    intervention_scores = _intervention_scoring(detection_log)

    # Episode-wide safety is evaluated over the complete logged episode.
    episode_safety = _safety_violations(
        reward_data,
        material_names,
        container_max,
        container_global_max,
        0,
        None,
        near_miss_ratio,
        severe_ratio,
        quality_near_miss_threshold,
        quality_severe_threshold,
    )

    result = {
        "fault_mode": fault_mode,
        "recovery_mode": recovery_mode,
        "seed": seed,
        "episode_length": len(total_series),
        "total_reward": round(sum(total_series), 4) if total_series else None,
        "fault_events": per_event,
        "episode_safety_violations": episode_safety,
        "computed_at": time.time(),
    }
    result.update(intervention_scores)

    return result


# ---------------------------------------------------------*/
# Printing / persistence
# ---------------------------------------------------------*/
def print_recovery_metrics(metrics):
    """Pretty-print a metrics report."""
    print(
        f"\n📊 Recovery metrics [fault: {metrics['fault_mode']}, "
        f"recovery: {metrics['recovery_mode']}, seed: {metrics['seed']}]"
    )

    if not metrics["fault_events"]:
        print("  No fault events logged this episode.")

    for i, ev in enumerate(metrics["fault_events"]):
        if ev["recovered"]:
            ttr = f"{ev['time_to_first_recovery']} steps"
        else:
            ttr = "DID NOT RECOVER within episode"

        duration_label = (
            "permanent" if ev["duration"] is None else str(ev["duration"])
        )

        print(
            f"  Fault #{i} ({ev['fault_type']}, onset step {ev['onset_step']}, "
            f"duration={duration_label}):"
        )
        print(f"    baseline reward/step : {ev['pre_fault_baseline_reward']}")
        print(f"    recovery threshold   : {ev.get('recovery_threshold')}")
        print(f"    degradation detected : {ev.get('degradation_detected')}"
              f" (step {ev.get('degradation_start_step')})")
        print(f"    reaction latency     : "
              f"{ev.get('reaction_latency') if ev.get('reaction_latency') is not None else 'n/a'} steps")
        print(f"    time-to-first-recovery : {ttr}")
        print(f"    first recovery step  : {ev['first_recovery_step']}")
        print(f"    sustained recovery   : {ev.get('sustained_recovery')}")
        print(f"    relapse count        : {ev.get('relapse_count')}")
        print(f"    fraction time healthy: {ev.get('fraction_time_healthy')}")
        print(f"    total degradation area : {ev['total_degradation_area']}")
        print(f"    cumulative reward deficit : {ev['cumulative_reward_deficit']}")

        if ev.get("sort_degradation_area") is not None:
            print(f"    sort degradation area: {ev['sort_degradation_area']} (recovered: {ev['sort_recovered']})")
        if ev.get("press_degradation_area") is not None:
            print(f"    press degradation area: {ev['press_degradation_area']} (recovered: {ev['press_recovered']})")

        print(f"    benchmark score      : {ev.get('benchmark_score')}")

        safety = ev["safety"]
        fault_window_end = ev["fault_active_window_end"]
        if fault_window_end > ev["onset_step"]:
            fault_window_label = f"{ev['onset_step']}-{fault_window_end - 1}"
        else:
            fault_window_label = str(ev["onset_step"])

        print(
            f"    safety violations    : "
            f"{safety['catastrophic_count']} catastrophic, "
            f"{safety['severe_count']} severe, "
            f"{safety['near_miss_count']} near-miss "
            f"(while fault active: steps {fault_window_label})"
        )
        if safety.get("quality_severe_count", 0) > 0 or safety.get("quality_near_miss_count", 0) > 0:
            print(
                f"    quality violations   : "
                f"{safety['quality_severe_count']} severe, "
                f"{safety['quality_near_miss_count']} near-miss"
            )

    ip = metrics.get("intervention_precision", {})
    ir = metrics.get("intervention_recall", {})
    if1 = metrics.get("intervention_f1", {})

    if ip.get("overall_false_positive_rate") is None:
        print(
            "  Intervention precision : n/a "
            "(no detection_log - baseline/none run, or never triggered)"
        )
    else:
        print(
            f"  Intervention precision : "
            f"overall_fpr={ip['overall_false_positive_rate']} "
            f"(sort={ip['sort_false_positive_rate']}, press={ip['press_false_positive_rate']}, "
            f"over {ip['n_flagged_steps']} flagged steps)"
        )
        print(
            f"  Intervention recall    : "
            f"overall={ir.get('overall_recall')} "
            f"(sort={ir.get('sort_recall')}, press={ir.get('press_recall')}, "
            f"over {ir.get('n_true_active_steps')} true-active steps)"
        )
        print(
            f"  Intervention F1        : "
            f"overall={if1.get('overall_f1')} "
            f"(sort={if1.get('sort_f1')}, press={if1.get('press_f1')})"
        )

    es = metrics["episode_safety_violations"]
    print(
        f"  Episode-wide safety  : {es['catastrophic_count']} catastrophic, "
        f"{es['severe_count']} severe, {es['near_miss_count']} near-miss"
    )
    if es.get("quality_severe_count", 0) > 0 or es.get("quality_near_miss_count", 0) > 0:
        print(
            f"  Episode-wide quality   : {es['quality_severe_count']} severe, "
            f"{es['quality_near_miss_count']} near-miss"
        )
    print(
        f"  Total reward (context only, NOT the eval metric): "
        f"{metrics['total_reward']}"
    )


def save_recovery_metrics(metrics, path=DEFAULT_LOG_PATH):
    """
    Append one JSON line per run.

    The accumulated JSONL can later be loaded into pandas for aggregation
    across fault types, mechanisms, and seeds.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")