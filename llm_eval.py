"""
LangGraph FaultBench JSON Analyzer

Takes one or more FaultBench JSON evaluation outputs as user input and
produces a compact TXT report containing:
  1. A standardized metrics table.
  2. One compact analysis card per evaluation run.
  3. A concise cross-run summary when multiple JSON objects are supplied.

Requirements:
    pip install langgraph langchain langchain-openai pydantic

Environment:
    OPENAI_API_KEY=...

Run:
    python faultbench_langgraph.py

The workflow accepts either:
    - a single JSON object
    - a JSON array of objects
    - multiple JSON objects pasted one after another
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()




# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FaultBenchState(TypedDict, total=False):
    user_input: str
    runs: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    analysis_cards: list[str]
    cross_run_summary: str
    report: str
    output_path: str


# ---------------------------------------------------------------------------
# Structured LLM output
# ---------------------------------------------------------------------------

class RunAnalysis(BaseModel):
    overall_assessment: str = Field(
        description="One short classification such as Good, Moderate, Poor, or Critical."
    )
    detection: str = Field(description="Short assessment of fault detection/reaction.")
    recovery: str = Field(description="Short assessment of recovery speed.")
    stability: str = Field(description="Short assessment of sustained recovery/relapse behavior.")
    intervention: str = Field(description="Short assessment of intervention quality.")
    safety: str = Field(description="Short assessment of safety violations.")
    primary_issue: str = Field(description="The single most important weakness.")
    key_findings: list[str] = Field(
        min_length=3, max_length=5,
        description="3-5 concise, evidence-based findings."
    )
    conclusion: str = Field(
        description="One concise sentence summarizing the run."
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Accept a JSON object, JSON array, or multiple pasted JSON objects."""
    text = text.strip()

    if not text:
        raise ValueError("No JSON input was provided.")

    # First try normal JSON parsing.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract balanced top-level JSON objects.
    decoder = json.JSONDecoder()
    objects = []
    pos = 0

    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break

        if text[pos] != "{":
            pos += 1
            continue

        try:
            obj, end = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON near character {pos}: {exc}") from exc

        if not isinstance(obj, dict):
            raise ValueError("Every extracted JSON value must be an object.")

        objects.append(obj)
        pos += end

    if not objects:
        raise ValueError("Could not find a valid JSON object.")

    return objects


def _first_event(run: dict[str, Any]) -> dict[str, Any]:
    events = run.get("fault_events") or []
    return events[0] if events else {}


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Deterministic metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(state: FaultBenchState) -> FaultBenchState:
    rows = []

    for run in state["runs"]:
        event = _first_event(run)
        precision = run.get("intervention_precision", {})
        recall = run.get("intervention_recall", {})
        f1 = run.get("intervention_f1", {})
        safety = run.get("episode_safety_violations", {})

        onset = event.get("onset_step")
        detected = event.get("degradation_start_step")
        recovery = event.get("first_recovery_step")
        reaction = event.get("reaction_step")

        row = {
            "Fault": run.get("fault_mode", event.get("fault_type", "N/A")),
            "Recovery": run.get("recovery_mode", "N/A"),
            "Seed": run.get("seed", "N/A"),
            "Episode": run.get("episode_length", "N/A"),
            "Benchmark Score": _num(event.get("benchmark_score"), 2),
            "Total Reward": _num(run.get("total_reward"), 4),
            "Baseline Reward": _num(
                event.get("pre_fault_baseline_reward", event.get("baseline_reward")), 4
            ),
            "Degradation Area": _num(
                event.get("total_degradation_area", event.get("degradation_area")), 4
            ),
            "Fault Onset": onset if onset is not None else "N/A",
            "Detection Step": detected if detected is not None else "N/A",
            "Detection Latency": (
                detected - onset
                if onset is not None and detected is not None
                else "N/A"
            ),
            "Reaction Latency": event.get("reaction_latency", "N/A"),
            "Recovery Step": recovery if recovery is not None else "N/A",
            "Recovery Time": event.get(
                "time_to_first_recovery",
                event.get("time_to_recovery", "N/A"),
            ),
            "Sustained Recovery": (
                "YES" if event.get("sustained_recovery") is True
                else "NO" if event.get("sustained_recovery") is False
                else "N/A"
            ),
            "Relapses": event.get("relapse_count", "N/A"),
            "Healthy Fraction": _pct(event.get("fraction_time_healthy")),
            "Intervention Precision": _pct(precision.get("overall_false_positive_rate") and 1 - precision["overall_false_positive_rate"]),
            "Intervention Recall": _pct(recall.get("overall")),
            "Intervention F1": _pct(f1.get("overall")),
            "False Positive Rate": _pct(precision.get("overall_false_positive_rate")),
            "Near Misses": safety.get("near_miss_count", "N/A"),
            "Severe Violations": safety.get("severe_count", "N/A"),
            "Catastrophic Violations": safety.get("catastrophic_count", "N/A"),
            "Quality Near Misses": safety.get("quality_near_miss_count", "N/A"),
            "Quality Severe": safety.get("quality_severe_count", "N/A"),
            "_sort_degradation": event.get("sort_degradation_area"),
            "_sort_recovery": event.get("sort_first_recovery_step"),
            "_sort_recall": recall.get("sort"),
            "_sort_f1": f1.get("sort"),
            "_sort_fpr": precision.get("sort_false_positive_rate"),
            "_press_degradation": event.get("press_degradation_area"),
            "_press_recovery": event.get("press_first_recovery_step"),
            "_press_recall": recall.get("press"),
            "_press_f1": f1.get("press"),
            "_press_fpr": precision.get("press_false_positive_rate"),
        }

        rows.append(row)

    return {**state, "rows": rows}


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def _build_analysis_prompt(run: dict[str, Any], row: dict[str, Any]) -> str:
    event = _first_event(run)

    return f"""
Analyze this FaultBench evaluation run.

Use ONLY the supplied data. Do not invent metrics or causes.

Return exactly these sections:

ASSESSMENT:
<one short sentence>

DETECTION:
<one short sentence>

RECOVERY:
<one short sentence>

STABILITY:
<one short sentence>

INTERVENTION:
<one short sentence>

SAFETY:
<one short sentence>

PRIMARY ISSUE:
<one short sentence>

KEY FINDINGS:
- <finding>
- <finding>
- <finding>
- <optional finding>

CONCLUSION:
<one short sentence>

Keep the entire response concise.

Fault: {row["Fault"]}
Recovery: {row["Recovery"]}
Seed: {row["Seed"]}

Benchmark score: {row["Benchmark Score"]}
Total reward: {row["Total Reward"]}
Baseline reward: {row["Baseline Reward"]}
Degradation area: {row["Degradation Area"]}

Fault onset: {row["Fault Onset"]}
Detection step: {row["Detection Step"]}
Detection latency: {row["Detection Latency"]}
Reaction latency: {row["Reaction Latency"]}
Recovery step: {row["Recovery Step"]}
Recovery time: {row["Recovery Time"]}
Sustained recovery: {row["Sustained Recovery"]}
Relapses: {row["Relapses"]}
Healthy fraction: {row["Healthy Fraction"]}

Intervention precision: {row["Intervention Precision"]}
Intervention recall: {row["Intervention Recall"]}
Intervention F1: {row["Intervention F1"]}
False positive rate: {row["False Positive Rate"]}

Near misses: {row["Near Misses"]}
Severe violations: {row["Severe Violations"]}
Catastrophic violations: {row["Catastrophic Violations"]}
Quality near misses: {row["Quality Near Misses"]}
Quality severe: {row["Quality Severe"]}

Sorting degradation: {row["_sort_degradation"]}
Sorting recovery: {row["_sort_recovery"]}
Sorting recall: {row["_sort_recall"]}
Sorting F1: {row["_sort_f1"]}

Pressing degradation: {row["_press_degradation"]}
Pressing recovery: {row["_press_recovery"]}
Pressing recall: {row["_press_recall"]}
Pressing F1: {row["_press_f1"]}
"""


def analyze_runs(state: FaultBenchState) -> FaultBenchState:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("MARL_LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found in .env")

    if not base_url:
        raise RuntimeError("MARL_LLM_BASE_URL was not found in .env")

    llm = ChatOpenAI(
        model="auto",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    cards = []

    for run, row in zip(state["runs"], state["rows"]):

        prompt = _build_analysis_prompt(run, row)

        response = llm.invoke(prompt)

        analysis = response.content

        card = f"""┌─────────────────────────────────────────┐
│ {str(row["Fault"])[:39]:<39} │
│ → {str(row["Recovery"])[:36]:<36} │
│ Seed: {str(row["Seed"])[:32]:<32} │
├─────────────────────────────────────────┤
│ Benchmark Score          {str(row["Benchmark Score"]):>10}         │
│ Recovery Time            {str(row["Recovery Time"]):>10} steps      │
│ Sustained Recovery       {str(row["Sustained Recovery"]):>10}         │
│ Intervention F1          {str(row["Intervention F1"]):>10}         │
│ Catastrophic Violations  {str(row["Catastrophic Violations"]):>10}         │
├─────────────────────────────────────────┤
│ LLM ANALYSIS                            │
│                                         │
{analysis}
│                                         │
└─────────────────────────────────────────┘"""

        cards.append(card)

    return {
        **state,
        "analysis_cards": cards,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "Fault", "Recovery", "Seed", "Episode", "Benchmark Score",
    "Total Reward", "Degradation Area", "Detection Latency",
    "Reaction Latency", "Recovery Time", "Sustained Recovery",
    "Relapses", "Healthy Fraction", "Intervention Precision",
    "Intervention Recall", "Intervention F1", "False Positive Rate",
    "Near Misses", "Severe Violations", "Catastrophic Violations",
]


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = TABLE_COLUMNS
    widths = []

    for h in headers:
        widths.append(max(len(h), *(len(str(r.get(h, "N/A"))) for r in rows)))

    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    divider = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = [
        "| " + " | ".join(str(r.get(h, "N/A")).ljust(w) for h, w in zip(headers, widths)) + " |"
        for r in rows
    ]

    return "\n".join([header, divider, *body])


def generate_report(state: FaultBenchState) -> FaultBenchState:
    lines = [
        "FAULTBENCH EVALUATION REPORT",
        "=" * 80,
        f"Evaluation runs: {len(state['runs'])}",
        "",
        "1. STANDARDIZED METRICS",
        "-" * 80,
        _markdown_table(state["rows"]),
        "",
        "2. INDIVIDUAL RUN ANALYSIS",
        "-" * 80,
        "",
    ]

    for card in state["analysis_cards"]:
        lines.extend([card, ""])

    # Include subsystem table because it is particularly useful for this schema.
    lines.extend([
        "3. SUBSYSTEM ANALYSIS",
        "-" * 80,
        "",
        "| Fault | Recovery | Seed | Sort Degradation | Sort Recovery | Sort F1 | Press Degradation | Press Recovery | Press F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in state["rows"]:
        lines.append(
            f"| {r['Fault']} | {r['Recovery']} | {r['Seed']} | "
            f"{_num(r['_sort_degradation'], 4)} | {r['_sort_recovery'] or 'N/A'} | "
            f"{_pct(r['_sort_f1'])} | {_num(r['_press_degradation'], 4)} | "
            f"{r['_press_recovery'] or 'N/A'} | {_pct(r['_press_f1'])} |"
        )

    lines.extend([
        "",
        "4. NOTES",
        "-" * 80,
        "• Metrics are extracted deterministically from the JSON.",
        "• Qualitative assessments are generated from the extracted metrics and raw event data.",
        "• No missing metric is inferred or fabricated; unavailable values are shown as N/A.",
        "",
    ])

    report = "\n".join(lines)

    output_path = os.getenv("FAULTBENCH_OUTPUT", "faultbench_report.txt")
    Path(output_path).write_text(report, encoding="utf-8")

    return {**state, "report": report, "output_path": output_path}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def parse_input(state: FaultBenchState) -> FaultBenchState:
    return {**state, "runs": _extract_json_objects(state["user_input"])}


def build_graph():
    graph = StateGraph(FaultBenchState)

    graph.add_node("parse_input", parse_input)
    graph.add_node("extract_metrics", extract_metrics)
    graph.add_node("analyze_runs", analyze_runs)
    graph.add_node("generate_report", generate_report)

    graph.add_edge(START, "parse_input")
    graph.add_edge("parse_input", "extract_metrics")
    graph.add_edge("extract_metrics", "analyze_runs")
    graph.add_edge("analyze_runs", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print("FaultBench LangGraph Analyzer")
    print("Paste your JSON output. Enter an empty line when finished.")
    print("For multiple runs, paste them as a JSON array or consecutive objects.")
    print()

    chunks = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line.strip():
            if chunks:
                break
            continue

        chunks.append(line)

    user_input = "\n".join(chunks)

    if not user_input.strip():
        print("No input supplied.")
        return

    app = build_graph()
    result = app.invoke({"user_input": user_input})

    print()
    print(result["report"])
    print()
    print(f"Report written to: {result['output_path']}")


if __name__ == "__main__":
    main()
