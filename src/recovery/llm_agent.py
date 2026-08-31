# ---------------------------------------------------------*\
# Title: LLM-Agent Replanning — LangGraph graph, tools, state formatting
# ---------------------------------------------------------*/
#
# Recovery mechanism #4 (ToDo #5, bullet 4): "LLM-agent replanning — an LLM
# that reads structured system state and outputs a recovery plan via tool
# calls into the env's control interface."
#
# This module is intentionally decoupled from the env classes: it takes a
# plain dict (the "system state snapshot") in and returns a plain dict (the
# "recovery plan") out. `llm_replanning.py` is what wires it into the
# fault-env mixin pattern (recovery_hook). Keeping the LangGraph plumbing
# here, separate from that mixin, makes it independently testable without
# spinning up a full env.
#
# Design decisions (see design discussion for the full rationale):
#   - Model: OpenAI-compatible chat model (via langchain-openai's ChatOpenAI).
#     Works against OpenAI directly, or any OpenAI-compatible proxy (e.g. a
#     local FreeLLMAPI-style router) by setting MARL_LLM_BASE_URL. Model
#     name defaults to "auto" (FreeLLMAPI's router-picks-for-you sentinel);
#     override with MARL_LLM_MODEL for a real OpenAI model name.
#   - Tools = the env's REAL control interface, with full control (not the
#     supervisor's fixed 4-way vocabulary): pass_through / override_sort
#     (choose any sort mode) / override_press (choose any press+material
#     target) / override_both. This is what makes this arm genuinely
#     "agentic" rather than just a re-implementation of the supervisor with
#     an LLM standing in for the PPO policy.
#   - Bounded ReAct loop: at most MAX_TOOL_TURNS turns, so one recovery
#     decision never costs more than a couple of round-trips. The agent CAN
#     use a read-only `get_more_detail` tool once to ask for a fuller state
#     dump before committing, but must ultimately call exactly one decision
#     tool (pass_through/override_sort/override_press/override_both).
#   - No ground-truth fault info is ever included in the snapshot - see
#     format_state_snapshot(). This mirrors rule_based_detect.py /
#     supervisor.py so the 4-arm comparison in the proposal (RQ1) stays
#     apples-to-apples: every non-oracle arm decides WHETHER/WHERE to act
#     from the same observable signals.

import os
import json
import time

from dotenv import load_dotenv
load_dotenv()  # picks up OPENAI_API_KEY / MARL_LLM_MODEL / MARL_LLM_BASE_URL from a local .env file, if present

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Any

DEFAULT_BASE_URL = os.environ.get("MARL_LLM_BASE_URL")  # e.g. http://localhost:3001/v1 for FreeLLMAPI-style proxies; unset = OpenAI's default endpoint
# "auto" (FreeLLMAPI's router-picks-for-you sentinel) only makes sense once a
# proxy base_url is actually set; plain OpenAI needs a real model name.
DEFAULT_MODEL = os.environ.get("MARL_LLM_MODEL") or ("auto" if DEFAULT_BASE_URL else "gpt-4o-mini")
MAX_TOOL_TURNS = 2          # cap on LLM<->tool round-trips per decision
REQUEST_TIMEOUT_S = 20       # per-call timeout
MATERIALS = ["A", "B", "C", "D", "E"]  # index -> mat_id, matches env_super.material_names + waste "E"


# ---------------------------------------------------------*/
# Decision object returned to the recovery mixin. Every path through the
# graph resolves to exactly one of these (falling back to pass_through on
# any failure - see build_llm_decision() at the bottom).
# ---------------------------------------------------------*/
class RecoveryDecision(TypedDict):
    action: str                    # "pass_through" | "override_sort" | "override_press" | "override_both"
    sort_mode: Optional[int]       # 0 | 1, only when action touches sort
    press_id: Optional[int]        # 1 | 2, only when action touches press
    material: Optional[str]        # one of MATERIALS, only when action touches press
    reasoning: Optional[str]       # short natural-language justification, for the decision log
    raw_tool_calls: List[dict]     # for debugging / the qualitative analysis section
    latency_ms: float
    error: Optional[str]           # set when a fallback was used


# ---------------------------------------------------------*/
# Tools — mirror the env's real control interface (env_super.py:
# press_action_to_discrete / sorting_rules-adjacent semantics), NOT the
# supervisor's fixed heuristic. The LLM commits to a specific plan, it
# doesn't just flip a "use the heuristic" switch.
# ---------------------------------------------------------*/
@tool
def pass_through() -> str:
    """No intervention needed this step - let the trained agents' own
    actions through unmodified. Use this when the system looks healthy,
    or once you judge a prior intervention is no longer necessary."""
    return "pass_through selected"


@tool
def override_sort(sort_mode: int) -> str:
    """Override the sorting agent's action for this step.
    sort_mode: 0 boosts materials A and C's sorting accuracy, 1 boosts
    materials B and D's sorting accuracy. Pick whichever matches the
    currently dominant material group on the belt."""
    return f"override_sort({sort_mode}) selected"


@tool
def override_press(press_id: int, material: str) -> str:
    """Override the pressing agent's action for this step: send a
    specific container to a specific free press.
    press_id: 1 or 2 - must be a press that is currently free (timer==0).
    material: one of "A","B","C","D","E" (E is the waste/reject container)
    - should be whichever container is fullest and worth pressing now."""
    return f"override_press({press_id}, {material}) selected"


@tool
def override_both(sort_mode: int, press_id: int, material: str) -> str:
    """Override BOTH the sorting and pressing actions this step in one
    call - use when both channels look compromised simultaneously."""
    return f"override_both({sort_mode}, {press_id}, {material}) selected"


DECISION_TOOLS = [pass_through, override_sort, override_press, override_both]
DECISION_TOOL_NAMES = {t.name for t in DECISION_TOOLS}


# ---------------------------------------------------------*/
# State snapshot — observable-only, no fault_context. Called once per LLM
# invocation (i.e. once per detector trigger, not every env step - see
# llm_replanning.py's gating).
# ---------------------------------------------------------*/
def format_state_snapshot(
    sort_obs, press_obs, reward_history_window, step, max_steps,
    recent_decisions, detector_reason,
):
    """
    Renders the current observable system state as a JSON-serializable dict
    for the LLM. Deliberately mirrors get_supervisor_obs()'s inputs (same
    raw signals, same "no ground truth" constraint) but as labeled fields
    instead of a flat vector, since an LLM reasons far better over named
    fields than an opaque array.

    sort_obs / press_obs: raw arrays from env_super.get_sort_obs() /
        get_press_obs() (13 and 16 dims respectively - see those docstrings
        for field order).
    reward_history_window: list of (r_sort, r_press) tuples, most recent K.
    recent_decisions: list of up to 3 prior RecoveryDecision-shaped dicts
        (trimmed to action/sort_mode/press_id/material) - short memory so
        the agent doesn't flap between corrections every trigger.
    detector_reason: str, why the lightweight detector decided to invoke
        the LLM this time (e.g. "sort reward dropped below baseline",
        "press container D near-overflow") - NOT ground truth about which
        fault is active, just the observable trigger condition.
    """
    sort_obs = list(map(float, sort_obs))
    press_obs = list(map(float, press_obs))

    snapshot = {
        "episode_progress": round(step / max(max_steps, 1), 3),
        "sort_channel": {
            "belt_occupancy": round(sort_obs[0], 3),
            "belt_proportions_A_B_C_D": [round(v, 3) for v in sort_obs[1:5]],
            "sorting_accuracy_A_B_C_D": [round(v, 3) for v in sort_obs[5:9]],
            "purity_diff_A_B_C_D": [round(v, 3) for v in sort_obs[9:13]],
        },
        "press_channel": {
            "container_fill_ratio_A_B_C_D_E": [round(v, 3) for v in press_obs[0:5]],
            "sorter_stage_fill_A_B_C_D": [round(v, 3) for v in press_obs[10:14]],
            "press_1_busy_fraction": round(press_obs[14], 3),
            "press_2_busy_fraction": round(press_obs[15], 3),
        },
        "recent_reward_window": [
            {"r_sort": round(float(r_s), 3), "r_press": round(float(r_p), 3)}
            for (r_s, r_p) in reward_history_window
        ],
        "why_you_were_called": detector_reason,
        "your_recent_decisions": recent_decisions,
    }
    return snapshot


SYSTEM_PROMPT = """You are the recovery-decision agent for an industrial \
sorting + pressing plant controlled by two RL agents (a sort agent and a \
press agent). Something in the plant's recent behavior looked anomalous, \
which is why you were invoked - see "why_you_were_called" in the state \
snapshot for the specific observable signal that triggered you.

You do NOT know the ground-truth fault type. You only see what a real \
plant operator would see: belt/container state, recent rewards, and your \
own recent decisions. Decide, from that alone, whether to override the \
sort channel, the press channel, both, or neither (pass_through) this \
step.

Guidance:
- Sort: pick whichever sort_mode (0 boosts A/C, 1 boosts B/D) matches the \
  dominant material group in belt_proportions_A_B_C_D.
- Press: pick the fullest container in container_fill_ratio_A_B_C_D_E \
  (index 0=A,1=B,2=C,3=D,4=E) and send it to whichever press has \
  busy_fraction == 0 (free). If both presses are busy, or nothing is \
  worth pressing (all fill ratios near 0), prefer pass_through for the \
  press channel.
- Don't flap: if your recent decisions already show a sensible override \
  in place and the state looks like it's recovering, it's fine to keep \
  intervening; if things look healthy, use pass_through.
- Call exactly ONE decision tool (pass_through / override_sort / \
  override_press / override_both) and briefly explain your reasoning in \
  one short sentence before calling it.
"""


# ---------------------------------------------------------*/
# LangGraph state + graph
# ---------------------------------------------------------*/
class AgentState(TypedDict):
    messages: List[Any]
    turns: int
    decision: Optional[RecoveryDecision]


def _get_llm():
    from langchain_openai import ChatOpenAI
    kwargs = dict(
        model=DEFAULT_MODEL,
        temperature=0,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=3,
    )
    if DEFAULT_BASE_URL:
        kwargs["base_url"] = DEFAULT_BASE_URL
    return ChatOpenAI(**kwargs).bind_tools(DECISION_TOOLS)


def _llm_node(state: AgentState):
    llm = _get_llm()
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response], "turns": state["turns"] + 1}


def _tool_node(state: AgentState):
    """Executes tool calls. If a DECISION tool was called, resolve the
    final decision and stop the graph. Non-decision tool calls (none
    currently defined, but the hook is here for future extension, e.g. a
    read-only "get_more_detail" tool) would just get a ToolMessage back and
    loop, bounded by MAX_TOOL_TURNS."""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []

    new_messages = list(state["messages"])
    decision = None

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {}) or {}
        if name in DECISION_TOOL_NAMES:
            decision = _resolve_decision(name, args, last.content, tool_calls)
        new_messages.append(ToolMessage(
            content=f"{name} recorded", tool_call_id=call["id"],
        ))

    return {"messages": new_messages, "decision": decision, "turns": state["turns"]}


def _resolve_decision(name, args, reasoning_text, raw_tool_calls) -> RecoveryDecision:
    sort_mode = press_id = material = None
    if name == "override_sort":
        sort_mode = int(args.get("sort_mode", 0))
    elif name == "override_press":
        press_id = int(args.get("press_id", 0)) or None
        material = str(args.get("material", "")).upper() or None
    elif name == "override_both":
        sort_mode = int(args.get("sort_mode", 0))
        press_id = int(args.get("press_id", 0)) or None
        material = str(args.get("material", "")).upper() or None

    return RecoveryDecision(
        action=name, sort_mode=sort_mode, press_id=press_id, material=material,
        reasoning=(reasoning_text or "")[:400],
        raw_tool_calls=raw_tool_calls, latency_ms=0.0, error=None,
    )


def _route(state: AgentState):
    if state.get("decision") is not None:
        return END
    if state["turns"] >= MAX_TOOL_TURNS:
        return END  # forced stop -> caller treats missing decision as fallback
    return "llm"


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm", _llm_node)
    graph.add_node("tools", _tool_node)
    graph.set_entry_point("llm")
    graph.add_edge("llm", "tools")
    graph.add_conditional_edges("tools", _route, {"llm": "llm", END: END})
    return graph.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------*/
# Public entry point — called by LLMRecoveryMixin.recovery_hook() (via
# llm_replanning.py), only when the detector says the LLM should be
# consulted this step. Never raises: any failure resolves to a safe
# pass_through decision with `error` set, so an eval run can't die on a
# network hiccup / malformed tool call / missing API key.
# ---------------------------------------------------------*/
def get_recovery_plan(snapshot: dict) -> RecoveryDecision:
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback_decision("OPENAI_API_KEY not set")

    start = time.monotonic()
    try:
        state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(snapshot)),
            ],
            "turns": 0,
            "decision": None,
        }
        result = _graph().invoke(state)
        decision = result.get("decision")
        if decision is None:
            return _fallback_decision("no decision tool called within MAX_TOOL_TURNS")
        decision["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return decision
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
        decision = _fallback_decision(f"{type(exc).__name__}: {exc}")
        decision["latency_ms"] = round((time.monotonic() - start) * 1000, 1)
        return decision


def _fallback_decision(error: str) -> RecoveryDecision:
    return RecoveryDecision(
        action="pass_through", sort_mode=None, press_id=None, material=None,
        reasoning=None, raw_tool_calls=[], latency_ms=0.0, error=error,
    )

# -----------------------------------------------------------------------------*/