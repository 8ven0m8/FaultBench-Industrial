# ---------------------------------------------------------*\
# Title: LLM-Agent Replanning Recovery Mechanism
# ---------------------------------------------------------*/
#
# Recovery mechanism #4 (ToDo #5, bullet 4): "LLM-agent replanning — an LLM
# that reads structured system state and outputs a recovery plan via tool
# calls into the env's control interface."
#
# How this differs from the other three arms:
#   - rule_based.py (oracle)      : told ground truth via fault_context.
#   - rule_based_detect.py        : self-detects from observable signals,
#                                   corrects with a FIXED heuristic.
#   - supervisor.py                : self-detects implicitly (a trained PPO
#                                   policy), corrects by choosing among a
#                                   fixed 4-way intervention vocabulary that
#                                   still applies the SAME fixed heuristic.
#   - llm_replanning (this file)  : self-detects via the SAME detector as
#                                   rule_based_detect (reused directly, not
#                                   reimplemented — see LLMRecoveryMixin's
#                                   base class below), but the CORRECTION
#                                   itself is decided by an LLM with full
#                                   control over which sort_mode / which
#                                   press+material to use — not limited to
#                                   the fixed heuristic. This is the
#                                   "novel, agentic" arm the proposal calls
#                                   out, and gives RQ1/RQ2 a real basis for
#                                   comparison against fixed-heuristic
#                                   recovery rather than just re-deriving
#                                   the same fixed heuristic through a
#                                   different decision mechanism.
#
# Gating the LLM: an eval episode is ~200 steps, run across 5 fault types x
# multiple seeds. Calling a real LLM API every single step is neither
# affordable nor a realistic model of how a human/agentic operator would
# actually work. So the LLM is only actually invoked:
#   (a) on the step the detector's suspicion state CHANGES (onset of a new
#       suspected fault, or a channel clearing), and
#   (b) periodically (every LLM_REINVOKE_EVERY steps) while still
#       suspected, so the agent gets a chance to notice the fault has been
#       fixed instead of just always finding a new fault while its own
#       corrective isn't causing that to be observable.
# Between invocations, the LAST decision stays sticky (mirrors
# rule_based_detect's own hysteresis-driven staying power) and is simply
# re-applied to whichever channel is still suspected — a real "replan"
# doesn't mean a fresh conversation every tick.
#
# Same plug-in shape as every other recovery mechanism in this project:
#     class Env_<Fault>_LLM(LLMRecoveryMixin, Env_<Fault>): pass

from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault
from src.recovery.rule_based_detect import DetectionRequiredRuleBasedMixin
from src.recovery.llm_agent import format_state_snapshot, get_recovery_plan


class LLMRecoveryMixin(DetectionRequiredRuleBasedMixin):
    """
    LLM-agent replanning recovery. Inherits the detector (_detect /
    _log_detection_step / warm-up / hysteresis) directly from
    DetectionRequiredRuleBasedMixin so detection is identical across both
    arms — only the correction differs. Overrides recovery_hook() to call
    an LLM (via src/recovery/llm_agent.py's bounded LangGraph loop) instead
    of the fixed sorting_rules()/check_container_level() heuristic, gated
    so the API is only actually hit on suspicion-state changes or every
    LLM_REINVOKE_EVERY steps while still suspected (see module docstring).

    Combine with any fault class, mixin first, same as every other
    recovery mechanism here:
        class Env_ByzantineFault_LLM(LLMRecoveryMixin, Env_ByzantineFault):
            pass
    """

    LLM_REINVOKE_EVERY = 8       # steps between re-plans while still suspected
    LLM_REWARD_WINDOW = 10       # K entries of (r_sort, r_press) shown to the LLM
    LLM_DECISION_MEMORY = 3      # how many past decisions are shown to the LLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_decision_log = []       # per-invocation record (post-hoc metrics + qualitative analysis)
        self._llm_cached_decision = None  # sticky decision reused between invocations
        self._llm_last_invoked_step = None
        self._llm_prev_suspected = (False, False)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.llm_decision_log = []
        self._llm_cached_decision = None
        self._llm_last_invoked_step = None
        self._llm_prev_suspected = (False, False)
        return obs, info

    # ---------------------------------------------------------*/
    # Gating: should the LLM actually be called this step?
    # ---------------------------------------------------------*/
    def _should_invoke_llm(self, sort_suspected, press_suspected):
        if not (sort_suspected or press_suspected):
            return False
        state_changed = (sort_suspected, press_suspected) != self._llm_prev_suspected
        if state_changed:
            return True
        if self._llm_last_invoked_step is None:
            return True
        return (self.current_step - self._llm_last_invoked_step) >= self.LLM_REINVOKE_EVERY

    def _detector_reason(self, sort_suspected, press_suspected):
        parts = []
        if sort_suspected:
            parts.append("sort-channel reward has dropped below its warm-up baseline "
                          "(or stayed suspected via hysteresis)")
        if press_suspected:
            parts.append("press-channel reward has dropped below its warm-up baseline "
                          "(or stayed suspected via hysteresis)")
        return "; ".join(parts) if parts else "periodic re-plan while still suspected"

    def _recent_decisions_for_prompt(self):
        trimmed = []
        for entry in self.llm_decision_log[-self.LLM_DECISION_MEMORY:]:
            trimmed.append({
                "step": entry["step"], "action": entry["action"],
                "sort_mode": entry["sort_mode"], "press_id": entry["press_id"],
                "material": entry["material"],
            })
        return trimmed

    # ---------------------------------------------------------*/
    # Apply a resolved decision to (sort_mode, press_action_discrete),
    # validating the press target the same way the env itself would
    # (busy press / insufficient material -> safe no-op), so a
    # hallucinated or stale plan can never crash or corrupt the run.
    # ---------------------------------------------------------*/
    def _apply_decision(self, sort_mode, press_action_discrete, decision):
        recovered_sort = sort_mode
        recovered_press = press_action_discrete

        if decision["action"] in ("override_sort", "override_both") and decision["sort_mode"] in (0, 1):
            recovered_sort = decision["sort_mode"]

        if decision["action"] in ("override_press", "override_both"):
            press_id = decision.get("press_id")
            material = decision.get("material")
            mat_id = None
            if material in self.material_names:
                mat_id = self.material_names.index(material)
            elif material == "E":
                mat_id = 4

            if press_id in (1, 2) and mat_id is not None and self.validate_press_action(press_id, mat_id):
                recovered_press = self.press_action_to_discrete(press_id, mat_id)
            else:
                # The LLM's specific press target is a one-shot action: once
                # it fires, that press is busy for the press's full duration
                # (12-15 steps), so re-applying the SAME stale (press_id,
                # material) on every following step - while the decision
                # stays sticky between LLM calls - would just keep getting
                # rejected here and silently degrade to a no-op for the rest
                # of the sticky window. That's effectively no recovery at
                # all on the press channel. Instead, fall back to a LIVE
                # re-check of the fullest container / free press (the same
                # check_container_level() rule_based/rule_based_detect call
                # every single step) so the press channel stays responsive
                # between LLM invocations. The LLM still owns WHETHER/WHEN
                # to intervene (that's the invocation-gating in
                # recovery_hook below); this only keeps execution alive once
                # its specific one-shot plan has already been carried out or
                # gone stale.
                press_id2, mat_id2 = self.check_container_level()
                recovered_press = (
                    self.press_action_to_discrete(press_id2, mat_id2) if press_id2 is not None else 0
                )

        return recovered_sort, recovered_press

    # ---------------------------------------------------------*/
    # Recovery hook
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        sort_suspected, press_suspected = self._detect()
        self._log_detection_step(sort_suspected, press_suspected, fault_context)

        if not (sort_suspected or press_suspected):
            self._llm_cached_decision = None
            self._llm_prev_suspected = (sort_suspected, press_suspected)
            return sort_mode, press_action_discrete

        if self._should_invoke_llm(sort_suspected, press_suspected):
            reward_history = self.reward_data.get("Reward", [])
            snapshot = format_state_snapshot(
                sort_obs=super().get_sort_obs(),
                press_obs=super().get_press_obs(),
                reward_history_window=reward_history[-self.LLM_REWARD_WINDOW:],
                step=self.current_step, max_steps=self.max_steps,
                recent_decisions=self._recent_decisions_for_prompt(),
                detector_reason=self._detector_reason(sort_suspected, press_suspected),
            )
            decision = get_recovery_plan(snapshot)
            self._llm_cached_decision = decision
            self._llm_last_invoked_step = self.current_step
            self.llm_decision_log.append({
                "step": self.current_step, "action": decision["action"],
                "sort_mode": decision["sort_mode"], "press_id": decision["press_id"],
                "material": decision["material"], "reasoning": decision["reasoning"],
                "latency_ms": decision["latency_ms"], "error": decision["error"],
                "sort_suspected": sort_suspected, "press_suspected": press_suspected,
            })

        self._llm_prev_suspected = (sort_suspected, press_suspected)

        decision = self._llm_cached_decision
        if decision is None:
            return sort_mode, press_action_discrete

        return self._apply_decision(sort_mode, press_action_discrete, decision)


# ---------------------------------------------------------*/
# One-line combo classes — same wiring footprint as the other three
# mechanisms.
# ---------------------------------------------------------*/
class Env_SensorNoiseFault_LLM(LLMRecoveryMixin, Env_SensorNoiseFault):
    pass


class Env_ActuatorDegradationFault_LLM(LLMRecoveryMixin, Env_ActuatorDegradationFault):
    pass


class Env_AgentDropoutFault_LLM(LLMRecoveryMixin, Env_AgentDropoutFault):
    pass


class Env_CommsLossFault_LLM(LLMRecoveryMixin, Env_CommsLossFault):
    pass


class Env_ByzantineFault_LLM(LLMRecoveryMixin, Env_ByzantineFault):
    pass


# Maps fault-type name -> the fault class combined with LLM-replanning
# recovery. main.py looks up FAULT_MODE in here when the person selects
# "llm_replanning".
LLM_REGISTRY = {
    "sensor_noise": Env_SensorNoiseFault_LLM,
    "actuator_degradation": Env_ActuatorDegradationFault_LLM,
    "agent_dropout": Env_AgentDropoutFault_LLM,
    "comms_loss": Env_CommsLossFault_LLM,
    "byzantine": Env_ByzantineFault_LLM,
}

# -----------------------------------------------------------------------------*/