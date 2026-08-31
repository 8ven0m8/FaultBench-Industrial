# ---------------------------------------------------------*\
# Title: Rule-Based Reconfiguration Recovery Mechanism
# ---------------------------------------------------------*/
#
# Baseline recovery mechanism per the proposal (ToDo #5, first bullet):
#   "Rule-based reconfiguration — fixed heuristics (e.g., redistribute a
#    dropped agent's tasks by a set rule). Build this first as your baseline."
#
# How it plugs in: every env_fault_*.py module now calls
# `self.recovery_hook(sort_mode, press_action_discrete, fault_context)`
# every step, right before the (possibly faulty/adversarial/degraded)
# action pair is actually applied to the environment. By default that
# hook is a no-op passthrough baked into each fault class. RuleBasedRecoveryMixin
# below overrides it, so combining the mixin with ANY fault class turns
# on rule-based recovery for that fault - no per-fault-type recovery
# logic needs to be written, and no 5x4 explosion of hand-written classes
# is needed for future recovery mechanisms either (each one is just
# another mixin + these same 5 one-line combo classes).
#
# fault_context (identical shape across all 5 fault modules):
#   {"active": bool, "fault_type": str|None, "sort_affected": bool, "press_affected": bool}
# "active" is the overall fault-active flag; "sort_affected"/"press_affected"
# tell you WHICH of the two decisions this step is actually compromised
# (e.g. comms_loss's "sort_affected" is always False - it never touches
# the sort agent - so recovery correctly never touches sort actions for
# that fault type either).

from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault


class RuleBasedRecoveryMixin:
    """
    Fixed-heuristic recovery: whichever channel (sort and/or press) is
    currently compromised gets overridden with the SAME rule-based
    heuristics already used elsewhere in this project as the non-RL
    reference policy (see Env_Super):
      - sort  -> sorting_rules(): boosts whichever of the A/C or B/D pair
                 is currently the dominant material group on the belt.
                 This is a stateless, single-step-lookahead rule - it has
                 no memory of the fault and simply reapplies "what would
                 a sensible fixed rule do right now", which is exactly
                 the ToDo's "redistribute ... by a set rule" spirit.
      - press -> check_container_level(): presses whichever container is
                 currently fullest, on whichever press is currently free.

    An untouched channel (fault_context[...] is False, including when no
    fault is active at all) passes the agent's own chosen action straight
    through - recovery only intervenes on exactly what's broken this step,
    the same way a real rule-based reconfigurer would only reroute the
    specific task that's actually failing rather than overriding a
    healthy agent's decisions too.

    Also logs a per-step detection_log entry with the SAME schema as
    DetectionRequiredRuleBasedMixin (rule_based_detect.py) - since this
    mechanism is an oracle, detected_*_active is trivially identical to
    true_*_active by construction (it never misses/over-triggers), but
    keeping the schema identical across all four recovery mechanisms lets
    utils/metrics.py score every mechanism uniformly, including this one
    as the "perfect detector" reference point (its false-recovery rate
    should always compute to exactly 0.0 - a useful sanity check on the
    metrics code itself).

    Usage - combine with any fault class, mixin FIRST so Python's MRO
    finds this recovery_hook() before the fault class's own no-op default:
        class Env_AgentDropoutFault_RuleBased(RuleBasedRecoveryMixin, Env_AgentDropoutFault):
            pass
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_log = []  # per-step record for post-hoc scoring (utils/metrics.py)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.detection_log = []
        return obs, info

    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        self.detection_log.append({
            "step": self.current_step,
            "detected_sort_active": bool(fault_context["active"] and fault_context["sort_affected"]),
            "detected_press_active": bool(fault_context["active"] and fault_context["press_affected"]),
            "true_sort_active": fault_context["sort_affected"],
            "true_press_active": fault_context["press_affected"],
        })

        if not fault_context["active"]:
            return sort_mode, press_action_discrete

        recovered_sort_mode = sort_mode
        recovered_press_action = press_action_discrete

        if fault_context["sort_affected"]:
            recovered_sort_mode = self.sorting_rules()

        if fault_context["press_affected"]:
            press_id, mat_id = self.check_container_level()
            recovered_press_action = (
                self.press_action_to_discrete(press_id, mat_id)
                if press_id is not None else 0  # no press free / nothing worth pressing -> no-op
            )

        return recovered_sort_mode, recovered_press_action


# ---------------------------------------------------------*/
# One-line combo classes - all the "wiring" needed per fault type.
# ---------------------------------------------------------*/
class Env_SensorNoiseFault_RuleBased(RuleBasedRecoveryMixin, Env_SensorNoiseFault):
    pass


class Env_ActuatorDegradationFault_RuleBased(RuleBasedRecoveryMixin, Env_ActuatorDegradationFault):
    pass


class Env_AgentDropoutFault_RuleBased(RuleBasedRecoveryMixin, Env_AgentDropoutFault):
    pass


class Env_CommsLossFault_RuleBased(RuleBasedRecoveryMixin, Env_CommsLossFault):
    pass


class Env_ByzantineFault_RuleBased(RuleBasedRecoveryMixin, Env_ByzantineFault):
    pass


# Maps fault-type name -> the fault class combined with rule-based recovery.
# main.py looks up FAULT_MODE in here when the person selects "rule_based".
RULE_BASED_REGISTRY = {
    "sensor_noise": Env_SensorNoiseFault_RuleBased,
    "actuator_degradation": Env_ActuatorDegradationFault_RuleBased,
    "agent_dropout": Env_AgentDropoutFault_RuleBased,
    "comms_loss": Env_CommsLossFault_RuleBased,
    "byzantine": Env_ByzantineFault_RuleBased,
}

# -----------------------------------------------------------------------------*/