# ---------------------------------------------------------*\
# Title: Rule-Based Reconfiguration Recovery Mechanism — Detection-Required
# ---------------------------------------------------------*/
#
# Same corrective actions as src/recovery/rule_based.py's oracle mixin
# (sorting_rules() for sort, check_container_level() for press), but with
# one deliberate difference: it is NEVER told fault_context["active"] /
# ["sort_affected"] / ["press_affected"] to decide WHETHER or WHERE to
# intervene. It has to notice trouble itself, from signals a real operator
# could actually observe:
#   - the reward the system actually logged last step (self.reward_data),
#     compared against a baseline established during a warm-up window
#   - live container fill ratios (a supervisor would obviously watch tank
#     levels, not just a scalar reward)
#
# fault_context IS still read here - but ONLY after a detection decision
# has already been made, purely to log whether that decision was correct
# (true/false positive/negative). That bookkeeping is for computing
# detection accuracy / false-recovery-rate later (ToDo #6); it never
# feeds back into the decision itself. See _log_detection_step() - it's
# the only place fault_context is touched, and it never influences
# `_sort_suspected` / `_press_suspected`.
#
# Why not just call self.calculate_sorting_reward() / calculate_press_reward()
# directly inside recovery_hook() to get a "live" reading? Because
# recovery_hook() runs BEFORE this step's action is applied, and
# calculate_press_reward() has a side effect - it consumes and resets
# self._last_press_started/_last_press_amount. Calling it early would
# silently zero out the real bonus the environment computes later in the
# same step, corrupting the actual logged reward. Reading the *previous*
# step's already-logged reward from self.reward_data avoids this entirely,
# and also models a realistic 1-step detection lag.

from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault


class DetectionRequiredRuleBasedMixin:
    """
    Rule-based recovery that has to detect the fault itself before it can
    correct for it. Detection logic:

      1. Warm-up: the first BASELINE_WARMUP_STEPS steps of reward history
         are treated as "known healthy" and averaged into a baseline
         (self._sort_baseline, self._press_baseline). No intervention is
         possible before the baseline exists - this models a real
         monitor's calibration period, and conveniently the default fault
         injection window (steps 40-60) always starts after it.
      2. Live signal: every step, the rolling mean reward over the last
         ROLLING_WINDOW logged steps is compared against the baseline.
         A drop bigger than *_DROP_THRESHOLD on a channel marks it
         "suspected". Container fill ratio above FILL_RATIO_WARNING also
         marks the press channel suspected on its own, since a starving
         container is dangerous even if it hasn't dragged the rolling
         reward average down yet.
      3. Hysteresis: once a channel is suspected, it stays "in recovery"
         for at least MIN_STICKY_STEPS steps even if the signal clears
         for a step or two, to avoid flapping between corrected and
         uncorrected actions every other step (which would itself
         destabilize the system and pollute time-to-recovery numbers).

    Combine with any fault class, mixin first, same as RuleBasedRecoveryMixin:
        class Env_ByzantineFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_ByzantineFault):
            pass
    """

    BASELINE_WARMUP_STEPS = 30
    ROLLING_WINDOW = 5
    SORT_REWARD_DROP_FRAC = 0.30
    PRESS_REWARD_DROP_FRAC = 0.30
    FILL_RATIO_WARNING = 0.85
    MIN_ABS_DROP = 0.03
    MIN_STICKY_STEPS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_baseline = None
        self._press_baseline = None
        self._sort_cooldown = 0
        self._press_cooldown = 0
        self.detection_log = []  # per-step record for post-hoc scoring, see _log_detection_step()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sort_baseline = None
        self._press_baseline = None
        self._sort_cooldown = 0
        self._press_cooldown = 0
        self.detection_log = []
        return obs, info

    # ---------------------------------------------------------*/
    # Detection - reads only self.reward_data (already-logged past
    # rewards) and live container levels. Never reads fault_context.
    # ---------------------------------------------------------*/
    def _rolling_mean(self, values, window):
        recent = values[-window:]
        return sum(recent) / len(recent) if recent else 0.0

    def _detect(self):
        reward_history = self.reward_data.get("Reward", [])  # list of (r_sort, r_press) tuples

        if len(reward_history) < self.BASELINE_WARMUP_STEPS:
            return False, False  # not enough history yet to trust a baseline

        if self._sort_baseline is None:
            warmup = reward_history[:self.BASELINE_WARMUP_STEPS]
            self._sort_baseline = sum(r[0] for r in warmup) / len(warmup)
            self._press_baseline = sum(r[1] for r in warmup) / len(warmup)

        recent_sort = self._rolling_mean([r[0] for r in reward_history], self.ROLLING_WINDOW)
        recent_press = self._rolling_mean([r[1] for r in reward_history], self.ROLLING_WINDOW)

        sort_threshold = max(abs(self._sort_baseline) * self.SORT_REWARD_DROP_FRAC, self.MIN_ABS_DROP)
        press_threshold = max(abs(self._press_baseline) * self.PRESS_REWARD_DROP_FRAC, self.MIN_ABS_DROP)
        sort_signal = (self._sort_baseline - recent_sort) > sort_threshold
        press_signal = (self._press_baseline - recent_press) > press_threshold

        # Proactive fill-level check - pure reads, no side effects.
        for mat in self.material_names + ["E"]:
            level = self.container_materials.get(mat, 0) + self.container_materials.get(f"{mat}_False", 0)
            cap = self.container_max.get(mat, self.container_global_max)
            if cap > 0 and (level / cap) > self.FILL_RATIO_WARNING:
                press_signal = True
                break

        # Hysteresis: sticky once suspected.
        if sort_signal:
            self._sort_cooldown = self.MIN_STICKY_STEPS
        elif self._sort_cooldown > 0:
            self._sort_cooldown -= 1
        sort_suspected = self._sort_cooldown > 0

        if press_signal:
            self._press_cooldown = self.MIN_STICKY_STEPS
        elif self._press_cooldown > 0:
            self._press_cooldown -= 1
        press_suspected = self._press_cooldown > 0

        return sort_suspected, press_suspected

    def _log_detection_step(self, sort_suspected, press_suspected, fault_context):
        """Post-hoc bookkeeping only - never feeds back into the decision.
        Lets a later metrics pass score detection precision/recall and
        false-recovery rate against ground truth."""
        self.detection_log.append({
            "step": self.current_step,
            "detected_sort_active": sort_suspected,
            "detected_press_active": press_suspected,
            "true_sort_active": fault_context["sort_affected"],
            "true_press_active": fault_context["press_affected"],
        })

    # ---------------------------------------------------------*/
    # Recovery hook
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        sort_suspected, press_suspected = self._detect()
        self._log_detection_step(sort_suspected, press_suspected, fault_context)

        recovered_sort_mode = sort_mode
        recovered_press_action = press_action_discrete

        if sort_suspected:
            recovered_sort_mode = self.sorting_rules()

        if press_suspected:
            press_id, mat_id = self.check_container_level()
            recovered_press_action = (
                self.press_action_to_discrete(press_id, mat_id) if press_id is not None else 0
            )

        return recovered_sort_mode, recovered_press_action


# ---------------------------------------------------------*/
# Combo classes
# ---------------------------------------------------------*/
class Env_SensorNoiseFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_SensorNoiseFault):
    pass


class Env_ActuatorDegradationFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_ActuatorDegradationFault):
    pass


class Env_AgentDropoutFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_AgentDropoutFault):
    pass


class Env_CommsLossFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_CommsLossFault):
    pass


class Env_ByzantineFault_RuleBasedDetect(DetectionRequiredRuleBasedMixin, Env_ByzantineFault):
    pass


RULE_BASED_DETECT_REGISTRY = {
    "sensor_noise": Env_SensorNoiseFault_RuleBasedDetect,
    "actuator_degradation": Env_ActuatorDegradationFault_RuleBasedDetect,
    "agent_dropout": Env_AgentDropoutFault_RuleBasedDetect,
    "comms_loss": Env_CommsLossFault_RuleBasedDetect,
    "byzantine": Env_ByzantineFault_RuleBasedDetect,
}

# -----------------------------------------------------------------------------*/