# ---------------------------------------------------------*\
# Title: Rule-Based Reconfiguration Recovery Mechanism — Detection-Required
# ---------------------------------------------------------*/
#
# Same corrective actions as src/recovery/rule_based.py's oracle mixin
# (sorting_rules() for sort, check_container_level() for press), but with
# one deliberate difference: it is NEVER told fault_context["active"] /
# ["sort_affected"] / ["press_affected"] to decide WHETHER or WHERE to
# intervene. It has to notice trouble itself, from a signal a real operator
# could actually observe:
#   - the reward the system actually logged last step (self.reward_data),
#     compared against a baseline established during a warm-up window
#
# (An earlier version also watched live container fill ratios, on the
# theory that a starving container is dangerous even before it drags the
# reward average down. Measured against the trained agents on fault-free
# episodes, fill ratio routinely sits above 0.85 for many consecutive
# steps as normal fill/press cycling, with no discriminative power over
# real faults - it made this mechanism false-positive on the press
# channel ~70% of a typical episode and score below the no-recovery
# baseline. Removed; see DetectionRequiredRuleBasedMixin's docstring.)
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
         ROLLING_WINDOW logged steps is compared against the baseline. A
         drop bigger than the per-channel threshold marks it "suspected".
         The threshold is the larger of a fixed fraction of the baseline
         AND STD_MULTIPLIER times the channel's own warm-up reward std -
         see below for why the std term is load-bearing, not decorative.

         (A proactive "container fill ratio above 0.85" trigger used to
         live here too, on the theory that a starving container is
         dangerous even before it drags the reward average down. Measured
         against the actual trained agents on fault-free episodes, fill
         ratio routinely reaches 0.85-1.1 and stays there for 3-20+
         consecutive steps as normal fill/press cycling - it has no
         discriminative power over real faults, and with MIN_STICKY_STEPS
         hysteresis it kept the press channel falsely "suspected" ~70% of
         a typical episode, hijacking a correctly-trained press policy
         with the crude check_container_level() heuristic almost
         continuously. Removed; detection relies solely on the reward-drop
         signal now.)

         (The reward-drop threshold itself was ALSO under-calibrated for
         the press channel specifically: press_reward has a near-zero
         warm-up mean (~0.15) but a large natural std (~0.3-0.35, full
         range -1..1, since it's a sparse/bursty per-press signal, not a
         smooth per-step one) - a fraction-of-mean threshold on a
         near-zero mean collapses to MIN_ABS_DROP, which is tiny next to
         that natural volatility. Measured on fault-free episodes, the
         rolling-mean press reward can dip up to ~0.85 below its own
         warm-up baseline with zero fault present, comparable to or larger
         than some real fault effects - a fixed fraction-of-mean threshold
         was flagging the press channel "suspected" on 60-95% of steps
         regardless of whether any fault was active. STD_MULTIPLIER anchors
         the threshold to that channel's own measured noise floor instead,
         which sort_reward (low-variance, mean ~0.4-0.5) barely changes -
         its threshold is still governed by SORT_REWARD_DROP_FRAC - but
         which pulls press_reward's effective threshold up to roughly
         where its noise ceiling actually sits. This trades away recall on
         the weakest configured faults (e.g. default sensor_noise,
         noise_std=0.1, whose ~0.45 average press-reward drop is itself
         smaller than the noise ceiling - no reward-only detector can
         cleanly separate that from natural variance) in exchange for
         collapsing false positives on healthy episodes; it reliably still
         catches the more severe fault types (agent_dropout /
         actuator_degradation "stuck" pin press reward at the reward
         floor, a ~1.15 drop, well clear of the noise ceiling). Worth
         reporting as a real detector limitation in the writeup, not
         silently smoothing over it.)
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
    STD_MULTIPLIER = 3.0
    MIN_ABS_DROP = 0.03
    MIN_STICKY_STEPS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_baseline = None
        self._press_baseline = None
        self._sort_std = None
        self._press_std = None
        self._sort_cooldown = 0
        self._press_cooldown = 0
        self.detection_log = []  # per-step record for post-hoc scoring, see _log_detection_step()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sort_baseline = None
        self._press_baseline = None
        self._sort_std = None
        self._press_std = None
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

    @staticmethod
    def _pstdev(values, mean):
        if not values:
            return 0.0
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    def _detect(self):
        reward_history = self.reward_data.get("Reward", [])  # list of (r_sort, r_press) tuples

        if len(reward_history) < self.BASELINE_WARMUP_STEPS:
            return False, False  # not enough history yet to trust a baseline

        if self._sort_baseline is None:
            warmup = reward_history[:self.BASELINE_WARMUP_STEPS]
            warmup_sort = [r[0] for r in warmup]
            warmup_press = [r[1] for r in warmup]
            self._sort_baseline = sum(warmup_sort) / len(warmup_sort)
            self._press_baseline = sum(warmup_press) / len(warmup_press)
            self._sort_std = self._pstdev(warmup_sort, self._sort_baseline)
            self._press_std = self._pstdev(warmup_press, self._press_baseline)

        recent_sort = self._rolling_mean([r[0] for r in reward_history], self.ROLLING_WINDOW)
        recent_press = self._rolling_mean([r[1] for r in reward_history], self.ROLLING_WINDOW)

        # Threshold is whichever is larger: a fraction of the baseline, or a
        # multiple of the channel's own warm-up noise. The std term is what
        # actually protects a near-zero-mean, high-variance channel like
        # press_reward from tripping on its own natural volatility - see
        # the class docstring for the measured false-positive rates this
        # fixes.
        sort_threshold = max(
            abs(self._sort_baseline) * self.SORT_REWARD_DROP_FRAC,
            self._sort_std * self.STD_MULTIPLIER,
            self.MIN_ABS_DROP,
        )
        press_threshold = max(
            abs(self._press_baseline) * self.PRESS_REWARD_DROP_FRAC,
            self._press_std * self.STD_MULTIPLIER,
            self.MIN_ABS_DROP,
        )
        sort_signal = (self._sort_baseline - recent_sort) > sort_threshold
        press_signal = (self._press_baseline - recent_press) > press_threshold

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