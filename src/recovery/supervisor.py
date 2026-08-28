# ---------------------------------------------------------*\
# Title: Hierarchical Supervisor-Agent Recovery Mechanism
# ---------------------------------------------------------*/
#
# Recovery mechanism #3 (ToDo #5, bullet 3): "Hierarchical supervisor-agent
# — an extra agent that monitors others and intervenes (reassign tasks,
# restart, trigger replanning)."
#
# How this differs from the two rule-based arms the same fault suite is run
# against:
#   - rule_based.py (oracle)      : told ground truth via fault_context, and
#                                   corrects the affected channel with a
#                                   fixed heuristic.
#   - rule_based_detect.py        : must infer the fault itself from the same
#                                   observable signals any operator could see
#                                   (reward history, container levels), then
#                                   corrects with the SAME heuristic.
#   - supervisor (this file)      : an RL policy (PPO) is trained to map
#                                   observable system state -> intervention.
#                                   It makes the same WHEN/WHERE decision a
#                                   detector makes (is something wrong, and on
#                                   which channel?), but that decision is
#                                   LEARNED from data rather than hand-tuned
#                                   thresholds. This is what makes the
#                                   comparison head-to-head meaningful: all
#                                   three arms share the same corrective tool,
#                                   only the decision side differs.
#
# The corrective vocabulary is intentionally IDENTICAL to rule_based.py so
# the comparison isolates the decision side. The supervisor's Discrete(4)
# action space maps onto the proposal's "reassign tasks" language — the
# supervisor takes over control of an affected channel using the fixed, safe
# rule until it decides things have recovered:
#   action 0 = pass-through (no intervention, let the trained agents act)
#   action 1 = override the sort channel with sorting_rules()
#   action 2 = override the press channel with check_container_level()
#   action 3 = override both channels
#
# The supervisor observation (get_supervisor_obs) contains ONLY signals a
# real supervisor could observe — no ground-truth fault_context, matching
# rule_based_detect's constraints:
#   - true (uncorrupted) sort observation     (13)
#   - true press observation                  (16)
#   - rolling window of last logged (r_sort, r_press) rewards  (2 * K)
#   - normalized episode progress (current_step / max_steps)   (1)
#   - one-hot of the supervisor's OWN last decision            (4)
# The last-decision term gives the policy the memory of its own intervention
# ("I fixed it and the reward looks healthy — did the fault clear, or am I
# the reason it looks healthy?"), which lets it learn sticky behavior similar
# to rule_based_detect's hysteresis instead of flapping.
#
# Execution path: during TRAINING, the policy's chosen action is injected by
# the training wrapper (src/envs_train/env_supervisor_train.py) through the
# `_pending_supervisor_action` attribute and consumed here by recovery_hook().
# At EVAL time (main.py -> "supervisor" recovery + a fault), the combo env
# holds a trained supervisor model (set via set_supervisor()), and
# recovery_hook() derives the decision itself by calling
# self.supervisor_agent.predict() on get_supervisor_obs().
#
# Same mixin/combo pattern as rule_based.py: `class Env_<Fault>_Supervisor(
# SupervisorRecoveryMixin, Env_<Fault>): pass` plugs the supervisor into any
# fault class with zero per-fault-type recovery logic.

import numpy as np
import gymnasium as gym

from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault


class SupervisorRecoveryMixin:
    """Learned-intervention recovery: an RL policy decides whether/where to
    take over control of the sort/press channels using the same fixed
    heuristics the rule-based arms correct with.

    Combine with any fault class, mixin FIRST so Python's MRO finds this
    recovery_hook()/get_supervisor_obs() before the fault class's own no-op
    default:
        class Env_ByzantineFault_Supervisor(SupervisorRecoveryMixin, Env_ByzantineFault):
            pass
    """

    SUPERVISOR_REWARD_WINDOW = 10   # K: entries of (r_sort, r_press) in the obs window
    SORT_OBS_DIM = 13               # Env_Super.get_sort_obs() length
    PRESS_OBS_DIM = 16              # Env_Super.get_press_obs() length

    INTERVENTION_PASS_THROUGH = 0
    INTERVENTION_SORT_ONLY = 1
    INTERVENTION_PRESS_ONLY = 2
    INTERVENTION_BOTH = 3

    def __init__(self, *args, supervisor_agent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.supervisor_agent = supervisor_agent    # SB3 policy, or None at train time
        self._pending_supervisor_action = None      # injected by Env_SupervisorTraining wrapper
        self.supervisor_decision_log = []           # per-step record for post-hoc metrics

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.supervisor_decision_log = []
        self._pending_supervisor_action = None
        return obs, info

    def set_supervisor(self, model):
        self.supervisor_agent = model

    @classmethod
    def supervisor_observation_space(cls):
        """Observation space matching get_supervisor_obs(). Mirrors
        Env_Combined._initialize_spaces()'s bounds for the raw sort/press
        obs the supervisor re-reads."""
        k = cls.SUPERVISOR_REWARD_WINDOW
        n_sort = cls.SORT_OBS_DIM
        n_press = cls.PRESS_OBS_DIM
        n_last = 4  # one-hot of own last decision
        sort_low = np.concatenate([
            np.zeros(1), np.zeros(4), np.zeros(4),  # occupancy, proportions, accuracies
            np.full(4, -1.0),                        # purity differences can go negative
        ])
        low = np.concatenate([
            sort_low, np.zeros(n_press),
            np.full(2 * k, -1.0), [0.0], np.zeros(n_last),
        ]).astype(np.float32)
        high = np.ones(n_sort + n_press + 2 * k + 1 + n_last, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _last_supervisor_decision_onehot(self):
        onehot = np.zeros(4, dtype=np.float32)
        if self.supervisor_decision_log:
            onehot[self.supervisor_decision_log[-1]["decision"]] = 1.0
        else:
            onehot[0] = 1.0
        return onehot

    def get_supervisor_obs(self):
        """Observable-state observation only — true (uncorrupted) system
        state plus the recent reward history and own last decision. Never
        reads fault_context (same constraint as rule_based_detect)."""
        k = self.SUPERVISOR_REWARD_WINDOW

        sort_obs = np.asarray(super().get_sort_obs(), dtype=np.float32)
        press_obs = np.asarray(super().get_press_obs(), dtype=np.float32)

        reward_history = self.reward_data.get("Reward", [])  # list of (r_sort, r_press)
        recent = reward_history[-k:]
        flat = np.zeros(2 * k, dtype=np.float32)
        for i, (r_sort, r_press) in enumerate(recent):
            flat[2 * i] = r_sort
            flat[2 * i + 1] = r_press

        step_frac = np.array([self.current_step / max(self.max_steps, 1)], dtype=np.float32)

        return np.concatenate([
            sort_obs, press_obs, flat, step_frac, self._last_supervisor_decision_onehot(),
        ])

    def _apply_intervention(self, sort_mode, press_action_discrete, decision):
        recovered_sort = sort_mode
        recovered_press = press_action_discrete

        if decision in (self.INTERVENTION_SORT_ONLY, self.INTERVENTION_BOTH):
            recovered_sort = self.sorting_rules()

        if decision in (self.INTERVENTION_PRESS_ONLY, self.INTERVENTION_BOTH):
            press_id, mat_id = self.check_container_level()
            recovered_press = (
                self.press_action_to_discrete(press_id, mat_id) if press_id is not None else 0
            )

        self.supervisor_decision_log.append({
            "step": self.current_step, "decision": int(decision),
        })
        return recovered_sort, recovered_press

    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        """Decides the intervention. During training the decision comes from
        the training wrapper via `_pending_supervisor_action`; at eval it is
        produced by the attached supervisor policy from observable state."""
        decision = self._pending_supervisor_action
        self._pending_supervisor_action = None

        if decision is None and self.supervisor_agent is not None:
            prediction, _ = self.supervisor_agent.predict(
                self.get_supervisor_obs(), deterministic=True
            )
            decision = int(prediction)

        if decision is None:
            return sort_mode, press_action_discrete  # no model attached yet -> pass-through

        return self._apply_intervention(sort_mode, press_action_discrete, decision)


# ---------------------------------------------------------*/
# One-line combo classes — same wiring footprint as rule_based.py.
# ---------------------------------------------------------*/
class Env_SensorNoiseFault_Supervisor(SupervisorRecoveryMixin, Env_SensorNoiseFault):
    pass


class Env_ActuatorDegradationFault_Supervisor(SupervisorRecoveryMixin, Env_ActuatorDegradationFault):
    pass


class Env_AgentDropoutFault_Supervisor(SupervisorRecoveryMixin, Env_AgentDropoutFault):
    pass


class Env_CommsLossFault_Supervisor(SupervisorRecoveryMixin, Env_CommsLossFault):
    pass


class Env_ByzantineFault_Supervisor(SupervisorRecoveryMixin, Env_ByzantineFault):
    pass


# Maps fault-type name -> the fault class combined with supervisor recovery.
# main.py looks up FAULT_MODE in here when the person selects "supervisor".
SUPERVISOR_REGISTRY = {
    "sensor_noise": Env_SensorNoiseFault_Supervisor,
    "actuator_degradation": Env_ActuatorDegradationFault_Supervisor,
    "agent_dropout": Env_AgentDropoutFault_Supervisor,
    "comms_loss": Env_CommsLossFault_Supervisor,
    "byzantine": Env_ByzantineFault_Supervisor,
}

# -----------------------------------------------------------------------------*/