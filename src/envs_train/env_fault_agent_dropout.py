# ---------------------------------------------------------*\
# Title: Agent Dropout Fault Injection for Env_Combined
# ---------------------------------------------------------*/
#
# Same MRO caveat as env_fault_sensor_noise.py / env_fault_actuator_degradation.py:
# Env_Combined.step() calls `super().get_sort_obs()` / `super().get_press_obs()`
# explicitly, so we override step() itself here too, mirroring Env_Combined's
# logic exactly.
#
# Conceptual difference from actuator_degradation:
#   - actuator_degradation ("stuck") = the agent is still deciding normally,
#     but its command never reaches the actuator correctly (hardware jam).
#     Absent a configured value, it freezes at whatever the actuator
#     happened to be doing when the fault hit.
#   - agent_dropout = the agent itself is gone (crashed process, lost
#     connection, disconnected node). There is no decision to execute at
#     all, so nothing calls .predict() for the dropped-out agent while the
#     fault is active, and the environment falls back to a fixed, externally
#     configured failsafe action — independent of whatever the system state
#     was at the moment of dropout. This models a hard-coded safe default
#     kicking in when a controller goes silent, rather than a jammed
#     actuator repeating its last command.

import numpy as np
from src.envs_train.env_combined import Env_Combined


class Env_AgentDropoutFault(Env_Combined):
    """
    Drop-in replacement for Env_Combined that zeroes out the sorting and/or
    pressing agent's participation entirely, starting at a (possibly
    randomized) step and lasting either persistently or for a fixed
    duration.

    While an agent is "dropped out":
      - `.predict()` is never called for it (it is not consulted at all).
      - The environment substitutes a fixed failsafe action for that
        agent's decision every step, for as long as the dropout is active.

    fault_config keys:
        injection_step        : int or None (fixed step; None -> randomize)
        injection_step_range  : (lo, hi) used when injection_step is None
        duration               : int or None (None = persistent for rest of episode)
        target                  : "sort", "press", or "both"
        seed                    : int or None, for the injection-step RNG

        dropout_sort_mode       : int (0/1), default 0 - failsafe sort mode
                                   used every step the sort agent is dropped
        dropout_press_action    : int in [0, 10], default 0 (no-op) -
                                   failsafe press action used every step the
                                   press agent is dropped
    """

    def __init__(self, *args, fault_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_config = fault_config or {}
        self._fault_rng = np.random.default_rng(self.fault_config.get("seed"))
        self._injection_step = None
        self._duration = None
        self.fault_log = []

    # ---------------------------------------------------------*/
    # Reset — pick this episode's injection step
    # ---------------------------------------------------------*/
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self.fault_config.get("injection_step") is not None:
            self._injection_step = self.fault_config["injection_step"]
        else:
            lo, hi = self.fault_config.get("injection_step_range", (40, 60))
            self._injection_step = int(self._fault_rng.integers(lo, hi + 1))

        # Resolve this episode's duration: fixed "duration" wins if set,
        # else a "duration_range" is sampled per-episode (mirrors how
        # injection_step_range works above), else the fault is persistent.
        if self.fault_config.get("duration") is not None:
            self._duration = self.fault_config["duration"]
        elif self.fault_config.get("duration_range") is not None:
            d_lo, d_hi = self.fault_config["duration_range"]
            self._duration = int(self._fault_rng.integers(d_lo, d_hi + 1))
        else:
            self._duration = None  # persistent for rest of episode

        info = dict(info)
        info["fault_injection_step"] = self._injection_step
        info["fault_duration"] = self._duration
        return obs, info

    # ---------------------------------------------------------*/
    # Active-window helper (identical logic to the other fault modules)
    # ---------------------------------------------------------*/
    def _fault_active_this_step(self):
        step = self.current_step  # not yet incremented at the point we check
        past_injection = step >= self._injection_step
        if self._duration is None:
            return past_injection
        return past_injection and (step < self._injection_step + self._duration)

    # ---------------------------------------------------------*/
    # Step — mirrors Env_Combined.step(); while dropped out, an agent's
    # .predict() is skipped entirely and its action is replaced by the
    # configured failsafe value.
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        """No-op by default; override in a recovery mixin. See
        env_fault_sensor_noise.py's recovery_hook() docstring for the
        shared fault_context shape used across all 5 fault modules."""
        return sort_mode, press_action_discrete

    def step(self, action=None, mode="model", check_overflow=False):
        active = self._fault_active_this_step()
        target = self.fault_config.get("target", "both")
        sort_dropped = active and target in ("sort", "both")
        press_dropped = active and target in ("press", "both")

        if active and self.current_step == self._injection_step:
            self.fault_log.append({
                "onset_step": self._injection_step,
                "type": "agent_dropout",
                "target": target,
            })

        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting Action ---
        if sort_dropped:
            # Agent is not consulted at all - failsafe default is used.
            intended_sort_mode = None
            sort_mode = int(self.fault_config.get("dropout_sort_mode", 0))
        elif self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            intended_sort_mode = int(predicted_sort_mode)
            sort_mode = intended_sort_mode
        else:
            intended_sort_mode = self.rng_sorting.choice([0, 1])
            sort_mode = intended_sort_mode

        # --- 2. Pressing Action ---
        if press_dropped:
            # Agent is not consulted at all - failsafe default is used.
            intended_press_action = None
            press_action_discrete = int(self.fault_config.get("dropout_press_action", 0))
        elif self.press_agent is not None:
            press_obs = super().get_press_obs()
            predicted_press_action, _ = self.press_agent.predict(press_obs, deterministic=True)
            intended_press_action = int(predicted_press_action)
            press_action_discrete = intended_press_action
        else:
            intended_press_action = self.rng_pressing.choice(11)
            press_action_discrete = intended_press_action

        fault_context = {
            "active": active, "fault_type": "agent_dropout" if active else None,
            "sort_affected": sort_dropped, "press_affected": press_dropped,
        }
        sort_mode, press_action_discrete = self.recovery_hook(sort_mode, press_action_discrete, fault_context)

        chosen_flat_action = int(sort_mode) * 11 + int(press_action_discrete)

        # --- 3. Apply the Actions ---
        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        press_action_tuple = super().press_discrete_to_action(press_action_discrete)
        super().press_action_rules(
            (press_action_tuple[0], press_action_tuple[1]) if press_action_tuple[0] != 0 else (None, None)
        )

        # --- 4. Reward, Logging, Return ---
        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                reward = self.overflow_penalty
                info = {
                    "overflow": True, "overflow_material": mat, "action": chosen_flat_action,
                    "fault_active": active, "fault_type": "agent_dropout" if active else None,
                    "sort_dropped": sort_dropped, "press_dropped": press_dropped,
                    "intended_sort_mode": intended_sort_mode, "executed_sort_mode": int(sort_mode),
                    "intended_press_action": intended_press_action, "executed_press_action": int(press_action_discrete),
                }
                self.current_step += 1
                self._log_step_data(r_sort=reward / 2, r_press=reward / 2)
                return self.get_obs(), float(reward), True, False, info

        sort_reward = self.calculate_sorting_reward()
        press_reward = self.calculate_press_reward()
        reward = sort_reward + press_reward

        obs_next = self.get_obs()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        self._log_step_data(r_sort=sort_reward, r_press=press_reward)
        info = {
            "action": chosen_flat_action,
            "fault_active": active,
            "fault_type": "agent_dropout" if active else None,
            "sort_dropped": sort_dropped, "press_dropped": press_dropped,
            "intended_sort_mode": intended_sort_mode, "executed_sort_mode": int(sort_mode),
            "intended_press_action": intended_press_action, "executed_press_action": int(press_action_discrete),
        }
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/