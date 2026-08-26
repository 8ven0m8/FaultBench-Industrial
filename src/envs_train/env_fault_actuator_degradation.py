# ---------------------------------------------------------*\
# Title: Actuator Degradation Fault Injection for Env_Combined
# ---------------------------------------------------------*/
#
# Same MRO caveat as env_fault_sensor_noise.py: Env_Combined.step() calls
# `super().get_sort_obs()` / `super().get_press_obs()` explicitly, so
# Python's super() resolution would skip any override placed in a
# subclass of Env_Combined. We therefore override step() itself here too,
# mirroring Env_Combined's logic exactly, and degrade the *action* right
# after each agent's .predict() call and before it's applied to the
# actuator — i.e. the agent still perceives the true state and still
# decides its intended action, but what actually reaches the sorter/press
# hardware is what's degraded. This is the key difference from sensor
# noise (which corrupts perception, not execution).

import numpy as np
from src.envs_train.env_combined import Env_Combined


class Env_ActuatorDegradationFault(Env_Combined):
    """
    Drop-in replacement for Env_Combined that degrades the sorting and/or
    pressing actuator, starting at a (possibly randomized) step and
    lasting either persistently or for a fixed duration.

    Three degradation modes are supported:
      - "stuck"    : the actuator seizes at a fixed value once the fault
                     is active. Sort mode freezes at whatever it was on
                     the fault-onset step (or a configured value); the
                     press freezes at a configured discrete action
                     (default: no-op, i.e. it stops pressing entirely).
                     Models a jammed/seized actuator.
      - "restrict" : one option becomes permanently unreachable while the
                     fault is active. A configured sort mode is disabled
                     (forced to the other mode), and/or a configured press
                     (1 or 2) is disabled (any command routed to it is
                     downgraded to a no-op). Models a partially broken
                     actuator that still works, just with reduced range.
      - "slip"     : each faulty step, with probability `degradation_prob`,
                     the executed action differs from the agent's intended
                     action (sort mode flips; press action drops to no-op)
                     instead of failing deterministically every step.
                     Models wear/intermittent actuator failure rather than
                     a hard fault.

    fault_config keys:
        injection_step        : int or None (fixed step; None -> randomize)
        injection_step_range  : (lo, hi) used when injection_step is None
        duration               : int or None (None = persistent for rest of episode)
        target                  : "sort", "press", or "both"
        mode                    : "stuck" | "restrict" | "slip" (default "stuck")
        seed                    : int or None, for the degradation RNG (used by "slip")

        # "stuck" mode
        stuck_sort_mode         : int (0/1) or None -> freeze at fault-onset value
        stuck_press_action      : int in [0, 10], default 0 (no-op / press seizes idle)

        # "restrict" mode
        restrict_sort_mode      : int (0/1) or None -> this mode becomes unreachable
        restrict_press_id       : int (1/2) or None -> this press becomes unreachable

        # "slip" mode
        degradation_prob        : float in [0, 1], default 0.3
    """

    def __init__(self, *args, fault_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_config = fault_config or {}
        self._fault_rng = np.random.default_rng(self.fault_config.get("seed"))
        self._injection_step = None
        self._duration = None
        self.fault_log = []

        # Latched state for "stuck" mode - set on the first faulty step,
        # then held fixed for the remainder of the fault window.
        self._stuck_sort_value = None
        self._stuck_press_value = None

    # ---------------------------------------------------------*/
    # Reset — pick this episode's injection step, clear latched state
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

        self._stuck_sort_value = None
        self._stuck_press_value = None

        info = dict(info)
        info["fault_injection_step"] = self._injection_step
        info["fault_duration"] = self._duration
        return obs, info

    # ---------------------------------------------------------*/
    # Active-window helper (identical logic to env_fault_sensor_noise.py)
    # ---------------------------------------------------------*/
    def _fault_active_this_step(self):
        step = self.current_step  # not yet incremented at the point we check
        past_injection = step >= self._injection_step
        if self._duration is None:
            return past_injection
        return past_injection and (step < self._injection_step + self._duration)

    # ---------------------------------------------------------*/
    # Degradation helpers — take the agent's intended action, return what
    # actually reaches the actuator.
    # ---------------------------------------------------------*/
    def _degrade_sort_mode(self, sort_mode):
        mode = self.fault_config.get("mode", "stuck")

        if mode == "stuck":
            if self._stuck_sort_value is None:
                configured = self.fault_config.get("stuck_sort_mode")
                self._stuck_sort_value = int(configured) if configured is not None else int(sort_mode)
            return self._stuck_sort_value

        if mode == "restrict":
            restricted = self.fault_config.get("restrict_sort_mode")
            if restricted is not None and int(sort_mode) == int(restricted):
                return 1 - int(restricted)  # only 2 sort modes (0/1) exist
            return sort_mode

        if mode == "slip":
            prob = self.fault_config.get("degradation_prob", 0.3)
            if self._fault_rng.random() < prob:
                return 1 - int(sort_mode)  # actuator flips to the other mode
            return sort_mode

        return sort_mode

    def _degrade_press_action(self, press_action_discrete):
        mode = self.fault_config.get("mode", "stuck")

        if mode == "stuck":
            if self._stuck_press_value is None:
                self._stuck_press_value = int(self.fault_config.get("stuck_press_action", 0))
            return self._stuck_press_value

        if mode == "restrict":
            broken_press_id = self.fault_config.get("restrict_press_id")
            if broken_press_id is not None:
                press_id, _ = super().press_discrete_to_action(press_action_discrete)
                if press_id == int(broken_press_id):
                    return 0  # commanded press is down -> action is downgraded to no-op
            return press_action_discrete

        if mode == "slip":
            prob = self.fault_config.get("degradation_prob", 0.3)
            if self._fault_rng.random() < prob:
                return 0  # intended press job drops to no-op this step
            return press_action_discrete

        return press_action_discrete

    # ---------------------------------------------------------*/
    # Step — mirrors Env_Combined.step(), with degradation applied to the
    # chosen action right after .predict() and before it's applied to the
    # actuator (super().set_multisensor_mode / super().press_action_rules).
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        """No-op by default; override in a recovery mixin. See
        env_fault_sensor_noise.py's recovery_hook() docstring for the
        shared fault_context shape used across all 5 fault modules."""
        return sort_mode, press_action_discrete

    def step(self, action=None, mode="model", check_overflow=False):
        active = self._fault_active_this_step()
        target = self.fault_config.get("target", "both")
        degrade_sort = active and target in ("sort", "both")
        degrade_press = active and target in ("press", "both")

        if active and self.current_step == self._injection_step:
            self.fault_log.append({
                "onset_step": self._injection_step,
                "type": "actuator_degradation",
                "mode": self.fault_config.get("mode", "stuck"),
                "target": target,
            })

        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting Action (perception is unaffected, only execution) ---
        if self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            intended_sort_mode = int(predicted_sort_mode)
        else:
            intended_sort_mode = self.rng_sorting.choice([0, 1])
        sort_mode = self._degrade_sort_mode(intended_sort_mode) if degrade_sort else intended_sort_mode

        # --- 2. Pressing Action (perception is unaffected, only execution) ---
        if self.press_agent is not None:
            press_obs = super().get_press_obs()
            predicted_press_action, _ = self.press_agent.predict(press_obs, deterministic=True)
            intended_press_action = int(predicted_press_action)
        else:
            intended_press_action = self.rng_pressing.choice(11)
        press_action_discrete = (
            self._degrade_press_action(intended_press_action) if degrade_press else intended_press_action
        )

        fault_context = {
            "active": active, "fault_type": "actuator_degradation" if active else None,
            "sort_affected": degrade_sort, "press_affected": degrade_press,
        }
        sort_mode, press_action_discrete = self.recovery_hook(sort_mode, press_action_discrete, fault_context)

        chosen_flat_action = int(sort_mode) * 11 + int(press_action_discrete)

        # --- 3. Apply the (possibly degraded) Actions ---
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
                    "fault_active": active, "fault_type": "actuator_degradation" if active else None,
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
            "fault_type": "actuator_degradation" if active else None,
            "intended_sort_mode": intended_sort_mode, "executed_sort_mode": int(sort_mode),
            "intended_press_action": intended_press_action, "executed_press_action": int(press_action_discrete),
        }
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/