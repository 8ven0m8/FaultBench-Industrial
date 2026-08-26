# ---------------------------------------------------------*\
# Title: Sensor Noise Fault Injection for Env_Combined
# ---------------------------------------------------------*/
#
# IMPORTANT: Env_Combined.step() calls `super().get_sort_obs()` /
# `super().get_press_obs()` explicitly (not `self.get_...`). Python's
# super() resolves statically starting *after* Env_Combined in the MRO,
# so overriding get_sort_obs/get_press_obs in a subclass of Env_Combined
# would silently never be called. We therefore override step() itself,
# mirroring its logic, and inject noise right before each agent's
# .predict() call — the exact point where the "sensor reading" is
# actually consumed.

import numpy as np
from src.envs_train.env_combined import Env_Combined


class Env_SensorNoiseFault(Env_Combined):
    """
    Drop-in replacement for Env_Combined that corrupts the sorting and/or
    pressing agent's observation with Gaussian noise, starting at a
    (possibly randomized) step and lasting either persistently or for a
    fixed duration.

    fault_config keys:
        injection_step        : int or None (fixed step; None -> randomize)
        injection_step_range  : (lo, hi) used when injection_step is None
        duration               : int or None (None = persistent for rest of episode)
        noise_std              : float, relative noise stddev (obs already in
                                  [-1,1] or [0,1], so this is directly comparable
                                  across both agents)
        target                  : "sort", "press", or "both"
        seed                    : int or None, for the noise RNG
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
    # Noise helpers — add noise, re-clip to the same bounds get_sort_obs/
    # get_press_obs already enforce, so corrupted values stay in-range for
    # the trained policy's input assumptions.
    # ---------------------------------------------------------*/
    def _noisy_sort_obs(self):
        obs = super().get_sort_obs()
        std = self.fault_config.get("noise_std", 0.1)
        obs = obs + self._fault_rng.normal(0.0, std, size=obs.shape).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _noisy_press_obs(self):
        obs = super().get_press_obs()
        std = self.fault_config.get("noise_std", 0.1)
        obs = obs + self._fault_rng.normal(0.0, std, size=obs.shape).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _fault_active_this_step(self):
        step = self.current_step  # not yet incremented at the point we check
        past_injection = step >= self._injection_step
        if self._duration is None:
            return past_injection
        return past_injection and (step < self._injection_step + self._duration)

    # ---------------------------------------------------------*/
    # Recovery hook — no-op by default. A recovery mechanism (rule-based,
    # supervisor, etc.) plugs in by overriding this in a mixin combined
    # with this class, e.g.:
    #     class Env_SensorNoiseFault_RuleBased(RuleBasedRecoveryMixin, Env_SensorNoiseFault): pass
    # Called every step, right before the (sort_mode, press_action_discrete)
    # pair is actually applied to the environment - whether or not a fault
    # is active this step. `fault_context` always has the same shape across
    # all 5 fault modules: {"active", "fault_type", "sort_affected", "press_affected"}.
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        return sort_mode, press_action_discrete

    # ---------------------------------------------------------*/
    # Step — mirrors Env_Combined.step(), with noise injected right
    # before each agent's .predict() call.
    # ---------------------------------------------------------*/
    def step(self, action=None, mode="model", check_overflow=False):
        active = self._fault_active_this_step()
        target = self.fault_config.get("target", "both")
        inject_sort = active and target in ("sort", "both")
        inject_press = active and target in ("press", "both")

        if active and self.current_step == self._injection_step:
            self.fault_log.append({"onset_step": self._injection_step, "type": "sensor_noise", "target": target})

        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting Action ---
        if self.sort_agent is not None:
            sort_obs = self._noisy_sort_obs() if inject_sort else super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            sort_mode = int(predicted_sort_mode)
        else:
            sort_mode = self.rng_sorting.choice([0, 1])

        # --- 2. Pressing Action ---
        if self.press_agent is not None:
            press_obs = self._noisy_press_obs() if inject_press else super().get_press_obs()
            predicted_press_action, _ = self.press_agent.predict(press_obs, deterministic=True)
            press_action_discrete = int(predicted_press_action)
        else:
            press_action_discrete = self.rng_pressing.choice(11)

        fault_context = {
            "active": active, "fault_type": "sensor_noise" if active else None,
            "sort_affected": inject_sort, "press_affected": inject_press,
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
                    "fault_active": active, "fault_type": "sensor_noise" if active else None,
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
            "fault_type": "sensor_noise" if active else None,
        }
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/