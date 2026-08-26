# ---------------------------------------------------------*\
# Title: Communication Loss Fault Injection for Env_Combined
# ---------------------------------------------------------*/
#
# Same MRO caveat as the other env_fault_*.py modules: Env_Combined.step()
# calls `super().get_sort_obs()` / `super().get_press_obs()` explicitly, so
# we override step() itself here too, mirroring Env_Combined's logic.
#
# IMPORTANT - this fault type does NOT behave like the other three. This
# codebase has no explicit inter-agent message channel (no comm vectors, no
# broadcast). The only place one agent's decisions actually flow into the
# other agent's observation is a single field: get_press_obs()'s
# `sorter_amounts` slice, which reports self.current_material_sorting - the
# direct downstream effect of whatever the sort agent just did. The sort
# agent's own observation (get_sort_obs()) contains nothing about the press
# agent at all, so there is no equivalent channel to sever in that
# direction.
#
# Consequently:
#   - comms_loss only has an effect when target includes "press". Setting
#     target="sort" is a no-op (nothing to cut) and no fault_log entry is
#     recorded for it - this is intentional, not a bug, and is documented
#     here rather than silently faked.
#   - This is NOT the same fault as agent_dropout. Under comms_loss the
#     press agent is alive and still runs its real trained policy every
#     step (.predict() is still called) - only the cross-agent slice of its
#     input is corrupted, while its own local sensing (container levels,
#     press timers) stays accurate. Under agent_dropout, the agent isn't
#     consulted at all and a fixed failsafe action is substituted instead.

import numpy as np
from src.envs_train.env_combined import Env_Combined

# Index range of the `sorter_amounts` slice within the 16-dim press
# observation vector: [levels(5) + ratios(5) + sorter_amounts(4) + press_timers(2)].
# See Env_Super.get_press_obs() for the authoritative layout.
_SORTER_AMOUNTS_SLICE = slice(10, 14)


class Env_CommsLossFault(Env_Combined):
    """
    Drop-in replacement for Env_Combined that corrupts the cross-agent
    slice of the press agent's observation (the part reporting what the
    sort agent's decisions did to the sorting-machine contents), starting
    at a (possibly randomized) step and lasting either persistently or for
    a fixed duration. The press agent's own local sensing (container
    levels/ratios, press timers) and the sort agent's observation are both
    left untouched.

    Two modes:
      - "stale"    : the sorter_amounts slice is latched at its
                     last-known-good value the instant the link drops, and
                     held there for the rest of the fault window - the
                     press agent keeps "seeing" a frozen snapshot of the
                     sorting stage rather than its live state.
      - "blackout" : the sorter_amounts slice is zeroed out for the
                     duration of the fault - the press agent sees nothing
                     coming from the sorting stage at all.

    fault_config keys:
        injection_step        : int or None (fixed step; None -> randomize)
        injection_step_range  : (lo, hi) used when injection_step is None
        duration               : int or None (None = persistent for rest of episode)
        target                  : "press" is the only value with an effect
                                   (kept as a key for API consistency with
                                   the other fault modules; "sort" or a
                                   target that excludes "press" is a no-op)
        seed                    : int or None, for the injection-step RNG
        comms_mode              : "stale" | "blackout", default "stale"
                                   (deliberately a distinct key from
                                   actuator_degradation's "mode" - they share
                                   DEFAULT_FAULT_CONFIG in main.py, and a
                                   colliding key would silently overwrite
                                   each other since it's the same dict)
    """

    def __init__(self, *args, fault_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_config = fault_config or {}
        self._fault_rng = np.random.default_rng(self.fault_config.get("seed"))
        self._injection_step = None
        self._duration = None
        self.fault_log = []

        # Latched value for "stale" mode - set the instant the link drops,
        # held fixed for the remainder of the fault window.
        self._latched_sorter_amounts = None

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

        self._latched_sorter_amounts = None

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
    # Corruption helper — press agent's own local sensing is untouched;
    # only the cross-agent (sorter_amounts) slice is affected.
    # ---------------------------------------------------------*/
    def _corrupt_press_obs(self, press_obs):
        mode = self.fault_config.get("comms_mode", "stale")
        corrupted = press_obs.copy()

        if mode == "stale":
            if self._latched_sorter_amounts is None:
                self._latched_sorter_amounts = press_obs[_SORTER_AMOUNTS_SLICE].copy()
            corrupted[_SORTER_AMOUNTS_SLICE] = self._latched_sorter_amounts
        elif mode == "blackout":
            corrupted[_SORTER_AMOUNTS_SLICE] = 0.0

        return corrupted

    # ---------------------------------------------------------*/
    # Step — mirrors Env_Combined.step(); only the press agent's perceived
    # observation is corrupted (target="sort" has no effect - see module
    # docstring for why).
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        """No-op by default; override in a recovery mixin. See
        env_fault_sensor_noise.py's recovery_hook() docstring for the
        shared fault_context shape used across all 5 fault modules."""
        return sort_mode, press_action_discrete

    def step(self, action=None, mode="model", check_overflow=False):
        active = self._fault_active_this_step()
        target = self.fault_config.get("target", "press")
        press_comms_lost = active and target in ("press", "both")

        if press_comms_lost and self.current_step == self._injection_step:
            self.fault_log.append({
                "onset_step": self._injection_step,
                "type": "comms_loss",
                "mode": self.fault_config.get("comms_mode", "stale"),
                "target": "press",
            })

        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting Action (this fault never touches the sort agent) ---
        if self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            sort_mode = int(predicted_sort_mode)
        else:
            sort_mode = self.rng_sorting.choice([0, 1])

        # --- 2. Pressing Action (observation corrupted if comms are down) ---
        if self.press_agent is not None:
            press_obs = super().get_press_obs()
            press_obs_seen = self._corrupt_press_obs(press_obs) if press_comms_lost else press_obs
            predicted_press_action, _ = self.press_agent.predict(press_obs_seen, deterministic=True)
            press_action_discrete = int(predicted_press_action)
        else:
            press_action_discrete = self.rng_pressing.choice(11)

        fault_context = {
            "active": press_comms_lost, "fault_type": "comms_loss" if press_comms_lost else None,
            "sort_affected": False,  # this fault never touches the sort agent
            "press_affected": press_comms_lost,
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
                    "fault_active": press_comms_lost, "fault_type": "comms_loss" if press_comms_lost else None,
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
            "fault_active": press_comms_lost,
            "fault_type": "comms_loss" if press_comms_lost else None,
        }
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/