# ---------------------------------------------------------*\
# Title: Byzantine / Adversarial Agent Fault Injection for Env_Combined
# ---------------------------------------------------------*/
#
# Same MRO caveat as the other env_fault_*.py modules: Env_Combined.step()
# calls `super().get_sort_obs()` / `super().get_press_obs()` explicitly, so
# we override step() itself here too, mirroring Env_Combined's logic.
#
# Conceptual difference from the other three "agent still present" faults:
#   - sensor_noise         corrupts what the agent PERCEIVES (before .predict()).
#   - actuator_degradation corrupts what actually reaches hardware (after .predict()).
#   - comms_loss           corrupts the cross-agent slice of observation.
#   - byzantine (this file) corrupts none of the above. The agent perceives
#     correctly and its command is executed faithfully - what's compromised
#     is the DECISION itself. `.predict()` on the real trained policy is
#     bypassed entirely for the targeted agent(s), and an adversarial action
#     source is substituted in its place. No adversarially-trained model is
#     required - the "adversary" is computed on the fly, same as every other
#     fault in this suite.
#
# This is also, deliberately, the hard-to-detect fault in the suite: unlike
# agent_dropout (agent goes silent) or sensor_noise (values look noisy), a
# Byzantine agent looks alive and its individual actions can look plausible.
# That's intentional - it's meant to stress-test whether a recovery
# mechanism can catch something that doesn't "look broken" from the
# outside, rather than something Claude/you engineered to be easy to catch.

import numpy as np
from src.envs_train.env_combined import Env_Combined


class Env_ByzantineFault(Env_Combined):
    """
    Drop-in replacement for Env_Combined that swaps the sorting and/or
    pressing agent's DECISION POLICY for an adversarial one, starting at a
    (possibly randomized) step and lasting either persistently or for a
    fixed duration. The compromised agent's `.predict()` is never called
    while the fault is active - it is not consulted at all, exactly like
    agent_dropout - but unlike agent_dropout, what replaces it is not a
    neutral failsafe: it is an action source that is actively trying to
    hurt system performance.

    Three adversarial modes are supported:
      - "fixed_malicious" : a single harmful action is chosen once, at the
                             instant the fault activates, and held fixed
                             (latched) for the rest of the fault window -
                             same latching pattern as actuator_degradation's
                             "stuck" mode. If no explicit malicious_* value
                             is configured, it is seeded from the adaptive
                             heuristic below (computed once, then frozen).
      - "worst_action"    : recomputed every faulty step from current state
                             using a cheap, honest heuristic (no adversarial
                             training or reward-simulation search needed):
                               sort  -> inverts sorting_rules(), i.e.
                                        deliberately boosts the LESS
                                        abundant material group, actively
                                        degrading purity of the dominant one.
                               press -> ties up a free press on whichever
                                        container currently has the LEAST
                                        material (instead of the fullest,
                                        which check_container_level() would
                                        pick), wasting that press's full
                                        cycle time while the genuinely full
                                        container is left to fill toward
                                        overflow.
                             This is deliberately adaptive/state-aware,
                             which is what makes it harder for a naive
                             "does this look wrong" detector to catch than
                             fixed_malicious.
      - "random_malicious": ignores both its trained policy and system
                             state entirely; emits a uniformly random legal
                             action each faulty step. Models a compromised
                             agent that is unpredictable/lying rather than
                             purposefully optimal.

    fault_config keys:
        injection_step         : int or None (fixed step; None -> randomize)
        injection_step_range   : (lo, hi) used when injection_step is None
        duration                : int or None (None = persistent for rest of episode)
        target                  : "sort", "press", or "both" - which
                                   agent(s) are compromised
        seed                    : int or None, for the adversarial-action RNG

        byzantine_mode          : "fixed_malicious" | "worst_action" |
                                   "random_malicious", default "worst_action"

        # "fixed_malicious" mode
        malicious_sort_mode     : int (0/1) or None -> if None, latched
                                   from the worst_action heuristic at onset
        malicious_press_action  : int in [0, 10] or None -> if None,
                                   latched from the worst_action heuristic
                                   at onset
    """

    def __init__(self, *args, fault_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_config = fault_config or {}
        self._fault_rng = np.random.default_rng(self.fault_config.get("seed"))
        self._injection_step = None
        self._duration = None
        self.fault_log = []

        # Latched state for "fixed_malicious" mode - set on the first
        # faulty step, then held fixed for the remainder of the fault
        # window (same pattern as actuator_degradation's "stuck" mode).
        self._latched_sort_mode = None
        self._latched_press_action = None

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

        self._latched_sort_mode = None
        self._latched_press_action = None

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
    # Adversarial action helpers
    # ---------------------------------------------------------*/
    def _worst_sort_mode(self):
        """Deliberately boosts the LESS abundant belt material group,
        i.e. the inverse of what sorting_rules() (the "good" heuristic
        already in Env_Super) would pick."""
        good_mode = super().sorting_rules()
        return 1 - int(good_mode)

    def _worst_press_action(self):
        """Ties up a free press on the LEAST-full container (instead of
        the fullest one, which check_container_level() would pick),
        wasting a full press cycle while a genuinely full container is
        left to fill toward overflow. Returns a discrete press action
        (0 = no-op, if no press is currently free)."""
        free_press = None
        if self.press_state["press_1"] == 0:
            free_press = 1
        elif self.press_state["press_2"] == 0:
            free_press = 2
        if free_press is None:
            return 0  # no press available anyway -> no-op

        worst_idx, worst_level = None, None
        for i, mat in enumerate(self.material_names):
            lvl = self.container_materials[mat] + self.container_materials.get(f"{mat}_False", 0)
            if worst_level is None or lvl < worst_level:
                worst_level, worst_idx = lvl, i
        lvl_e = self.container_materials["E"]
        if worst_level is None or lvl_e < worst_level:
            worst_level, worst_idx = lvl_e, 4  # index 4 = E

        return super().press_action_to_discrete(free_press, worst_idx)

    def _malicious_sort_mode(self):
        mode = self.fault_config.get("byzantine_mode", "worst_action")

        if mode == "fixed_malicious":
            if self._latched_sort_mode is None:
                configured = self.fault_config.get("malicious_sort_mode")
                self._latched_sort_mode = (
                    int(configured) if configured is not None else self._worst_sort_mode()
                )
            return self._latched_sort_mode

        if mode == "random_malicious":
            return int(self._fault_rng.choice([0, 1]))

        # default: "worst_action" - recomputed every faulty step
        return self._worst_sort_mode()

    def _malicious_press_action(self):
        mode = self.fault_config.get("byzantine_mode", "worst_action")

        if mode == "fixed_malicious":
            if self._latched_press_action is None:
                configured = self.fault_config.get("malicious_press_action")
                self._latched_press_action = (
                    int(configured) if configured is not None else self._worst_press_action()
                )
            return self._latched_press_action

        if mode == "random_malicious":
            return int(self._fault_rng.choice(11))

        # default: "worst_action" - recomputed every faulty step
        return self._worst_press_action()

    # ---------------------------------------------------------*/
    # Step — mirrors Env_Combined.step(); for a compromised agent,
    # .predict() is skipped entirely (never called, same as agent_dropout)
    # and an adversarial action is substituted instead.
    # ---------------------------------------------------------*/
    def recovery_hook(self, sort_mode, press_action_discrete, fault_context):
        """No-op by default; override in a recovery mixin. See
        env_fault_sensor_noise.py's recovery_hook() docstring for the
        shared fault_context shape used across all 5 fault modules."""
        return sort_mode, press_action_discrete

    def step(self, action=None, mode="model", check_overflow=False):
        active = self._fault_active_this_step()
        target = self.fault_config.get("target", "both")
        sort_compromised = active and target in ("sort", "both")
        press_compromised = active and target in ("press", "both")

        if active and self.current_step == self._injection_step:
            self.fault_log.append({
                "onset_step": self._injection_step,
                "type": "byzantine",
                "byzantine_mode": self.fault_config.get("byzantine_mode", "worst_action"),
                "target": target,
            })

        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting Action ---
        if sort_compromised:
            # Real policy is not consulted at all - adversarial action used.
            intended_sort_mode = None
            sort_mode = self._malicious_sort_mode()
        elif self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            intended_sort_mode = int(predicted_sort_mode)
            sort_mode = intended_sort_mode
        else:
            intended_sort_mode = self.rng_sorting.choice([0, 1])
            sort_mode = intended_sort_mode

        # --- 2. Pressing Action ---
        if press_compromised:
            # Real policy is not consulted at all - adversarial action used.
            intended_press_action = None
            press_action_discrete = self._malicious_press_action()
        elif self.press_agent is not None:
            press_obs = super().get_press_obs()
            predicted_press_action, _ = self.press_agent.predict(press_obs, deterministic=True)
            intended_press_action = int(predicted_press_action)
            press_action_discrete = intended_press_action
        else:
            intended_press_action = self.rng_pressing.choice(11)
            press_action_discrete = intended_press_action

        fault_context = {
            "active": active, "fault_type": "byzantine" if active else None,
            "sort_affected": sort_compromised, "press_affected": press_compromised,
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
                    "fault_active": active, "fault_type": "byzantine" if active else None,
                    "sort_compromised": sort_compromised, "press_compromised": press_compromised,
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
            "fault_type": "byzantine" if active else None,
            "sort_compromised": sort_compromised, "press_compromised": press_compromised,
            "intended_sort_mode": intended_sort_mode, "executed_sort_mode": int(sort_mode),
            "intended_press_action": intended_press_action, "executed_press_action": int(press_action_discrete),
        }
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/