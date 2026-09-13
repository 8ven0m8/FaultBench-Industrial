# ---------------------------------------------------------*\
# Title: Pressing Agent Training Env — Domain-Randomized (Fault-Tolerant)
# ---------------------------------------------------------*/
#
# See env_1_sort_fault_tolerant.py's header for the overall design rationale
# (recovery mechanism #2, ToDo #5 bullet 2). The key difference here: unlike
# Env_1_Sorting, Env_2_Pressing genuinely consults a real pretrained
# self.sort_agent during training, so teammate-fault domain randomization
# IS meaningful here (it wasn't for the sort trainer - see that file).
#
# Because the teammate corruption has to happen BETWEEN "sort_agent.predict()"
# and "set_multisensor_mode()" - i.e. mid-step, not before or after - step()
# is overridden in full here (same reason every env_fault_*.py module in
# src/envs_train/ does the same instead of calling super().step()).
#
# Fault types randomized per episode:
#   - "sensor_noise"          (self): gaussian noise added to the returned
#                              observation - this env's step()/reset() IS
#                              what SB3 trains the policy on directly.
#   - "actuator_degradation"  (self): the press action this policy chose is
#                              corrupted before being applied - "stuck"
#                              (frozen at the first chosen action for the
#                              window), "restrict" (one press id becomes
#                              permanently unreachable for the window - the
#                              action downgrades to a no-op whenever it
#                              targets that press), or "slip" (replaced by
#                              a random action with some probability each
#                              step).
#   - "teammate_dropout"      : self.sort_agent.predict() is skipped
#                              entirely; a fixed failsafe sort_mode is used
#                              instead, simulating a silent/dropped sorting
#                              teammate exactly like env_fault_agent_dropout.py
#                              does at eval time.
#   - "teammate_byzantine"    : self.sort_agent.predict() is skipped and
#                              replaced with an adversarial sort_mode,
#                              mirroring all 3 modes env_fault_byzantine.py
#                              supports at eval time (sampled per episode,
#                              not just "worst_action"):
#                                "worst_action"     -> recomputed every
#                                                      faulty step, inverse
#                                                      of sorting_rules()
#                                "fixed_malicious"  -> one sort_mode chosen
#                                                      at fault onset
#                                                      (seeded from the
#                                                      worst_action heuristic),
#                                                      then latched for the
#                                                      rest of the window
#                                "random_malicious" -> uniform random {0,1}
#                                                      every faulty step,
#                                                      ignoring state entirely
#   - "teammate_comms_loss"   : NOT a teammate-action corruption like the
#                              two above - this corrupts THIS agent's own
#                              perceived observation, mirroring
#                              env_fault_comms_loss.py exactly: the
#                              sorter_amounts slice (obs[10:14], see
#                              Env_Super.get_press_obs()) is either latched
#                              at its last value ("stale") or zeroed
#                              ("blackout") for the fault window, while the
#                              sort agent's real action is still computed
#                              and applied normally - only the pressing
#                              agent's downstream *perception* of it is
#                              corrupted. Handled in _maybe_corrupt_obs(),
#                              not in the teammate-action block of step(),
#                              since the underlying dynamics are unaffected.
#
# self-dropout / self-byzantine on the pressing agent itself are excluded
# for the same reason given in env_1_sort_fault_tolerant.py: they bypass
# .predict() entirely, so training the trainee under them provides no
# usable gradient signal for the trainee's own weights.

import numpy as np
from src.envs_train.env_2_press import Env_2_Pressing

# Index range of the `sorter_amounts` slice within the 16-dim press
# observation vector: [levels(5) + ratios(5) + sorter_amounts(4) + press_timers(2)].
# Must match _SORTER_AMOUNTS_SLICE in env_fault_comms_loss.py exactly, so
# training-time and eval-time comms_loss corrupt the same field.
_SORTER_AMOUNTS_SLICE = slice(10, 14)


class Env_2_Pressing_FaultTolerant(Env_2_Pressing):
    FAULT_PROB = 0.40
    TRANSIENT_PROB = 0.70
    TRANSIENT_DURATION_RANGE = (10, 40)
    INJECTION_STEP_RANGE = (20, 120)

    FAULT_TYPES = [
        "sensor_noise", "actuator_degradation",
        "teammate_dropout", "teammate_byzantine", "teammate_comms_loss",
    ]

    SENSOR_NOISE_STD = 0.08
    ACTUATOR_MODE_CHOICES = ("stuck", "restrict", "slip")
    ACTUATOR_SLIP_PROB = 0.30
    DROPOUT_SORT_MODE = 0  # fixed failsafe sort mode used while "teammate dropped"

    BYZANTINE_MODE_CHOICES = ("worst_action", "fixed_malicious", "random_malicious")
    COMMS_LOSS_MODE_CHOICES = ("stale", "blackout")

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._fault_rng = np.random.default_rng(seed)
        self._episode_fault_type = None
        self._injection_step = None
        self._duration = None
        self._latched_press_action = None
        self._latched_actuator_mode = None
        self._latched_restrict_press_id = None
        self._episode_byzantine_mode = None
        self._latched_byzantine_sort_mode = None
        self._episode_comms_mode = None
        self._latched_sorter_amounts = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if self._fault_rng.random() < self.FAULT_PROB:
            self._episode_fault_type = str(self._fault_rng.choice(self.FAULT_TYPES))
            lo, hi = self.INJECTION_STEP_RANGE
            self._injection_step = int(self._fault_rng.integers(lo, hi + 1))
            if self._fault_rng.random() < self.TRANSIENT_PROB:
                d_lo, d_hi = self.TRANSIENT_DURATION_RANGE
                self._duration = int(self._fault_rng.integers(d_lo, d_hi + 1))
            else:
                self._duration = None  # permanent for the rest of the episode
        else:
            self._episode_fault_type = None
            self._injection_step = None
            self._duration = None

        self._latched_press_action = None
        self._latched_actuator_mode = None
        self._latched_restrict_press_id = None

        self._episode_byzantine_mode = None
        self._latched_byzantine_sort_mode = None
        if self._episode_fault_type == "teammate_byzantine":
            self._episode_byzantine_mode = str(self._fault_rng.choice(self.BYZANTINE_MODE_CHOICES))

        self._episode_comms_mode = None
        self._latched_sorter_amounts = None
        if self._episode_fault_type == "teammate_comms_loss":
            self._episode_comms_mode = str(self._fault_rng.choice(self.COMMS_LOSS_MODE_CHOICES))

        return self._maybe_corrupt_obs(obs), info

    def _fault_active(self):
        if self._episode_fault_type is None:
            return False
        step = self.current_step
        if step < self._injection_step:
            return False
        if self._duration is None:
            return True
        return step < self._injection_step + self._duration

    def _maybe_corrupt_obs(self, obs):
        if self._episode_fault_type == "sensor_noise" and self._fault_active():
            noise = self._fault_rng.normal(0.0, self.SENSOR_NOISE_STD, size=obs.shape).astype(np.float32)
            return np.clip(obs + noise, 0.0, 1.0)  # press obs is normalized to [0, 1], not [-1, 1]

        if self._episode_fault_type == "teammate_comms_loss" and self._fault_active():
            corrupted = obs.copy()
            if self._episode_comms_mode == "blackout":
                corrupted[_SORTER_AMOUNTS_SLICE] = 0.0
            else:  # "stale" (default)
                if self._latched_sorter_amounts is None:
                    self._latched_sorter_amounts = obs[_SORTER_AMOUNTS_SLICE].copy()
                corrupted[_SORTER_AMOUNTS_SLICE] = self._latched_sorter_amounts
            return corrupted

        return obs

    def _malicious_teammate_sort_mode(self):
        """Mirrors Env_ByzantineFault._malicious_sort_mode() in
        env_fault_byzantine.py so the training-time proxy matches the
        eval-time fault exactly, mode for mode."""
        worst_mode = 1 - int(super().sorting_rules())

        if self._episode_byzantine_mode == "fixed_malicious":
            if self._latched_byzantine_sort_mode is None:
                self._latched_byzantine_sort_mode = worst_mode
            return self._latched_byzantine_sort_mode

        if self._episode_byzantine_mode == "random_malicious":
            return int(self._fault_rng.choice([0, 1]))

        return worst_mode  # "worst_action" (default)

    def step(self, action, use_action_masking=True, check_overflow=False):
        fault_active = self._fault_active()
        fault_type = self._episode_fault_type if fault_active else None

        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # --- 1. Sorting side (teammate) ---
        if fault_type == "teammate_dropout":
            sort_mode = self.DROPOUT_SORT_MODE
        elif fault_type == "teammate_byzantine":
            sort_mode = self._malicious_teammate_sort_mode()
        elif self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
        else:
            sort_mode = super().sorting_rules()

        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        # --- 2. Pressing side (self) ---
        chosen_action = int(action)

        if fault_type == "actuator_degradation":
            if self._latched_actuator_mode is None:
                self._latched_actuator_mode = str(self._fault_rng.choice(self.ACTUATOR_MODE_CHOICES))

            if self._latched_actuator_mode == "stuck":
                if self._latched_press_action is None:
                    self._latched_press_action = chosen_action
                chosen_action = self._latched_press_action
            elif self._latched_actuator_mode == "restrict":
                if self._latched_restrict_press_id is None:
                    self._latched_restrict_press_id = int(self._fault_rng.choice([1, 2]))
                broken_press_id, _ = super().press_discrete_to_action(chosen_action)
                if broken_press_id == self._latched_restrict_press_id:
                    chosen_action = 0  # commanded press is down -> action downgraded to no-op
            else:  # "slip"
                if self._fault_rng.random() < self.ACTUATOR_SLIP_PROB:
                    chosen_action = int(self._fault_rng.integers(0, 11))
        else:
            self._latched_press_action = None
            self._latched_actuator_mode = None
            self._latched_restrict_press_id = None

        if not use_action_masking:
            sanitized_action, press_action_tuple, invalid_info = super().sanitize_press_action(chosen_action)
            if invalid_info is not None:
                self.press_actions_per_timestep.append(invalid_info)
        else:
            press_action_tuple = super().press_discrete_to_action(chosen_action)
            press_action_tuple = (press_action_tuple[0], press_action_tuple[1])

        super().press_action_rules(press_action_tuple if press_action_tuple[0] != 0 else (None, None))

        # --- 3. Reward / logging / return ---
        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                reward = self.overflow_penalty
                info = {"overflow": True, "overflow_material": mat, "action": chosen_action, "fault_type": fault_type}
                self.current_step += 1
                self._log_step_data(r_sort=0, r_press=reward)
                return self._maybe_corrupt_obs(self.get_obs()), float(reward), True, False, info

        reward = self.calculate_press_reward()
        obs_next = self._maybe_corrupt_obs(self.get_obs())
        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        self._log_step_data(r_sort=0, r_press=reward)
        info = {"action": chosen_action, "fault_type": fault_type}

        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/