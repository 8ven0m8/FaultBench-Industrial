# ---------------------------------------------------------*\
# Title: Sorting Agent Training Env — Domain-Randomized (Fault-Tolerant)
# ---------------------------------------------------------*/
#
# Recovery mechanism #2 (ToDo #5, bullet 2): "Fault-tolerant-trained MARL —
# retrain/fine-tune policies with faults injected during training itself
# (domain randomization), so the policy is inherently more robust."
#
# Unlike src/recovery/rule_based*.py, there is no test-time correction
# here at all - this env is only ever used DURING TRAINING (see
# src/training_fault_tolerant.py). At eval time, the resulting checkpoint
# is loaded into the SAME plain fault-injection envs everything else uses
# (Env_ByzantineFault, etc., with no recovery mixin) - whatever robustness
# exists is baked into the weights, not applied as a correction.
#
# Only two fault types are domain-randomized here: sensor_noise and
# actuator_degradation. This is a deliberate scope decision, not an
# oversight:
#   - sensor_noise / actuator_degradation still consult this agent's own
#     .predict() (on a corrupted observation, or its output gets corrupted
#     after) - a hardened policy can plausibly learn to compensate.
#   - agent_dropout / byzantine bypass .predict() entirely at eval time
#     (see env_fault_agent_dropout.py / env_fault_byzantine.py). Training
#     THIS agent under a simulated "self-dropout/self-byzantine" would mean
#     overriding its own chosen action every faulty step - it would receive
#     reward for actions it never actually took, which provides no useful
#     gradient signal. Those two fault types are structurally something
#     only a recovery mechanism, or a TEAMMATE's hardened policy, can
#     address - never the compromised agent's own training.
#   - "teammate fault" domain randomization (corrupting what the press
#     side does during sort training) is also skipped here: Env_1_Sorting's
#     own step() never actually consults a real press agent regardless of
#     what set_agents(press_agent=...) is given - it always uses
#     sample_masked_press_action(), an already-random valid action. Layering
#     more corruption onto an already-random teammate wouldn't add a
#     meaningful training signal. (The press trainer below doesn't have
#     this limitation - a real pretrained sort_agent is genuinely in the
#     loop there, so its trainer DOES randomize teammate faults.)

import numpy as np
from src.envs_train.env_1_sort import Env_1_Sorting


class Env_1_Sorting_FaultTolerant(Env_1_Sorting):
    FAULT_PROB = 0.40                  # fraction of training episodes that get a fault at all
    TRANSIENT_PROB = 0.70              # of faulted episodes, fraction transient vs permanent
    TRANSIENT_DURATION_RANGE = (10, 40)
    INJECTION_STEP_RANGE = (20, 120)   # out of a 200-step training episode

    SENSOR_NOISE_STD = 0.08
    ACTUATOR_MODE_CHOICES = ("stuck", "restrict", "slip")
    ACTUATOR_SLIP_PROB = 0.30

    def __init__(self, *args, seed=None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._fault_rng = np.random.default_rng(seed)
        self._episode_fault_type = None
        self._injection_step = None
        self._duration = None
        self._latched_sort_mode = None
        self._latched_actuator_mode = None
        self._latched_restrict_mode = None

    def reset(self, seed=None):
        obs, info = super().reset(seed=seed)

        if self._fault_rng.random() < self.FAULT_PROB:
            self._episode_fault_type = str(self._fault_rng.choice(["sensor_noise", "actuator_degradation"]))
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

        self._latched_sort_mode = None
        self._latched_actuator_mode = None
        self._latched_restrict_mode = None
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
            return np.clip(obs + noise, -1.0, 1.0)
        return obs

    def step(self, action=None, use_action_masking=True, check_overflow=False):
        if self._episode_fault_type == "actuator_degradation" and self._fault_active():
            if self._latched_actuator_mode is None:
                self._latched_actuator_mode = str(self._fault_rng.choice(self.ACTUATOR_MODE_CHOICES))

            if self._latched_actuator_mode == "stuck":
                if self._latched_sort_mode is None:
                    self._latched_sort_mode = action
                action = self._latched_sort_mode
            elif self._latched_actuator_mode == "restrict":
                if self._latched_restrict_mode is None:
                    self._latched_restrict_mode = int(self._fault_rng.choice([0, 1]))
                if int(action) == self._latched_restrict_mode:
                    action = 1 - self._latched_restrict_mode
            else:  # "slip"
                if self._fault_rng.random() < self.ACTUATOR_SLIP_PROB:
                    action = 1 - int(action)
        else:
            self._latched_sort_mode = None
            self._latched_actuator_mode = None
            self._latched_restrict_mode = None

        obs, reward, terminated, truncated, info = super().step(
            action=action, use_action_masking=use_action_masking, check_overflow=check_overflow
        )
        obs = self._maybe_corrupt_obs(obs)
        info["fault_type"] = self._episode_fault_type if self._fault_active() else None
        return obs, reward, terminated, truncated, info

# -----------------------------------------------------------------------------*/