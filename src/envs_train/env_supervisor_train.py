# ---------------------------------------------------------*\
# Title: Supervisor-Agent Training Environment
# ---------------------------------------------------------*/
#
# The gym env that SB3 trains the supervisor policy (recovery mechanism #3,
# ToDo #5 bullet 3) directly on. Design decisions:
#
#   - Domain randomization across the WHOLE fault suite, mirroring how
#     training_fault_tolerant.py randomizes during the fault-tolerant arm:
#     every episode randomly picks a fault type (or "none" — the false-
#     recovery teaching signal), an injection step (default 40-60, matching
#     the standardized eval suite), a transient/permanent duration, and a
#     target channel. One trained supervisor policy therefore handles all 5
#     fault types, exactly like the two rule-based arms that run against the
#     same fault classes at eval time.
#
#   - The underlying env is recreated every reset() with this episode's
#     randomly-sampled fault_config — that is what lets one env class switch
#     fault types between episodes while still reusing the exact eval-time
#     fault classes and the exact deployment-time recovery wiring (the
#     supervisor combo classes in src/recovery/supervisor.py).
#
#   - The sort/press agents being supervised are the SAME trained vanilla
#     modular pair that rule_based evaluation runs on (attached via
#     set_agents() to the underlying fault env), so the comparison is
#     apples-to-apples: the supervisor learns to recover the base system,
#     not a stand-in.
#
#   - The action the supervisor picks is injected into the underlying env's
#     recovery_hook() via `_pending_supervisor_action`, consumed exactly once
#     per step. At eval time (outside this wrapper) the same mixin derives
#     the decision itself from the attached supervisor model's predict().
#
#   - The reward handed to SB3 is the pure combined system reward
#     (sort_reward + press_reward). No extra intervention penalty: whether an
#     intervention helped or hurt is already reflected in the reward the
#     policy observes, which is precisely the false-recovery signal we want
#     it to internalize (overriding a healthy channel costs reward).

import gymnasium as gym
import numpy as np

from src.recovery.supervisor import SUPERVISOR_REGISTRY, SupervisorRecoveryMixin


class Env_SupervisorTraining(gym.Env):
    """PPO training env for the hierarchical supervisor policy.

    observation space : SupervisorRecoveryMixin.supervisor_observation_space()
                        (true state + reward window + episode progress + own
                        last decision — see src/recovery/supervisor.py)
    action space      : Discrete(4) — the intervention vocabulary shared with
                        rule_based recovery (0 = pass, 1 = sort override,
                        2 = press override, 3 = both).

    Args:
        max_steps    : episode length (200, matching STEPS_TEST).
        seed         : master seed for episode-level fault randomization.
        sort_model   : trained vanilla Sorting PPO model to supervise.
        press_model  : trained vanilla Pressing PPO model to supervise.
    """

    FAULT_TYPE_CHOICES = [
        "sensor_noise", "actuator_degradation", "agent_dropout", "comms_loss", "byzantine",
    ]
    FAULT_PROB = 0.8                  # fraction of episodes that get a fault at all
    TRANSIENT_PROB = 0.7              # of faulted episodes, fraction transient vs permanent
    TRANSIENT_DURATION_RANGE = (10, 40)
    INJECTION_STEP_RANGE = (40, 60)   # matches DEFAULT_FAULT_CONFIG in main.py

    TARGET_CHOICES = ("sort", "press", "both")

    def __init__(self, max_steps=200, seed=42, sort_model=None, press_model=None):
        super().__init__()
        self.max_steps = max_steps
        self.seed = seed or 0
        self.sort_model = sort_model
        self.press_model = press_model

        self._rng = np.random.default_rng(self.seed)
        self._episode_fault_type = None

        self.env = None                 # underlying combo fault env, recreated per episode

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = SupervisorRecoveryMixin.supervisor_observation_space()

    # ---------------------------------------------------------*/
    # Per-episode fault configuration
    # ---------------------------------------------------------*/
    def _sample_fault_config(self, fault_type):
        cfg = {
            "injection_step_range": self.INJECTION_STEP_RANGE,
            "duration": None,
            "duration_range": None,
            "target": str(self._rng.choice(self.TARGET_CHOICES)),
            "seed": int(self._rng.integers(0, 2**31 - 1)),
            "noise_std": 0.1,
            "mode": "stuck",
            "stuck_press_action": 0,
            "dropout_sort_mode": 0,
            "dropout_press_action": 0,
            "comms_mode": "stale",
            "byzantine_mode": "worst_action",
        }

        if fault_type == "actuator_degradation":
            cfg["mode"] = str(self._rng.choice(["stuck", "slip"]))
        elif fault_type == "comms_loss":
            cfg["comms_mode"] = str(self._rng.choice(["stale", "blackout"]))
            cfg["target"] = "press"  # comms_loss only has an effect on the press channel
        elif fault_type == "byzantine":
            cfg["byzantine_mode"] = str(self._rng.choice(
                ["fixed_malicious", "worst_action", "random_malicious"]
            ))

        if self._rng.random() < self.TRANSIENT_PROB:
            d_lo, d_hi = self.TRANSIENT_DURATION_RANGE
            cfg["duration_range"] = (d_lo, d_hi)
        else:
            cfg["duration_range"] = None  # permanent

        return cfg

    def _no_fault_config(self):
        lo = max(self.max_steps, 1)  # injectable only at a step the episode never reaches
        return {
            "injection_step_range": (lo, lo),
            "duration": 0,
            "target": "both",
            "seed": int(self._rng.integers(0, 2**31 - 1)),
        }

    def _make_env(self, combo_key, fault_config):
        env = SUPERVISOR_REGISTRY[combo_key](
            max_steps=self.max_steps, seed=self.seed, fault_config=fault_config,
        )
        env.set_agents(sort_agent=self.sort_model, press_agent=self.press_model)
        return env

    # ---------------------------------------------------------*/
    # Gym interface
    # ---------------------------------------------------------*/
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(seed)

        if self._rng.random() < self.FAULT_PROB:
            self._episode_fault_type = str(self._rng.choice(self.FAULT_TYPE_CHOICES))
            fault_config = self._sample_fault_config(self._episode_fault_type)
            combo_key = self._episode_fault_type
        else:
            self._episode_fault_type = "none"
            fault_config = self._no_fault_config()
            combo_key = "sensor_noise"  # any combo class; fault can never fire with this config

        self.env = self._make_env(combo_key, fault_config)
        obs, info = self.env.reset(seed=self.seed)
        return self.env.get_supervisor_obs(), info

    def step(self, action):
        self.env._pending_supervisor_action = int(action)
        _, reward, terminated, truncated, info = self.env.step()
        info = dict(info)
        info["supervisor_action"] = int(action)
        info["episode_fault_type"] = self._episode_fault_type
        return self.env.get_supervisor_obs(), float(reward), terminated, truncated, info

    def render(self, *args, **kwargs):
        pass

# -----------------------------------------------------------------------------*/