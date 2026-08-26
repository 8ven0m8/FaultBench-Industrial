# ---------------------------------------------------------*\
# Title: Environment (Combined Runner) - Modular Agents, No Masking
# ---------------------------------------------------------*/

# src/envs_train/env_combined.py
#
# This is NOT a single-agent "monolith" training environment. It exists
# purely as a shared-state harness so the independently trained Sorting
# agent and Pressing agent can act together, step by step, in one
# environment (material flows from sorting -> pressing, so they need to
# share state to be evaluated/visualized together).
#
# No action masking is used anywhere in this file.

import numpy as np
from gymnasium import spaces
from src.envs_train.env_super import Env_Super


class Env_Combined(Env_Super):
    """
    Runner environment for the modular (Sorting + Pressing) agent pair.
    - Holds references to the two trained sub-agents.
    - Each step: sorting agent predicts its action, pressing agent
      predicts its action, both are applied to the shared environment
      state, and the combined reward is returned.
    """

    def __init__(self, max_steps: int = 50, seed: int = None,
                 noise_sorting: float = 0.05, balesize: int = 200, simulation=False):

        super().__init__(max_steps=max_steps, seed=seed,
                         noise_sorting=noise_sorting, balesize=balesize, simulation=simulation)

        self.name = "combined"

        # Modular sub-agents (assigned via set_agents)
        self.sort_agent = None
        self.press_agent = None

        self._initialize_spaces()

    # ---------------------------------------------------------*/
    # Agent assignment
    # ---------------------------------------------------------*/
    def set_agents(self, sort_agent=None, press_agent=None):
        """Assign the pre-trained modular Sorting/Pressing agents."""
        self.sort_agent = sort_agent
        self.press_agent = press_agent

    # ---------------------------------------------------------*/
    # Spaces (kept for gym compatibility / rendering; this env is never trained)
    # ---------------------------------------------------------*/
    def _initialize_spaces(self):
        sort_low = np.concatenate([
            np.zeros(1),         # Belt occupancy
            np.zeros(4),         # Belt proportions
            np.zeros(4),         # Sorting accuracy
            np.full(4, -1.0)     # Purity differences
        ])
        sort_high = np.concatenate([
            np.ones(1), np.ones(4), np.ones(4), np.ones(4)
        ])

        press_low = np.zeros(16)
        press_high = np.ones(16)

        self.observation_space = spaces.Box(
            low=np.concatenate([sort_low, press_low]),
            high=np.concatenate([sort_high, press_high]),
            dtype=np.float32
        )

        # 2 sorting modes x 11 pressing actions
        self.action_space = spaces.Discrete(2 * 11)

    # ---------------------------------------------------------*/
    # Reset
    # ---------------------------------------------------------*/
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.get_obs(), {}

    # ---------------------------------------------------------*/
    # Observation
    # ---------------------------------------------------------*/
    def get_obs(self):
        sort_obs = super().get_sort_obs()
        press_obs = super().get_press_obs()
        return np.concatenate([sort_obs, press_obs])

    # ---------------------------------------------------------*/
    # Step
    # ---------------------------------------------------------*/
    def step(self, action=None, mode="model", check_overflow=False):
        """
        Runs one shared timestep for the modular agent pair.
        'action' and 'mode' are accepted for interface compatibility with
        test_env(), but this harness only ever drives the two modular
        agents (or falls back to random per-agent behavior if an agent
        hasn't been assigned).
        """
        # --- Environment dynamics ---
        occ = super().input_action_rules()
        super().update_environment(batchsize=occ)

        # ---------------------------------------------------------
        # --- 1. Sorting Action ---
        # ---------------------------------------------------------
        if self.sort_agent is not None:
            sort_obs = super().get_sort_obs()
            predicted_sort_mode, _ = self.sort_agent.predict(sort_obs, deterministic=True)
            sort_mode = int(predicted_sort_mode)
        else:  # Fallback: random sorting mode
            sort_mode = self.rng_sorting.choice([0, 1])

        # ---------------------------------------------------------
        # --- 2. Pressing Action ---
        # ---------------------------------------------------------
        if self.press_agent is not None:
            press_obs = super().get_press_obs()
            predicted_press_action, _ = self.press_agent.predict(press_obs, deterministic=True)
            press_action_discrete = int(predicted_press_action)
        else:  # Fallback: random (unmasked) pressing action
            press_action_discrete = self.rng_pressing.choice(11)

        chosen_flat_action = int(sort_mode) * 11 + int(press_action_discrete)

        # ---------------------------------------------------------
        # --- 3. Apply the Actions ---
        # ---------------------------------------------------------
        super().set_multisensor_mode(sort_mode)
        super().update_accuracy()
        super().sort_material()

        press_action_tuple = super().press_discrete_to_action(press_action_discrete)
        super().press_action_rules(
            (press_action_tuple[0], press_action_tuple[1]) if press_action_tuple[0] != 0 else (None, None)
        )

        # ---------------------------------------------------------
        # --- 4. Reward, Logging, Return ---
        # ---------------------------------------------------------
        if check_overflow:
            overflow, mat = super().detect_overflow()
            if overflow:
                reward = self.overflow_penalty
                info = {"overflow": True, "overflow_material": mat, "action": chosen_flat_action}
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
        info = {"action": chosen_flat_action}
        return obs_next, float(reward), terminated, False, info

# -----------------------------------------------------------------------------*/
