# ---------------------------------------------------------*\
# Title: Testing / Running the Modular (No-Mask) Agents
# ---------------------------------------------------------*/

import numpy as np

# ---------------------------------------------------------*/
# Run an episode in the environment and log/plot the result
# ---------------------------------------------------------*/


def test_env(env=None, tag="", save=False, title="", steps=50, dir="./img/", seed=None, show=False,
             stats=True, mode="model", model=None):
    """
    Runs a simulation for a given number of steps.

    Args:
        env: The environment instance to test. For a single agent (e.g. the
             Sorting env on its own), pass `model`. For the modular pair
             (Sorting + Pressing acting together), use env_combined.Env_Combined
             with agents assigned via set_agents() and leave `model=None` -
             the environment drives both agents internally.
        mode (str): Kept for interface compatibility; only "model" is used
                    in the no-mask modular pipeline.
        model: Single-agent model to use, if testing one agent in isolation.
    """
    if env is None:
        raise ValueError("Environment must be provided")

    obs, info = env.reset(seed=seed)
    action_sequence = []
    cumulative_reward = 0.0

    for i in range(steps):
        action = None

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        # If no model is passed, the environment's internal logic handles
        # the modular agents (env.step() uses self.sort_agent / self.press_agent).

        obs, reward, done, _, info = env.step(action=action, mode=mode)
        cumulative_reward += reward

        chosen_action = info.get("action", action)
        action_sequence.append(chosen_action)

        if done:
            if stats:
                print(f"\n---- Testing Results - {mode} ----")
                print(f"🏁 Epoch ended after \033[1m{i + 1}\033[0m steps.")
                env.render(save=True, log_dir=dir, filename=f'{tag}_env_simulation', title=title, show=show,
                           steps_test=steps)
            else:
                env.render(save=True, log_dir=dir, filename=f'{tag}_env_simulation', title=title, show=show,
                           checksum=False, steps_test=steps)

            total_rewards = [sum(reward) for reward in env.reward_data['Reward']]
            cumulative_total_reward = np.cumsum(total_rewards)[-1]

            if stats:
                print(f"👑 Total Reward: {cumulative_total_reward:.2f}")

            break

    if 'Reward' in env.reward_data and env.reward_data['Reward']:
        total_rewards = [sum(reward) for reward in env.reward_data['Reward']]
        final_cumulative = np.cumsum(total_rewards)[-1]
    else:
        final_cumulative = cumulative_reward

    return final_cumulative, action_sequence

# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*\
