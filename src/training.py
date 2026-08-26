# ---------------------------------------------------------*/
# Title: Training the Modular RL Agents (PPO, no action masking)
# ---------------------------------------------------------*/

import os
import copy
import time
import shutil
import glob

from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")

# ---------------------------------------------------------*/
# Helper Functions
# ---------------------------------------------------------*/

def find_latest_model(prefix):
    """
    Find the latest saved model file with the given prefix.
    Returns the path to the most recent model or None if not found.
    """
    models_dir = "./models"
    pattern = os.path.join(models_dir, f"{prefix}_*.zip")
    model_files = glob.glob(pattern)

    if not model_files:
        return None

    latest_file = max(model_files, key=os.path.getmtime)
    return latest_file

# ---------------------------------------------------------*/
# Train a single RL Agent (PPO, unmasked)
# ---------------------------------------------------------*/

def Train_Agent(env, total_timesteps, save_prefix, experiment=None, logpath=None):
    if env is None:
        raise ValueError("Environment must be provided")

    env = Monitor(env)
    check_env(env)

    # --- Wrap the evaluation environment ---
    # Temporarily remove agents before deepcopying to avoid serialization issues with PPO models
    sort_agent_ref = getattr(env.unwrapped, 'sort_agent', None)
    if sort_agent_ref:
        env.unwrapped.sort_agent = None

    eval_env = copy.deepcopy(env.unwrapped)  # clean unwrapped copy

    if sort_agent_ref:
        env.unwrapped.sort_agent = sort_agent_ref
        eval_env.sort_agent = sort_agent_ref

    eval_env = Monitor(eval_env)
    eval_env.reset(seed=99)

    # --- TensorBoard logging directory (single folder per run) ---
    tensorboard_log = os.path.join(logpath, str(save_prefix))
    os.makedirs(tensorboard_log, exist_ok=True)
    time.sleep(0.1)  # short sleep to avoid race conditions on some filesystems

    device = "cpu"
    print(f"Using device: {device}")

    # --- Policy & Model (always standard, unmasked PPO) ---
    policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=tensorboard_log,
        ent_coef=0.05,
        seed=42,
        device=device
    )

    # --- Eval Callback ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./models/best_model/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0
    )

    # --- Load Sorting Agent for Pressing Training ---
    # Needed when training the Pressing agent, which depends on a trained Sorting agent.
    if hasattr(env.unwrapped, "name") and env.unwrapped.name == "press":
        sorting_model_path = find_latest_model("PPO_Sorting_NoMask")
        if sorting_model_path:
            print(f"📂 Pressing Agent training: Loading pre-trained Sorting model from: {sorting_model_path}")
            try:
                from src.envs_train.env_1_sort import Env_1_Sorting
                temp_sort_env = Env_1_Sorting()
                sorting_agent = PPO.load(sorting_model_path, env=temp_sort_env)

                env.unwrapped.set_agents(sort_agent=sorting_agent)
                if hasattr(eval_env, 'unwrapped'):
                    eval_env.unwrapped.set_agents(sort_agent=sorting_agent)
                else:
                    eval_env.set_agents(sort_agent=sorting_agent)

                print("✅ Successfully loaded and assigned pre-trained Sorting Agent for Pressing training.")
            except Exception as e:
                print(f"❌ Failed to load Sorting model: {e}")
                print("⚠️ WARNING: Training Pressing Agent without pre-trained Sorting Agent!")
        else:
            print("⚠️ WARNING: No saved Sorting model found! Training Pressing Agent without pre-trained Sorting Agent!")

    # --- Train ---
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    dur = time.time() - start_time
    print(f"✅ Training done in {dur//60:.0f} m {dur%60:.0f} s")

    # --- Evaluate ---
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final Performance: {mean_r:.2f} ± {std_r:.2f}")

    # --- Optional: Best Checkpoint ---
    best_path = "./models/best_model/best_model.zip"
    if os.path.exists(best_path):
        print(f"📂 Loading best checkpoint model from: {best_path}")
        best_model = PPO.load(best_path, env=env)
        shutil.rmtree("./models/best_model/", ignore_errors=True)
        mean_best, std_best = evaluate_policy(best_model, env, n_eval_episodes=10)
        print(f"Best-Checkpoint: {mean_best:.2f} ± {std_best:.2f}")
        if mean_best > mean_r:
            model = best_model
            print("🏅 Using best checkpoint for saving.")

    # --- Save ---
    save_model(model, prefix=save_prefix, timesteps=total_timesteps)
    return model

# ---------------------------------------------------------*/
# High-level Trainer for the two modular agents
# ---------------------------------------------------------*/


def RL_Trainer(env, env_class, total_timesteps, max_steps, tag, seed, experiment=None):
    """
    Trains a single modular agent (Sorting or Pressing), always as plain
    unmasked PPO, and saves it with a '_NoMask' suffix.
    """
    exp_name = experiment or f"{tag}_{env_class}_{int(time.time())}"

    print("\n----------------------------------------")
    print(f"🏋🏽 Training PPO - {env_class} (No Masking) ...")
    print("----------------------------------------")

    env.reset(seed=None)

    agent = Train_Agent(
        env=env,
        total_timesteps=total_timesteps,
        save_prefix=f"PPO_{env_class}_NoMask",
        experiment=exp_name,
        logpath=f"./log/tensorboard/{tag}",
    )
    shutil.rmtree("./models/best_model/", ignore_errors=True)

    return agent


# ---------------------------------------------------------*/
# Model saving helper
# ---------------------------------------------------------*/
def save_model(model, prefix, timesteps):
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)

    fname = f"{prefix}_{timesteps}.zip"
    fpath = os.path.join(models_dir, fname)

    # move older versions to ./models/prev/
    existing = [f for f in os.listdir(models_dir) if f.startswith(prefix) and f.endswith(".zip")]
    if existing:
        prev_dir = os.path.join(models_dir, "prev")
        os.makedirs(prev_dir, exist_ok=True)
        for old in existing:
            shutil.move(os.path.join(models_dir, old), os.path.join(prev_dir, old))

    model.save(fpath)
    print(f"💾 Saved {prefix} model → {fpath}")


# -------------------------Notes-----------------------------------------------*\
#
# -----------------------------------------------------------------------------*/
