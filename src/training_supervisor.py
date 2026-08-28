# ---------------------------------------------------------*\
# Title: Hierarchical Supervisor-Agent — Standardalone Training Driver
# ---------------------------------------------------------*/
#
# Recovery mechanism #3 (ToDo #5, bullet 3). Trains the supervisor policy on
# Env_SupervisorTraining — which runs the SAME vanilla modular pair that
# rule-based evaluation uses, under domain-randomized faults across the whole
# 5-type suite — and saves the checkpoint under its own prefix
# ("PPO_Supervisor_NoMask_*") so no other model files are touched.
#
# This is a SEPARATE, self-contained SB3 loop rather than a reuse of
# training.py's Train_Agent()/RL_Trainer(): Train_Agent hardcodes a branch
# that overwrites any assigned sort_agent with the latest
# "PPO_Sorting_NoMask" checkpoint (see the same rationale in
# training_fault_tolerant.py), plus Train_Agent's deepcopy-of-eval-env trick
# does not survive an env that holds live SB3 models. The supervisor trainer
# needs a *different* env class (the supervisor's own obs/action spaces), so
# it cannot meaningfully reuse the agent trainers at all.
#
# The only things reused from training.py are save_model() (pure checkpoint
# writer) and find_latest_model() (locating the vanilla sort/press agents the
# supervisor will watch).

import os
import time
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.training import save_model, find_latest_model
from src.envs_train.env_supervisor_train import Env_SupervisorTraining

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")


def train_supervisor(total_timesteps=1_000_000, steps_train=200, steps_test=200,
                     seed=42, tag=None, sort_model_path=None, press_model_path=None):
    """
    Trains the hierarchical supervisor policy over the vanilla modular pair.

    Args:
        total_timesteps : supervisor training budget (the underlying env
                          additionally runs both modular agents' .predict()
                          every step, so this is the dominant cost).
        steps_train     : length of a training episode.
        steps_test      : length of the EvalCallback evaluation episodes.
        seed            : master seed.
        tag             : run tag for the "log/tensorboard/{tag}_Supervisor" dir.
        sort_model_path : PPO Sorting checkpoint to supervise; None -> latest
                          "PPO_Sorting_NoMask_*" in ./models/.
        press_model_path: PPO Pressing checkpoint to supervise; None -> latest
                          "PPO_Pressing_NoMask_*" in ./models/.
    """
    sort_path = sort_model_path or find_latest_model("PPO_Sorting_NoMask")
    press_path = press_model_path or find_latest_model("PPO_Pressing_NoMask")

    if not sort_path or not press_path:
        print("⚠️ Need trained vanilla Sorting + Pressing models for the supervisor to watch.")
        print("   Train them first via main.py -> 'train' -> 'vanilla'.")
        return None
    print(f"📂 Supervising Sorting agent from: {sort_path}")
    print(f"📂 Supervising Pressing agent from: {press_path}")
    sort_model = PPO.load(sort_path)
    press_model = PPO.load(press_path)

    train_env = Env_SupervisorTraining(
        max_steps=steps_train, seed=seed, sort_model=sort_model, press_model=press_model,
    )
    check_env(train_env)
    train_env = Monitor(train_env)

    eval_env = Env_SupervisorTraining(
        max_steps=steps_test, seed=seed + 13, sort_model=sort_model, press_model=press_model,
    )
    eval_env = Monitor(eval_env)

    log_name = f"{tag}_Supervisor" if tag else "PPO_Supervisor"
    tensorboard_log = os.path.join("./log/tensorboard", log_name)
    os.makedirs(tensorboard_log, exist_ok=True)
    time.sleep(0.1)

    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log,
        ent_coef=0.01,
        seed=42,
        device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_supervisor/",
        log_path="./models/best_model_supervisor/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    print(f"\n🏋🏽 Training Supervisor policy (Discrete(4) interventions over the "
          f"vanilla modular pair), {total_timesteps} timesteps ...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    dur = time.time() - start_time
    print(f"✅ Supervisor training done in {dur//60:.0f} m {dur%60:.0f} s")

    mean_r, std_r = evaluate_policy(model, train_env, n_eval_episodes=10)
    print(f"Final Performance: {mean_r:.3f} ± {std_r:.3f}")

    best_path = "./models/best_model_supervisor/best_model.zip"
    if os.path.exists(best_path):
        print(f"📂 Loading best checkpoint model from: {best_path}")
        best_model = PPO.load(best_path, env=train_env)
        mean_best, std_best = evaluate_policy(best_model, train_env, n_eval_episodes=10)
        print(f"Best-Checkpoint: {mean_best:.3f} ± {std_best:.3f}")
        if mean_best > mean_r:
            model = best_model
            print("🏅 Using best checkpoint for saving.")
        shutil.rmtree("./models/best_model_supervisor/", ignore_errors=True)

    save_model(model, prefix="PPO_Supervisor_NoMask", timesteps=total_timesteps)
    print(f"✅ Supervisor checkpoint saved → ./models/PPO_Supervisor_NoMask_{total_timesteps}.zip")
    print("   (existing vanilla / fault-tolerant modular checkpoints untouched)")
    return model


# -----------------------------------------------------------------------------*/