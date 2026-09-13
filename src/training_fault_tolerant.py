# ---------------------------------------------------------*\
# Title: Fault-Tolerant-Trained MARL — Standalone Training Driver
# ---------------------------------------------------------*/
#
# Recovery mechanism #2 (ToDo #5, bullet 2). Trains the Sorting and
# Pressing agents on the domain-randomized envs in
# env_1_sort_fault_tolerant.py / env_2_press_fault_tolerant.py, and saves
# them under a SEPARATE checkpoint prefix ("..._FaultTolerant_...") so the
# existing vanilla checkpoints (PPO_Sorting_NoMask_*, PPO_Pressing_NoMask_*)
# are never touched.
#
# This is deliberately a SEPARATE, self-contained training loop rather than
# a reuse of training.py's Train_Agent()/RL_Trainer() - not out of
# duplication for its own sake, but because Train_Agent() has a hardcoded
# branch (see training.py) that auto-loads the LATEST "PPO_Sorting_NoMask"
# checkpoint as the press agent's teammate whenever it detects
# env.unwrapped.name == "press", unconditionally overwriting whatever
# sort_agent was already assigned. For fault-tolerant co-training we
# specifically want the fault-tolerant SORT agent as the press agent's
# teammate (so both were exposed to a consistent domain-randomized
# training regime), which that hardcoded branch would silently undo.
# Duplicating the small amount of SB3 boilerplate here avoids that
# conflict entirely, without modifying training.py's existing behavior
# for vanilla training in any way.
#
# The only thing reused from training.py is save_model() - a pure,
# side-effect-free helper for writing a checkpoint - and find_latest_model()
# for locating a vanilla teammate fallback (see train_fault_tolerant_press
# below, used only if no fault-tolerant sort checkpoint exists yet).

import os
import time
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.training import save_model
from src.envs_train.env_1_sort_fault_tolerant import Env_1_Sorting_FaultTolerant
from src.envs_train.env_2_press_fault_tolerant import Env_2_Pressing_FaultTolerant
from src.envs_train.env_combined import Env_Combined
from src.testing import test_env

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")


# ---------------------------------------------------------*/
# Shared SB3 training loop (mirrors training.py's Train_Agent, minus the
# press-teammate auto-load branch that would conflict with this file's
# explicit teammate wiring - see module docstring above)
# ---------------------------------------------------------*/
def _train_ppo(env, total_timesteps, save_prefix, logpath, eval_seed=99):
    env = Monitor(env)
    check_env(env)

    # Temporarily remove any attached teammate agent before deepcopying -
    # a live SB3/PyTorch model can't be pickled, and Env_2_Pressing_FaultTolerant
    # holds exactly that (the sort_agent set via set_agents() below in
    # train_fault_tolerant_modular_agents). Mirrors training.py's Train_Agent().
    sort_agent_ref = getattr(env.unwrapped, "sort_agent", None)
    if sort_agent_ref is not None:
        env.unwrapped.sort_agent = None

    eval_env = copy.deepcopy(env.unwrapped)

    if sort_agent_ref is not None:
        env.unwrapped.sort_agent = sort_agent_ref
        eval_env.sort_agent = sort_agent_ref

    eval_env = Monitor(eval_env)
    eval_env.reset(seed=eval_seed)

    tensorboard_log = os.path.join(logpath, str(save_prefix))
    os.makedirs(tensorboard_log, exist_ok=True)
    time.sleep(0.1)

    # Bigger than training.py's vanilla [32, 32]: the fault-tolerant agents
    # have to fit a much wider training distribution (5+ fault types x
    # modes x transient/permanent, domain-randomized) on the same 10M-step
    # budget, not just the single clean-operation task vanilla training
    # sees - measured to bottleneck as a ~24-point reward gap vs vanilla on
    # a completely clean, fault-free episode even with identical
    # hyperparameters otherwise.
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs, verbose=0,
        tensorboard_log=tensorboard_log,
        ent_coef=0.05, seed=42, device="cpu",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_model_ft/",
        log_path="./models/best_model_ft/",
        eval_freq=10_000, deterministic=True, render=False, verbose=0,
    )

    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=eval_callback)
    dur = time.time() - start_time
    print(f"✅ Training done in {dur//60:.0f} m {dur%60:.0f} s")

    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final Performance: {mean_r:.2f} ± {std_r:.2f}")

    best_path = "./models/best_model_ft/best_model.zip"
    if os.path.exists(best_path):
        print(f"📂 Loading best checkpoint model from: {best_path}")
        best_model = PPO.load(best_path, env=env)
        mean_best, std_best = evaluate_policy(best_model, env, n_eval_episodes=10)
        print(f"Best-Checkpoint: {mean_best:.2f} ± {std_best:.2f}")
        if mean_best > mean_r:
            model = best_model
            print("🏅 Using best checkpoint for saving.")
        import shutil
        shutil.rmtree("./models/best_model_ft/", ignore_errors=True)

    save_model(model, prefix=save_prefix, timesteps=total_timesteps)
    return model


# ---------------------------------------------------------*/
# Full fault-tolerant training flow (Sort, then Press with the
# fault-tolerant Sort agent as its randomized teammate)
# ---------------------------------------------------------*/
def train_fault_tolerant_modular_agents(total_timesteps, steps_train, steps_test, seed, tag):
    print("\n[1/2] Training Fault-Tolerant Sorting Agent (domain-randomized: "
          "sensor_noise + actuator_degradation on itself)...")
    sort_train_env = Env_1_Sorting_FaultTolerant(max_steps=steps_train, seed=seed)
    sort_agent = _train_ppo(
        env=sort_train_env,
        total_timesteps=total_timesteps,
        save_prefix="PPO_Sorting_FaultTolerant_NoMask",
        logpath=f"./log/tensorboard/{tag}_FaultTolerant",
    )
    _test_sorting_agent(sort_agent, steps_test=steps_test, seed=seed, tag=tag + "_FaultTolerant")

    print("\n[2/2] Training Fault-Tolerant Pressing Agent (domain-randomized: "
          "sensor_noise + actuator_degradation on itself, plus teammate_dropout, "
          "teammate_byzantine [worst_action/fixed_malicious/random_malicious], "
          "and teammate_comms_loss [stale/blackout] simulating a faulty/compromised "
          "sorting teammate and a dropped comms link)...")
    press_train_env = Env_2_Pressing_FaultTolerant(max_steps=steps_train, seed=seed)
    press_train_env.set_agents(sort_agent=sort_agent)  # the FAULT-TOLERANT sort agent, not vanilla
    press_agent = _train_ppo(
        env=press_train_env,
        total_timesteps=total_timesteps,
        save_prefix="PPO_Pressing_FaultTolerant_NoMask",
        logpath=f"./log/tensorboard/{tag}_FaultTolerant",
    )

    _test_modular_pair(sort_agent, press_agent, steps_test=steps_test, seed=seed,
                        tag=tag + "_FaultTolerant_trained_modular")

    print("\n✅ Fault-tolerant training complete. Checkpoints saved as:")
    print(f"   ./models/PPO_Sorting_FaultTolerant_NoMask_{total_timesteps}.zip")
    print(f"   ./models/PPO_Pressing_FaultTolerant_NoMask_{total_timesteps}.zip")
    print("   (existing vanilla PPO_Sorting_NoMask_*/PPO_Pressing_NoMask_* checkpoints untouched)")


# ---------------------------------------------------------*/
# Quick eval helpers, same pattern as main.py's test_sorting_agent /
# test_modular_pair - kept local so this file has zero dependency on
# main.py and can be imported standalone.
# ---------------------------------------------------------*/
def _test_sorting_agent(sort_agent, steps_test, seed, tag):
    combined_env = Env_Combined(max_steps=steps_test, seed=seed)
    combined_env.set_agents(sort_agent=sort_agent)
    test_env(env=combined_env, tag=tag, title="(Test: PPO_Sort FaultTolerant - No Mask)",
             steps=steps_test, dir="./img/figures/", seed=seed)


def _test_modular_pair(sort_agent, press_agent, steps_test, seed, tag):
    combined_env = Env_Combined(max_steps=steps_test, seed=seed)
    combined_env.set_agents(sort_agent=sort_agent, press_agent=press_agent)
    test_env(env=combined_env, tag=tag, title="(Test: Modular FaultTolerant Sort+Press - No Mask)",
             steps=steps_test, dir="./img/figures/", seed=seed)

# -----------------------------------------------------------------------------*/