# ---------------------------------------------------------*/
# Title: MARL - Modular Agents (No Masking) - Main
# ---------------------------------------------------------*/
import os
from datetime import datetime

# Environments
from src.envs_train.env_1_sort import Env_1_Sorting
from src.envs_train.env_2_press import Env_2_Pressing
from src.envs_train.env_combined import Env_Combined
from src.envs_train.env_fault_sensor_noise import Env_SensorNoiseFault
from src.envs_train.env_fault_actuator_degradation import Env_ActuatorDegradationFault
from src.envs_train.env_fault_agent_dropout import Env_AgentDropoutFault
from src.envs_train.env_fault_comms_loss import Env_CommsLossFault
from src.envs_train.env_fault_byzantine import Env_ByzantineFault
from src.recovery.rule_based import RULE_BASED_REGISTRY
from src.recovery.rule_based_detect import RULE_BASED_DETECT_REGISTRY
from src.recovery.supervisor import SUPERVISOR_REGISTRY


# RL: Trainer / Tester
from src.testing import test_env
from src.training import RL_Trainer, find_latest_model

# ---------------------------------------------------------*/
# Parameters
# ---------------------------------------------------------*/

TAG = f"Gold_{datetime.now().strftime('%d-%m-%Y_%H-%M')}_NoMask"

TRAIN_MODULAR = 0   # Train the Sorting + Pressing agents (no action masking)
RUN_MODULAR = 1     # Load the trained NoMask modular agents and simulate them together

# Fault types implemented so far -> class. Anything else here is a
# placeholder for a fault mechanism not built yet (see FAULT_STUBS below).
FAULT_REGISTRY = {
    "none": None,
    "sensor_noise": Env_SensorNoiseFault,
    "actuator_degradation": Env_ActuatorDegradationFault,
    "agent_dropout": Env_AgentDropoutFault,
    "comms_loss": Env_CommsLossFault,
    "byzantine": Env_ByzantineFault,
}

# Faults planned but not yet implemented (kept here so the prompt already
# lists the full fault suite from the proposal; implement + move into
# FAULT_REGISTRY above as each one is built).
FAULT_STUBS = set()

FAULT_OPTIONS = ["none", "sensor_noise", "actuator_degradation", "agent_dropout", "comms_loss", "byzantine"]

# Recovery mechanisms implemented so far -> {fault_type: recovery-wrapped
# env class}. Each entry mirrors FAULT_REGISTRY's keys (minus "none") but
# points at the fault class combined with that recovery mechanism's mixin
# (see src/recovery/rule_based.py for how the mixin/combo pattern works).
RECOVERY_REGISTRY = {
    "rule_based": RULE_BASED_REGISTRY,               # oracle: reads ground-truth fault_context
    "rule_based_detect": RULE_BASED_DETECT_REGISTRY,  # must detect the fault itself from observable signals
    "supervisor": SUPERVISOR_REGISTRY,                # learned policy monitors + intervenes (set_supervisor below)
}

# Recovery mechanisms planned but not yet implemented (ToDo #5, bullets 4).
RECOVERY_STUBS = {
    "llm_replanning",
}

RECOVERY_OPTIONS = ["none", "rule_based", "rule_based_detect", "fault_tolerant_marl", "supervisor", "llm_replanning"]

# Default fault config used when a fault (other than "none") is selected.
# Tune per-experiment; see marl-sortingenv fault-injection plan.
# Shared keys (injection_step_range/duration/target/seed) apply to every
# fault type; fault-specific keys (e.g. noise_std for sensor_noise, mode/
# stuck_*/restrict_*/degradation_prob for actuator_degradation) are simply
# ignored by fault types that don't use them.
DEFAULT_FAULT_CONFIG = {
    "injection_step_range": (40, 60),
    "duration": None,        # fixed transient duration in steps; None = see duration_range
    "duration_range": None,  # (lo, hi) -> a duration is sampled per-episode from this range;
                              # only used when "duration" above is None. Both None = permanent
                              # (fault persists for the rest of the episode once injected).
    "target": "both",       # "sort", "press", or "both"
    "seed": 123,

    # sensor_noise
    "noise_std": 0.1,

    # actuator_degradation
    "mode": "stuck",              # "stuck" | "restrict" | "slip"
    "stuck_press_action": 0,      # press seizes idle (no-op) once stuck
    "stuck_sort_mode": None,      # None -> freeze at whatever mode was active on fault onset
    "restrict_sort_mode": 1,      # used only when mode == "restrict"
    "restrict_press_id": 2,       # used only when mode == "restrict"
    "degradation_prob": 0.3,      # used only when mode == "slip"

    # agent_dropout
    "dropout_sort_mode": 0,       # failsafe sort mode while the sort agent is dropped
    "dropout_press_action": 0,    # failsafe press action (no-op) while the press agent is dropped

    # comms_loss (target is forced to "press" internally - see env_fault_comms_loss.py)
    "comms_mode": "blackout",        # "stale" | "blackout" - intentionally a separate key
                                   # from actuator_degradation's "mode" above, since both
                                   # faults share this same DEFAULT_FAULT_CONFIG dict and a
                                   # colliding key would silently overwrite one another.

    # byzantine
    "byzantine_mode": "worst_action",  # "fixed_malicious" | "worst_action" | "random_malicious"
    "malicious_sort_mode": None,       # used only by fixed_malicious; None -> latch from heuristic
    "malicious_press_action": None,    # used only by fixed_malicious; None -> latch from heuristic
}

RECOVERY_MODE = "none"  # default; only ever overridden below when a fault is actually injected

if input("Train or Test? (train/test): ") == "train":
    TRAIN_MODULAR = 1
    RUN_MODULAR = 0
    FAULT_MODE = "none"  # fault injection only applies during RUN_MODULAR eval

    TRAIN_VARIANT = input(
        "Train which variant? (vanilla/fault_tolerant/supervisor) [vanilla]: "
    ).strip().lower()
    if TRAIN_VARIANT in ("fault_tolerant", "ft"):
        TRAIN_VARIANT = "fault_tolerant"
    elif TRAIN_VARIANT in ("supervisor", "supervisor_agent", "sup", "sv"):
        TRAIN_VARIANT = "supervisor"
    else:
        TRAIN_VARIANT = "vanilla"
else:
    TRAIN_MODULAR = 0
    RUN_MODULAR = 1
    TRAIN_VARIANT = "vanilla"  # unused outside the train branch

    print(f"\nAvailable faults: {', '.join(FAULT_OPTIONS)}")
    FAULT_MODE = input("Inject which fault? (none/sensor_noise/actuator_degradation/agent_dropout/comms_loss/byzantine): ").strip()

    if FAULT_MODE not in FAULT_OPTIONS:
        print(f"⚠️ Unrecognized fault '{FAULT_MODE}', defaulting to 'none'.")
        FAULT_MODE = "none"

    if FAULT_MODE in FAULT_STUBS:
        raise NotImplementedError(
            f"Fault '{FAULT_MODE}' is planned but not implemented yet. "
            f"Currently implemented: {[k for k in FAULT_REGISTRY if k != 'none']}"
        )

    if FAULT_MODE != "none":
        DEFAULT_TRANSIENT_DURATION = 20  # steps, used when Enter is pressed with no value

        duration_choice = input(
            "Transient or Permanent fault? (transient/permanent) [permanent]: "
        ).strip().lower()

        if duration_choice in ("transient", "t"):
            raw = input(
                f"Duration in steps, or a range 'lo-hi' sampled per episode "
                f"[Enter for default {DEFAULT_TRANSIENT_DURATION}]: "
            ).strip()

            if raw == "":
                DEFAULT_FAULT_CONFIG["duration"] = DEFAULT_TRANSIENT_DURATION
                DEFAULT_FAULT_CONFIG["duration_range"] = None
            elif "-" in raw:
                try:
                    lo_str, hi_str = raw.split("-", 1)
                    lo, hi = int(lo_str), int(hi_str)
                    if lo <= 0 or hi < lo:
                        raise ValueError
                    DEFAULT_FAULT_CONFIG["duration"] = None
                    DEFAULT_FAULT_CONFIG["duration_range"] = (lo, hi)
                except ValueError:
                    print(f"⚠️ Couldn't parse range '{raw}', using default duration={DEFAULT_TRANSIENT_DURATION}.")
                    DEFAULT_FAULT_CONFIG["duration"] = DEFAULT_TRANSIENT_DURATION
                    DEFAULT_FAULT_CONFIG["duration_range"] = None
            else:
                try:
                    val = int(raw)
                    if val <= 0:
                        raise ValueError
                    DEFAULT_FAULT_CONFIG["duration"] = val
                    DEFAULT_FAULT_CONFIG["duration_range"] = None
                except ValueError:
                    print(f"⚠️ Couldn't parse '{raw}', using default duration={DEFAULT_TRANSIENT_DURATION}.")
                    DEFAULT_FAULT_CONFIG["duration"] = DEFAULT_TRANSIENT_DURATION
                    DEFAULT_FAULT_CONFIG["duration_range"] = None

            print(f"-> Transient fault, duration={DEFAULT_FAULT_CONFIG['duration']}, "
                  f"duration_range={DEFAULT_FAULT_CONFIG['duration_range']}")
        else:
            # Permanent: persists for the rest of the episode once injected.
            DEFAULT_FAULT_CONFIG["duration"] = None
            DEFAULT_FAULT_CONFIG["duration_range"] = None
            print("-> Permanent fault (persists for the rest of the episode).")

        print(f"\nAvailable recovery mechanisms: {', '.join(RECOVERY_OPTIONS)}")
        RECOVERY_MODE = input(
            "Apply which recovery mechanism? (none/rule_based/rule_based_detect/fault_tolerant_marl/supervisor/llm_replanning) [none]: "
        ).strip()

        if RECOVERY_MODE == "":
            RECOVERY_MODE = "none"
        if RECOVERY_MODE not in RECOVERY_OPTIONS:
            print(f"⚠️ Unrecognized recovery mechanism '{RECOVERY_MODE}', defaulting to 'none'.")
            RECOVERY_MODE = "none"
        if RECOVERY_MODE in RECOVERY_STUBS:
            raise NotImplementedError(
                f"Recovery mechanism '{RECOVERY_MODE}' is planned but not implemented yet. "
                f"Currently implemented: {list(RECOVERY_REGISTRY.keys())}"
            )

TOTAL_TIMESTEPS = 10_000_000
# Supervisor policy budget: it only needs to learn a 4-action intervention
# gating policy over two frozen agents, so far fewer steps than the 10M used
# to train the modular pair suffice.
SUPERVISOR_TOTAL_TIMESTEPS = 5_000_000
STEPS_TRAIN = 200
STEPS_TEST = 200
SEED = 42

SAVE = 1
DIR = "./img/figures/"


# ---------------------------------------------------------*/
# Run
# ---------------------------------------------------------*/

def run_sim(
    TRAIN_MODULAR=TRAIN_MODULAR,
    TRAIN_VARIANT=TRAIN_VARIANT,
    RUN_MODULAR=RUN_MODULAR,
    TOTAL_TIMESTEPS=TOTAL_TIMESTEPS,
    STEPS_TRAIN=STEPS_TRAIN,
    STEPS_TEST=STEPS_TEST,
    SEED=SEED,
    TAG=TAG,
    FAULT_MODE=FAULT_MODE,
    RECOVERY_MODE=RECOVERY_MODE,
):
    print("\n--------------------------------")
    print("Starting Simulation... 🚀")
    print("--------------------------------")

    if TRAIN_MODULAR:
        if TRAIN_VARIANT == "fault_tolerant":
            print("\n--- Training FAULT-TOLERANT Modular Agents (domain-randomized), No Masking ---")
            from src.training_fault_tolerant import train_fault_tolerant_modular_agents
            train_fault_tolerant_modular_agents(
                total_timesteps=TOTAL_TIMESTEPS,
                steps_train=STEPS_TRAIN,
                steps_test=STEPS_TEST,
                seed=SEED,
                tag=TAG,
            )
        elif TRAIN_VARIANT == "supervisor":
            print("\n--- Training SUPERVISOR Agent (learned intervention over the vanilla modular pair) ---")
            from src.training_supervisor import train_supervisor
            train_supervisor(
                total_timesteps=SUPERVISOR_TOTAL_TIMESTEPS,
                steps_train=STEPS_TRAIN,
                steps_test=STEPS_TEST,
                seed=SEED,
                tag=TAG,
            )
        else:
            print("\n--- Training Modular Agents (Sorting + Pressing), No Masking ---")
            train_modular_agents(
                total_timesteps=TOTAL_TIMESTEPS,
                steps_train=STEPS_TRAIN,
                steps_test=STEPS_TEST,
                seed=SEED,
                tag=TAG,
            )

    if RUN_MODULAR:
        print(f"\n--- Running Trained Modular (No-Mask) Agents [fault: {FAULT_MODE}, recovery: {RECOVERY_MODE}] ---")
        run_trained_modular_agents(steps_test=STEPS_TEST, seed=SEED, tag=TAG,
                                    fault_mode=FAULT_MODE, recovery_mode=RECOVERY_MODE)

    print("\n--------------------------------")
    print("Simulation Completed. 🌵")
    print("--------------------------------")


# ---------------------------------------------------------*/
# Training Flow
# ---------------------------------------------------------*/
def train_modular_agents(total_timesteps, steps_train, steps_test, seed, tag):
    """
    Trains the Sorting agent, then the Pressing agent (which loads the
    just-trained Sorting agent), both as plain unmasked PPO.
    """
    # --- 1. Train Sorting Agent ---
    print("\n[1/2] Training Sorting Agent...")
    sort_train_env = Env_1_Sorting(max_steps=steps_train, seed=seed)
    sort_agent = RL_Trainer(
        env=sort_train_env, env_class="Sorting",
        total_timesteps=total_timesteps, max_steps=steps_train, tag=tag, seed=seed,
    )
    test_sorting_agent(sort_agent, steps_test=steps_test, seed=seed, tag=tag)

    # --- 2. Train Pressing Agent (with the trained Sorting agent assigned) ---
    print("\n[2/2] Training Pressing Agent...")
    press_train_env = Env_2_Pressing(max_steps=steps_train, seed=seed)
    press_train_env.set_agents(sort_agent=sort_agent)
    press_agent = RL_Trainer(
        env=press_train_env, env_class="Pressing",
        total_timesteps=total_timesteps, max_steps=steps_train, tag=tag, seed=seed,
    )

    # --- Test the modular pair together ---
    test_modular_pair(sort_agent, press_agent, steps_test=steps_test, seed=seed, tag=tag + "_trained_modular")


def test_sorting_agent(sort_agent, steps_test, seed, tag):
    combined_env = Env_Combined(max_steps=steps_test, seed=seed)
    combined_env.set_agents(sort_agent=sort_agent)
    test_env(env=combined_env, tag=tag, title="(Test: PPO_Sort - No Mask)",
             steps=steps_test, dir=DIR, seed=seed)


def test_modular_pair(sort_agent, press_agent, steps_test, seed, tag):
    combined_env = Env_Combined(max_steps=steps_test, seed=seed)
    combined_env.set_agents(sort_agent=sort_agent, press_agent=press_agent)
    test_env(env=combined_env, tag=tag, title="(Test: Modular Sort+Press - No Mask)",
             steps=steps_test, dir=DIR, seed=seed)


# ---------------------------------------------------------*/
# Run Trained Models
# ---------------------------------------------------------*/
def run_trained_modular_agents(steps_test, seed, tag, fault_mode="none", recovery_mode="none"):
    from stable_baselines3 import PPO

    if recovery_mode == "fault_tolerant_marl":
        # "Recovery" here is baked into the weights via domain-randomized
        # training (see src/training_fault_tolerant.py) - no recovery_hook
        # correction is applied at eval time, so these load into the SAME
        # plain fault env everything else uses (see env_class selection below).
        sort_path = "./models/PPO_Sorting_FaultTolerant_NoMask_10000000.zip"
        press_path = "./models/PPO_Pressing_FaultTolerant_NoMask_10000000.zip"
    else:
        sort_path = "./models/PPO_Sorting_NoMask_10000000.zip"
        press_path = "./models/PPO_Pressing_NoMask_10000000.zip"

    if not (os.path.exists(sort_path) and os.path.exists(press_path)):
        print(f"⚠️ Models not found. Please ensure both {sort_path} and {press_path} exist.")
        if recovery_mode == "fault_tolerant_marl":
            print("   (Run 'Train or Test? -> train' then choose 'fault_tolerant' to produce them.)")
        return

    sort_model = PPO.load(sort_path)
    press_model = PPO.load(press_path)

    if fault_mode == "none":
        # No fault -> plain Env_Combined, no fault injected, no recovery involved
        # (a recovery mechanism has nothing to recover from without a fault).
        env_class = None
    elif recovery_mode in ("none", "fault_tolerant_marl"):
        # fault_tolerant_marl uses the same plain fault env as "none" -
        # its robustness lives in the loaded weights above, not an
        # action-level correction, so no recovery mixin is needed here.
        env_class = FAULT_REGISTRY.get(fault_mode)
    else:
        # Recovery-wrapped fault class, e.g. Env_ByzantineFault_RuleBased.
        # RECOVERY_REGISTRY[recovery_mode] maps fault_type -> combo class
        # (see src/recovery/rule_based.py). Every implemented fault type
        # has an entry for every implemented recovery mechanism.
        env_class = RECOVERY_REGISTRY[recovery_mode][fault_mode]

    if env_class is None:
        env = Env_Combined(max_steps=steps_test, seed=seed)
        eval_tag = tag + "_trained_modular"
        title_suffix = ""
    else:
        env = env_class(max_steps=steps_test, seed=seed, fault_config=DEFAULT_FAULT_CONFIG)
        if recovery_mode == "none":
            eval_tag = tag + f"_trained_modular_fault-{fault_mode}"
            title_suffix = f" [fault: {fault_mode}]"
        else:
            eval_tag = tag + f"_trained_modular_fault-{fault_mode}_recovery-{recovery_mode}"
            title_suffix = f" [fault: {fault_mode}, recovery: {recovery_mode}]"

    env.set_agents(sort_agent=sort_model, press_agent=press_model)

    if recovery_mode == "supervisor":
        # The supervisor policy is trained to recover the vanilla modular pair
        # (see src/training_supervisor.py). Attach the latest checkpoint so the
        # combo env's recovery_hook() derives its intervention from predict().
        supervisor_path = find_latest_model("PPO_Supervisor_NoMask")
        if not supervisor_path:
            print("⚠️ No trained Supervisor model found. Train one via 'Train or Test? -> train' then 'supervisor'.")
            return
        supervisor_model = PPO.load(supervisor_path)
        env.set_supervisor(supervisor_model)
        print(f"📂 Loading Supervisor model from: {supervisor_path}")

    test_env(env=env, tag=eval_tag, save=SAVE, show=True,
             title=f"Trained Modular Agents (No Mask){title_suffix}",
             steps=steps_test, dir=DIR, seed=seed)

    if env_class is not None:
        print(f"\nFault log for this run: {env.fault_log}")
    if recovery_mode == "supervisor" and env_class is not None:
        print(f"Supervisor decision log: {env.supervisor_decision_log}")


# ---------------------------------------------------------*/
# Main
# ---------------------------------------------------------*/
if __name__ == "__main__":
    run_sim(TAG=TAG, FAULT_MODE=FAULT_MODE, RECOVERY_MODE=RECOVERY_MODE, TRAIN_VARIANT=TRAIN_VARIANT)

# -------------------------Notes-----------------------------------------------*\
# -----------------------------------------------------------------------------