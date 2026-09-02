"""
Prints the actual error messages from a handful of failed llm_replanning
calls -- stop guessing at rate-limit vs config vs malformed-tool-call and
just read what the API is actually saying.
"""
from src.recovery.llm_replanning import LLM_REGISTRY
from stable_baselines3 import PPO

STEPS, SEED = 200, 0
FAULT_CONFIG = {"injection_step_range": (40, 60), "duration": None,
                 "duration_range": None, "target": "both", "seed": SEED}

sort_model = PPO.load("./models/PPO_Sorting_NoMask_10000000.zip")
press_model = PPO.load("./models/PPO_Pressing_NoMask_10000000.zip")

env = LLM_REGISTRY["actuator_degradation"](max_steps=STEPS, seed=SEED, fault_config=dict(FAULT_CONFIG))
env.set_agents(sort_agent=sort_model, press_agent=press_model)
env.reset(seed=SEED)
for _ in range(STEPS):
    _, _, term, trunc, _ = env.step()
    if term or trunc:
        break

log = getattr(env, "llm_decision_log", [])
errors = [d for d in log if d.get("error")]
print(f"{len(errors)}/{len(log)} calls errored\n")
for d in errors[:10]:
    print(f"step={d.get('step')}  error={d.get('error')!r}")
