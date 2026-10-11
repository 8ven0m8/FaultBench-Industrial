# ---------------------------------------------------------*\
# Title: Checkpoint naming for training-seed replicates
# ---------------------------------------------------------*/
#
# fault_tolerant_marl and supervisor are trained once per TRAINING seed
# (see train_seeds.py), so a reviewer can tell a paradigm's real effect
# apart from the variance of a single training run. Each replicate gets
# its own checkpoint name with a "_ts<seed>" tag:
#
#   PPO_Sorting_FaultTolerant_ts<k>_NoMask_<timesteps>.zip
#   PPO_Pressing_FaultTolerant_ts<k>_NoMask_<timesteps>.zip
#   PPO_Supervisor_ts<k>_NoMask_<timesteps>.zip
#
# The tag sits BEFORE "_NoMask" on purpose: find_latest_model() globs
# "<prefix>_*.zip" and save_model() archives files that start with the
# prefix, so a name like "PPO_Supervisor_NoMask_ts1_..." would be picked
# up by the legacy "PPO_Supervisor_NoMask" lookups (and archived away by a
# legacy retrain). With the tag first, the legacy single-seed names and
# every seeded replicate stay fully separate.
#
# train_seed=None everywhere below means the legacy single-seed checkpoint
# (PPO_Sorting_FaultTolerant_NoMask_10000000.zip etc.), kept working so
# main.py's interactive mode and older checkpoints still load.

import glob
import os
import re

from src.training import find_latest_model

MODELS_DIR = "./models"

VANILLA_SORT_PATH = "./models/PPO_Sorting_NoMask_10000000.zip"
VANILLA_PRESS_PATH = "./models/PPO_Pressing_NoMask_10000000.zip"
LEGACY_FT_SORT_PATH = "./models/PPO_Sorting_FaultTolerant_NoMask_10000000.zip"
LEGACY_FT_PRESS_PATH = "./models/PPO_Pressing_FaultTolerant_NoMask_10000000.zip"
LEGACY_SUPERVISOR_PREFIX = "PPO_Supervisor_NoMask"


def ft_prefixes(train_seed=None):
    """(sort_prefix, press_prefix) passed to save_model() for this replicate."""
    if train_seed is None:
        return "PPO_Sorting_FaultTolerant_NoMask", "PPO_Pressing_FaultTolerant_NoMask"
    return (f"PPO_Sorting_FaultTolerant_ts{train_seed}_NoMask",
            f"PPO_Pressing_FaultTolerant_ts{train_seed}_NoMask")


def supervisor_prefix(train_seed=None):
    if train_seed is None:
        return LEGACY_SUPERVISOR_PREFIX
    return f"PPO_Supervisor_ts{train_seed}_NoMask"


def find_ft_pair(train_seed=None):
    """(sort_path, press_path) for one fault-tolerant replicate, or None if
    either half is missing."""
    if train_seed is None:
        paths = (LEGACY_FT_SORT_PATH, LEGACY_FT_PRESS_PATH)
        return paths if all(os.path.exists(p) for p in paths) else None
    sort_prefix, press_prefix = ft_prefixes(train_seed)
    sort_path, press_path = find_latest_model(sort_prefix), find_latest_model(press_prefix)
    if sort_path and press_path:
        return sort_path, press_path
    return None


def find_supervisor(train_seed=None):
    return find_latest_model(supervisor_prefix(train_seed))


def _seeds_matching(regex):
    seeds = set()
    for path in glob.glob(os.path.join(MODELS_DIR, "*.zip")):
        m = re.fullmatch(regex, os.path.basename(path))
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds)


def available_ft_train_seeds():
    """Training seeds for which BOTH the sort and press FT checkpoints exist."""
    sort_seeds = _seeds_matching(r"PPO_Sorting_FaultTolerant_ts(\d+)_NoMask_\d+\.zip")
    press_seeds = set(_seeds_matching(r"PPO_Pressing_FaultTolerant_ts(\d+)_NoMask_\d+\.zip"))
    return [s for s in sort_seeds if s in press_seeds]


def available_supervisor_train_seeds():
    return _seeds_matching(r"PPO_Supervisor_ts(\d+)_NoMask_\d+\.zip")


def default_ft_pair():
    """For single-run tools (main.py, verify_workings.py): the legacy
    checkpoint if present, else the lowest-numbered seeded replicate.
    Returns (sort_path, press_path, train_seed) or None."""
    legacy = find_ft_pair(None)
    if legacy:
        return legacy[0], legacy[1], None
    for seed in available_ft_train_seeds():
        pair = find_ft_pair(seed)
        if pair:
            return pair[0], pair[1], seed
    return None


def default_supervisor():
    """Same fallback rule as default_ft_pair(). Returns (path, train_seed) or None."""
    legacy = find_supervisor(None)
    if legacy:
        return legacy, None
    for seed in available_supervisor_train_seeds():
        path = find_supervisor(seed)
        if path:
            return path, seed
    return None

# -----------------------------------------------------------------------------*/
