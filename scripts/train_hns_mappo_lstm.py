"""
train_hns_mappo_lstm.py
-----------------------
MAPPO + LSTM on the Hide-and-Seek (HNS) hidenseek scenario, built on MARLlib.

Hardware target
---------------
  GPU  : NVIDIA A100-SXM4-80 GB
  CPU  : 256 cores
  RAM  : 2 TB

Worker / batch rationale
------------------------
  num_workers  = 32
      32 parallel CPU rollout workers.  Each worker runs one full 6-agent HNS
      episode (300 steps) at a time.  With 256 CPUs available this is
      conservative, leaving room for Ray overhead and the trainer process, and
      can be increased via HNS_NUM_WORKERS if collection becomes the bottleneck.

  batch_episode = 64  →  train_batch_size = 64 × 300 = 19 200 env-steps
      With the shared-policy and agent_level_batch_update=True, all 6 agents'
      trajectories are treated independently, giving ~115 200 agent-steps per
      SGD update.  This is well within A100 capacity and provides low-variance
      gradient estimates for the complex HNS task.

  num_sgd_iter = 10
      10 update epochs per collected batch — standard for MAPPO in complex
      environments (SMAC, MPE).  PPO's clip prevents destructive over-fitting.

Other hyperparameter choices
-----------------------------
  lambda = 0.95   (GAE)  – better bias-variance balance than the default λ=1.
  clip_param = 0.2       – tighter clip (default 0.3) for training stability.
  entropy_coeff = 0.01   – standard; keeps the policy sufficiently stochastic.
  lr = 3e-4              – as specified.

TensorBoard
-----------
  Ray Tune writes logs under RESULTS_DIR automatically.  After starting the
  training run, open TensorBoard with:

      tensorboard --logdir runs/hns_mappo_lstm_80m_tensorboard

Note on share_policy="all"
--------------------------
  HNS is a mixed cooperative-competitive game (hiders vs seekers) with
  share_reward=False, so the two teams receive opposing reward signals.
  Sharing one policy across all 6 agents means the network must learn to
  condition on its role via observation content.  This required setting
  all_agents_one_policy=True in marllib/envs/base_env/hns.py.

Note on epsilon_timesteps / final_epsilon
------------------------------------------
  These are epsilon-greedy parameters used by value-based methods (IQL, QMIX).
  They have no effect on MAPPO, which uses entropy regularisation for
  exploration.  They are not passed to the algorithm here.
"""

# ── Compatibility patches: Python 3.11 + Ray 2.3.0 + numpy 2.0 ─────────────
#
# 1. numpy 2.0 removed several type aliases (bool8, float, int, …) that
#    mujoco_worldgen and Ray 2.3.0 rely on.  Restore them before any import.
#
# 2. Ray 2.3.0's ray.data package crashes on Python 3.11 due to a mutable
#    dataclass default.  We stub out ray.data and ray.train entirely (MARLlib
#    only uses ray.rllib + ray.tune for RL training; ray.data is not needed).
#    A sys.meta_path hook intercepts every "ray.data.*" / "ray.train.*" import
#    and returns an empty stub module, preventing the broken code from running.
import sys as _sys
import types as _types
import numpy as _np

# Restore numeric numpy aliases removed in numpy 2.0
for _attr, _alias in [
    ('bool8',      _np.bool_),
    ('float',      float),
    ('int',        int),
    ('complex',    complex),
    ('product',    _np.prod),
    ('cumproduct', _np.cumprod),
    ('sometrue',   _np.any),
    ('alltrue',    _np.all),
]:
    if not hasattr(_np, _attr):
        setattr(_np, _attr, _alias)
del _attr, _alias, _np

# Ensure MARLlib is importable from any working directory
import os as _os, pathlib as _pl
_repo = str(_pl.Path(__file__).resolve().parents[1])
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
del _os, _pl, _repo


class _StubModule(_types.ModuleType):
    """Minimal stub module that returns child stubs on attribute access."""
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        child = _StubModule(self.__name__ + '.' + name)
        child.__file__ = '<stub>'
        child.__path__ = []
        child.__package__ = self.__name__
        child.__spec__ = None
        object.__setattr__(self, name, child)
        return child


class _RayDataTrainStubFinder:
    """sys.meta_path hook: intercept ray.data.* and ray.train.* imports."""
    _STUB_PREFIXES = ('ray.data', 'ray.train')

    def find_module(self, fullname, path=None):
        if any(fullname == p or fullname.startswith(p + '.')
               for p in self._STUB_PREFIXES):
            return self
        return None

    def load_module(self, fullname):
        if fullname in _sys.modules:
            return _sys.modules[fullname]
        m = _StubModule(fullname)
        m.__file__ = '<stub>'
        m.__path__ = []
        m.__package__ = fullname.rpartition('.')[0] or fullname
        m.__spec__ = None
        _sys.modules[fullname] = m
        return m


_sys.meta_path.insert(0, _RayDataTrainStubFinder())
del _types
# ────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

from marllib import marl


# ── configurable via environment variables ───────────────────────────────────
RESULTS_DIR = os.path.abspath(
    os.environ.get("HNS_MAPPO_RESULTS_DIR", "runs/hns_mappo_lstm_80m_tensorboard")
)
STOP_TIMESTEPS  = int(os.environ.get("HNS_STOP_TIMESTEPS",  "80000000"))
NUM_GPUS        = int(os.environ.get("HNS_NUM_GPUS",        "1"))
NUM_WORKERS     = int(os.environ.get("HNS_NUM_WORKERS",     "24"))
CHECKPOINT_FREQ = int(os.environ.get("HNS_CHECKPOINT_FREQ", "100"))
SEED            = int(os.environ.get("HNS_SEED",            "321"))
RESTORE_MODEL_PATH  = os.environ.get("HNS_RESTORE_MODEL_PATH",  "")
RESTORE_PARAMS_PATH = os.environ.get("HNS_RESTORE_PARAMS_PATH", "")


def build_env():
    env, env_config = marl.make_env(
        "hns",
        "hidenseek",
        force_coop=False,
        # ── map layout ────────────────────────────────────────────────────
        scenario_name="quadrant",   # required for hidenseek
        task_type="all-return",
        num_agents=6,
        num_seekers=3,
        num_hiders=3,
        num_boxes=6,
        num_ramps=4,
        num_food=0,
        floor_size=18.0,
        grid_size=90,
        fixed_door=True,
        spawn_obs=False,
        env_horizon=300,
        share_reward=False,
        seed=SEED,
    )
    return env, env_config


def verify_env():
    env, _ = build_env()
    try:
        if env.num_agents != 6:
            raise ValueError("Expected 6 agents, got {}".format(env.num_agents))
        if env.env_config.get("env_horizon") != 300:
            raise ValueError("Expected env_horizon 300, got {}".format(
                env.env_config.get("env_horizon")))
        print("verified  hns hidenseek  |  agents=6  |  horizon=300")
    finally:
        env.close()


def latest_checkpoint():
    stage_dir = os.path.join(RESULTS_DIR, "mappo_lstm_hidenseek")
    checkpoints = sorted(Path(stage_dir).glob("*/checkpoint_*/checkpoint-*"))
    checkpoints = [p for p in checkpoints if not str(p).endswith(".tune_metadata")]
    if not checkpoints:
        return None
    model_path  = str(checkpoints[-1])
    params_path = str(checkpoints[-1].parents[1] / "params.json")
    return {"model_path": model_path, "params_path": params_path}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    verify_env()
    env_tuple = build_env()   # (env_instance, env_config) tuple

    # ── MAPPO hyperparameters ─────────────────────────────────────────────────
    # "lambda" is a Python keyword, so pass it via dict unpacking.
    algo_overrides = {
        # batch: 64 episodes × 300 steps = 19 200 env-steps per update
        "batch_episode": 64,
        # learning
        "lr": 3e-4,
        "entropy_coeff": 0.01,
        "num_sgd_iter": 10,
        # PPO clipping
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "vf_loss_coeff": 1.0,
        # GAE
        "use_gae": True,
        "lambda": 0.95,     # dict key avoids Python keyword conflict
        # KL penalty
        "kl_coeff": 0.2,
    }
    mappo = marl.algos.mappo("common", **algo_overrides)

    # ── model: FC encoder 256-256 → LSTM → action / value heads ──────────────
    model = marl.build_model(
        env_tuple, mappo,
        {"core_arch": "lstm", "encode_layer": "256-256"},
    )

    # ── launch ────────────────────────────────────────────────────────────────
    # TensorBoard logs are written automatically to local_dir by Ray Tune.
    # Start TensorBoard with:
    #   tensorboard --logdir runs/hns_mappo_lstm_80m_tensorboard
    mappo.fit(
        env_tuple,
        model,
        stop={"timesteps_total": STOP_TIMESTEPS},
        local_mode=False,           # False = distributed; True = debug only
        num_gpus=NUM_GPUS,          # A100 80 GB handles all SGD updates
        num_workers=NUM_WORKERS,    # 32 CPU workers for parallel rollout
        share_policy="all",         # all 6 agents share one LSTM policy
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_end=True,
        local_dir=RESULTS_DIR,
        seed=SEED,
        restore_path={
            "model_path": RESTORE_MODEL_PATH,
            "params_path": RESTORE_PARAMS_PATH,
        },
    )

    print("finished hidenseek  |  checkpoint:", latest_checkpoint())


if __name__ == "__main__":
    main()
