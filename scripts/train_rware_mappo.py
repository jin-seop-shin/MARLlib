import os
from pathlib import Path

from ray import tune
import gymnasium as gymnasium
import numpy as np
import rware  # noqa: F401 - registers rware:* Gymnasium environments
from gym.spaces import Box, Dict as GymDict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from marllib import marl
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY


RESULTS_DIR = os.path.abspath(
    os.environ.get("RWARE_MAPPO_RESULTS_DIR", "runs/rware_mappo_gru_small_20m_tensorboard")
)
RWARE_GYM_ID = os.environ.get("RWARE_GYM_ID", "rware-small-4ag-v2")
STOP_TIMESTEPS = int(os.environ.get("RWARE_STOP_TIMESTEPS", "20000000"))
NUM_GPUS = int(os.environ.get("RWARE_NUM_GPUS", "1"))
NUM_WORKERS = int(os.environ.get("RWARE_NUM_WORKERS", "6"))
CHECKPOINT_FREQ = int(os.environ.get("RWARE_CHECKPOINT_FREQ", "100"))
SEED = int(os.environ.get("RWARE_SEED", "321"))
RESTORE_MODEL_PATH = os.environ.get("RWARE_RESTORE_MODEL_PATH", "")
RESTORE_PARAMS_PATH = os.environ.get("RWARE_RESTORE_PARAMS_PATH", "")


POLICY_MAPPING = {
    "all_scenario": {
        "description": "RWARE v2 4-agent cooperative warehouse",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class RWAREV2ForMARLlib(MultiAgentEnv):
    """Gymnasium RWARE v2 wrapper compatible with MARLlib/RLlib 1.8."""

    def __init__(self, env_config):
        self.env_config = env_config
        self.env = gymnasium.make(env_config.get("gym_id", RWARE_GYM_ID))
        self.raw_env = self.env.unwrapped
        self.num_agents = self.raw_env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]

        obs_space = self.raw_env.observation_space[0]
        act_space = self.raw_env.action_space[0]
        self.action_space = Discrete(act_space.n)
        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=np.asarray(obs_space.low, dtype=obs_space.dtype),
                    high=np.asarray(obs_space.high, dtype=obs_space.dtype),
                    shape=obs_space.shape,
                    dtype=obs_space.dtype,
                )
            }
        )

    def reset(self):
        observations, _ = self.env.reset()
        return {
            "agent_{}".format(i): {"obs": observations[i]}
            for i in range(self.num_agents)
        }

    def step(self, action_dict):
        actions = tuple(action_dict["agent_{}".format(i)] for i in range(self.num_agents))
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        done = bool(terminated or truncated)
        team_reward = float(sum(rewards)) / self.num_agents

        obs = {}
        reward = {}
        infos = {}
        for i in range(self.num_agents):
            agent_id = "agent_{}".format(i)
            obs[agent_id] = {"obs": observations[i]}
            reward[agent_id] = team_reward
            infos[agent_id] = dict(info)
            infos[agent_id]["individual_reward"] = float(rewards[i])
            infos[agent_id]["team_reward"] = team_reward

        return obs, reward, {"__all__": done}, infos

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["max_steps"],
            "policy_mapping_info": POLICY_MAPPING,
        }

    def close(self):
        self.env.close()


def build_env():
    env_reg_name = "rware_v2_{}".format(RWARE_GYM_ID)
    tune.register_env(
        env_reg_name,
        lambda _: RWAREV2ForMARLlib({"map_name": RWARE_GYM_ID, "gym_id": RWARE_GYM_ID, "max_steps": 500}),
    )

    env_config = {
        "env": "rware_v2",
        "env_args": {
            "map_name": RWARE_GYM_ID,
            "gym_id": RWARE_GYM_ID,
            "max_steps": 500,
        },
        "force_coop": True,
        "mask_flag": False,
        "global_state_flag": False,
        "opp_action_in_cc": True,
        "agent_level_batch_update": True,
    }
    COOP_ENV_REGISTRY["rware_v2"] = RWAREV2ForMARLlib
    return RWAREV2ForMARLlib(env_config["env_args"]), marl.set_ray(env_config)


def verify_env():
    env = gymnasium.make(RWARE_GYM_ID)
    raw = env.unwrapped
    if raw.n_agents != 4:
        raise ValueError("{} has {} agents, expected 4".format(RWARE_GYM_ID, raw.n_agents))
    env.close()
    print("verified", RWARE_GYM_ID)


def latest_checkpoint():
    stage_dir = os.path.join(RESULTS_DIR, "mappo_gru_{}".format(RWARE_GYM_ID))
    checkpoints = sorted(Path(stage_dir).glob("*/checkpoint_*/checkpoint-*"))
    checkpoints = [p for p in checkpoints if not str(p).endswith(".tune_metadata")]
    if not checkpoints:
        return None
    model_path = str(checkpoints[-1])
    params_path = str(checkpoints[-1].parents[1] / "params.json")
    return {"model_path": model_path, "params_path": params_path}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    verify_env()
    env = build_env()

    # README pattern: choose algorithm, build GRU model, then call fit.
    mappo = marl.algos.mappo(
        hyperparam_source="common",
        batch_episode=10,
        lr=0.0005,
        entropy_coeff=0.01,
        num_sgd_iter=5,
    )
    model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-256"})

    mappo.fit(
        env,
        model,
        stop={"timesteps_total": STOP_TIMESTEPS},
        local_mode=False,
        num_gpus=NUM_GPUS,
        num_workers=NUM_WORKERS,
        share_policy="all",
        checkpoint_freq=CHECKPOINT_FREQ,
        checkpoint_end=True,
        local_dir=RESULTS_DIR,
        seed=SEED,
        restore_path={"model_path": RESTORE_MODEL_PATH, "params_path": RESTORE_PARAMS_PATH},
    )

    print("finished", RWARE_GYM_ID, "checkpoint", latest_checkpoint())


if __name__ == "__main__":
    main()
