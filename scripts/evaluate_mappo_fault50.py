import json
import os
from pathlib import Path

import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog

from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.models.zoo.rnn.cc_rnn import CentralizedCriticRNN
from scripts.train_rware_mappo import RWAREV2ForMARLlib
from scripts.train_rware_mappo_fault import RWAREV2FaultForMARLlib


BASE_RUN = Path(
    "/workspace/MARLlib/runs/rware_mappo_gru_small_20m_tensorboard/"
    "mappo_gru_rware-small-4ag-v2/"
    "MAPPOTrainer_rware_v2_rware-small-4ag-v2_fd22d_00000_0_2026-05-22_22-02-56"
)
ROBUST_RUN = Path(
    "/workspace/MARLlib/runs/rware_mappo_gru_small_20m_tensorboard/"
    "mappo_gru_rware-small-4ag-v2-fault50/"
    "MAPPOTrainer_rware_v2_fault_rware-small-4ag-v2-fault50_12fdc_00000_0_2026-05-23_00-33-52"
)
BASE_CHECKPOINT = BASE_RUN / "checkpoint_002001" / "checkpoint-2001"
ROBUST_CHECKPOINT = ROBUST_RUN / "checkpoint_003300" / "checkpoint-3300"

EVAL_EPISODES = int(os.environ.get("RWARE_EVAL_EPISODES", "200"))
FAULT_PROB = float(os.environ.get("RWARE_EVAL_FAULT_PROB", "0.5"))
SEED = int(os.environ.get("RWARE_EVAL_SEED", "20260523"))
OUTPUT_PATH = Path(
    os.environ.get(
        "RWARE_EVAL_OUTPUT",
        "/workspace/MARLlib/runs/rware_mappo_fault50_evaluation.json",
    )
)


def register_envs():
    tune.register_env(
        "rware_v2_rware-small-4ag-v2",
        lambda _: RWAREV2ForMARLlib(
            {"map_name": "rware-small-4ag-v2", "gym_id": "rware-small-4ag-v2", "max_steps": 500}
        ),
    )
    tune.register_env(
        "rware_v2_fault_rware-small-4ag-v2-fault50",
        lambda _: RWAREV2FaultForMARLlib(
            {
                "map_name": "rware-small-4ag-v2-fault50",
                "gym_id": "rware-small-4ag-v2",
                "max_steps": 500,
                "fault_prob": FAULT_PROB,
            }
        ),
    )


def load_config(run_dir):
    with open(run_dir / "params.json", "r") as f:
        config = json.load(f)

    eval_env = RWAREV2FaultForMARLlib(
        {
            "map_name": "rware-small-4ag-v2-fault50",
            "gym_id": "rware-small-4ag-v2",
            "max_steps": 500,
            "fault_prob": FAULT_PROB,
        }
    )

    eval_info = eval_env.get_env_info()
    eval_info["agent_name_ls"] = eval_env.agents

    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["create_env_on_driver"] = True
    config["explore"] = False
    config["evaluation_interval"] = None
    config["log_level"] = "ERROR"
    config["multiagent"] = {
        "policies": {
            "default_policy": (
                None,
                eval_env.observation_space,
                eval_env.action_space,
                {},
            )
        },
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "default_policy",
    }
    custom_config = config["model"]["custom_model_config"]
    custom_config.update(eval_info)
    custom_config["env"] = "rware_v2_fault"
    custom_config["env_args"] = {
        "map_name": "rware-small-4ag-v2-fault50",
        "gym_id": "rware-small-4ag-v2",
        "max_steps": 500,
        "fault_prob": FAULT_PROB,
    }
    custom_config["force_coop"] = True
    custom_config["mask_flag"] = False
    custom_config["global_state_flag"] = False
    custom_config["opp_action_in_cc"] = True
    custom_config["agent_level_batch_update"] = True
    eval_env.close()
    return config


def make_eval_env(seed):
    env = RWAREV2FaultForMARLlib(
        {
            "map_name": "rware-small-4ag-v2-fault50",
            "gym_id": "rware-small-4ag-v2",
            "max_steps": 500,
            "fault_prob": FAULT_PROB,
        }
    )
    env.rng = np.random.default_rng(seed)
    return env


def evaluate_checkpoint(label, run_dir, checkpoint_path, episodes):
    trainer = MAPPOTrainer(config=load_config(run_dir))
    trainer.restore(str(checkpoint_path))
    policy = trainer.get_policy("default_policy")

    env = make_eval_env(SEED)
    returns = []
    normal_returns = []
    fault_returns = []
    fault_agent_counts = {agent: 0 for agent in env.agents}

    for episode_idx in range(episodes):
        obs = env.reset()
        states = {agent: policy.get_initial_state() for agent in env.agents}
        done = False
        episode_return = 0.0
        fault_active = env.broken_agent_index is not None
        if fault_active:
            fault_agent_counts["agent_{}".format(env.broken_agent_index)] += 1

        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                action, state_out, _ = trainer.compute_single_action(
                    agent_obs,
                    state=states[agent_id],
                    policy_id="default_policy",
                    explore=False,
                )
                states[agent_id] = state_out
                actions[agent_id] = int(action)

            obs, rewards, dones, _ = env.step(actions)
            episode_return += float(sum(rewards.values()))
            done = bool(dones["__all__"])

        returns.append(episode_return)
        if fault_active:
            fault_returns.append(episode_return)
        else:
            normal_returns.append(episode_return)

        if (episode_idx + 1) % 25 == 0:
            print(label, "episodes", episode_idx + 1, "mean_return", float(np.mean(returns)))

    env.close()
    trainer.stop()

    def summary(values):
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "episodes": episodes,
        "fault_prob": FAULT_PROB,
        "all": summary(returns),
        "normal_episodes": summary(normal_returns),
        "fault_episodes": summary(fault_returns),
        "fault_agent_counts": fault_agent_counts,
        "returns": returns,
    }


def main():
    ModelCatalog.register_custom_model("Centralized_Critic_Model", CentralizedCriticRNN)
    register_envs()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, num_gpus=0, log_to_driver=False)

    results = {
        "eval_episodes": EVAL_EPISODES,
        "seed": SEED,
        "fault_prob": FAULT_PROB,
        "models": [
            evaluate_checkpoint("base_mappo_40m", BASE_RUN, BASE_CHECKPOINT, EVAL_EPISODES),
            evaluate_checkpoint("robust_mappo_fault50", ROBUST_RUN, ROBUST_CHECKPOINT, EVAL_EPISODES),
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({m["label"]: m["all"] for m in results["models"]}, indent=2))
    print("saved", OUTPUT_PATH)
    ray.shutdown()


if __name__ == "__main__":
    main()
