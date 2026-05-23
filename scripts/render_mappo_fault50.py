import json
import os
from pathlib import Path

import cv2
import numpy as np
import ray
from ray.rllib.models import ModelCatalog

from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.models.zoo.rnn.cc_rnn import CentralizedCriticRNN
from scripts.evaluate_mappo_fault50 import (
    BASE_CHECKPOINT,
    BASE_RUN,
    FAULT_PROB,
    ROBUST_CHECKPOINT,
    ROBUST_RUN,
    load_config,
    make_eval_env,
    register_envs,
)


OUTPUT_DIR = Path(
    os.environ.get(
        "RWARE_RENDER_DIR",
        "/workspace/MARLlib/runs/rware_mappo_fault50_render_videos",
    )
)
EPISODES_PER_MODEL = int(os.environ.get("RWARE_RENDER_EPISODES", "5"))
FPS = int(os.environ.get("RWARE_RENDER_FPS", "12"))
FRAME_SKIP = int(os.environ.get("RWARE_RENDER_FRAME_SKIP", "2"))
SEED = int(os.environ.get("RWARE_RENDER_SEED", "20260524"))


def frame_with_overlay(frame, lines):
    frame = np.ascontiguousarray(frame.copy())
    cv2.rectangle(frame, (8, 8), (610, 82), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 32 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return frame


def save_video(path, frames):
    if not frames:
        raise ValueError("no frames to save")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("could not open video writer for {}".format(path))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def load_trainer(run_dir, checkpoint_path):
    trainer = MAPPOTrainer(config=load_config(run_dir))
    trainer.restore(str(checkpoint_path))
    return trainer


def render_model(label, run_dir, checkpoint_path):
    trainer = load_trainer(run_dir, checkpoint_path)
    policy = trainer.get_policy("default_policy")
    del policy  # Keep explicit restore check while using trainer.compute_single_action below.

    model_dir = OUTPUT_DIR / label
    model_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for episode_idx in range(EPISODES_PER_MODEL):
        env = make_eval_env(SEED + episode_idx)
        obs = env.reset()
        states = {agent: trainer.get_policy("default_policy").get_initial_state() for agent in env.agents}
        done = False
        episode_return = 0.0
        step = 0
        frames = []
        fault_active = env.broken_agent_index is not None
        broken_agent = "agent_{}".format(env.broken_agent_index) if fault_active else "none"

        while not done:
            if step % FRAME_SKIP == 0:
                frame = env.env.unwrapped.render(mode="rgb_array")
                frames.append(
                    frame_with_overlay(
                        frame,
                        [
                            "{}  ep {}  step {}".format(label, episode_idx + 1, step),
                            "fault={} broken={}".format(fault_active, broken_agent),
                            "return={:.1f}".format(episode_return),
                        ],
                    )
                )

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
            step += 1

        final_frame = env.env.unwrapped.render(mode="rgb_array")
        frames.append(
            frame_with_overlay(
                final_frame,
                [
                    "{}  ep {}  done".format(label, episode_idx + 1),
                    "fault={} broken={}".format(fault_active, broken_agent),
                    "return={:.1f}".format(episode_return),
                ],
            )
        )

        status = "fault_{}".format(broken_agent) if fault_active else "normal"
        video_path = model_dir / "{}_ep{:02d}_{}_return{:.0f}.mp4".format(
            label,
            episode_idx + 1,
            status,
            episode_return,
        )
        save_video(video_path, frames)
        env.close()

        summary = {
            "model": label,
            "episode": episode_idx + 1,
            "fault_active": fault_active,
            "broken_agent": broken_agent,
            "return": episode_return,
            "steps": step,
            "frames": len(frames),
            "video": str(video_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    trainer.stop()
    return summaries


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ModelCatalog.register_custom_model("Centralized_Critic_Model", CentralizedCriticRNN)
    register_envs()
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, num_gpus=0, log_to_driver=False)

    all_summaries = []
    all_summaries.extend(render_model("base_mappo_40m", BASE_RUN, BASE_CHECKPOINT))
    all_summaries.extend(render_model("robust_mappo_fault50", ROBUST_RUN, ROBUST_CHECKPOINT))

    summary_path = OUTPUT_DIR / "render_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print("saved_summary", summary_path)
    ray.shutdown()


if __name__ == "__main__":
    main()
