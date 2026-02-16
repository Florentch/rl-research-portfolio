"""Evaluation script"""

import os
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
from environment import PredatorPreyEnv
from utils import load_model, get_action
from config import *


def run_episode(env, pred_model, prey_model, output_dir, episode_num):
    observations, _ = env.reset()
    frames = []
    rewards = [0, 0]
    done = False
    step = 0
    
    while not done:
        actions = [get_action(pred_model, observations[0]), get_action(prey_model, observations[1])]
        observations, r, terminated, truncated, info = env.step(actions)
        rewards[0] += r[0]
        rewards[1] += r[1]
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        step += 1
        done = terminated or truncated
    
    if frames:
        try:
            gif_path = f"{output_dir}/ep{episode_num:02d}.gif"
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
        except:
            pass
    
    return step, rewards, info['caught']


def evaluate(predator_path, prey_path, num_episodes, test_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"evaluation/{test_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nEvaluation: {test_name}")
    print(f"Predator: {predator_path or 'random'}")
    print(f"Prey: {prey_path or 'random'}\n")
    
    pred_model = load_model(predator_path)
    prey_model = load_model(prey_path)
    env = PredatorPreyEnv()
    
    all_lengths = []
    all_pred_rewards = []
    all_prey_rewards = []
    catches = 0
    
    for ep in range(num_episodes):
        length, rewards, caught = run_episode(env, pred_model, prey_model, output_dir, ep+1)
        all_lengths.append(length)
        all_pred_rewards.append(rewards[0])
        all_prey_rewards.append(rewards[1])
        if caught:
            catches += 1
        print(f"Ep {ep+1:2d}: Len={length:3d} | Pred={rewards[0]:6.1f} | Prey={rewards[1]:6.1f} | Caught={caught}")
    
    env.close()
    
    print(f"\nResults:")
    print(f"Avg Length: {np.mean(all_lengths):.1f}")
    print(f"Catch Rate: {catches}/{num_episodes} ({catches/num_episodes*100:.0f}%)\n")
    
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"{test_name}\n")
        f.write(f"Avg Length: {np.mean(all_lengths):.1f}\n")
        f.write(f"Catch Rate: {catches}/{num_episodes}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents")
    parser.add_argument("--predator", type=str, default=None)
    parser.add_argument("--prey", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--name", type=str, default="test")
    args = parser.parse_args()
    evaluate(args.predator, args.prey, args.episodes, args.name)
