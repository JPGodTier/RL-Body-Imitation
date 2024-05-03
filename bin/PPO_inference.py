import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.Poppy.PoppyTorsoEnv import PoppyTorsoEnv
from src.utils.movement_parser import targets_from_skeleton


def main():
    trained_model_path = "./Results/TrainedModels/"

    skeletons = torch.from_numpy(np.load("target_skeleton.npy"))
    topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    targets, _ = targets_from_skeleton(skeletons, np.array(topology), 3)

    # Initialize the environment
    env = PoppyTorsoEnv(targets.numpy())
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(os.path.join(trained_model_path, "poppy_torso_ppo"), env=env)

    num_steps = 279
    obs = env.reset()
    left_arm_x, left_arm_y = [], []
    right_arm_x, right_arm_y = [], []

    for i in range(num_steps):
        # Recall obs here are of the form [[x,y,z,x,y,z]] for left & right hand
        left_arm_x.append(obs[0][0])
        left_arm_y.append(obs[0][1])

        right_arm_x.append(obs[0][3])
        right_arm_y.append(obs[0][4])

        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            obs = env.reset()

    targets = targets.numpy()
    frame_steps = [i for i in range(num_steps)]
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(frame_steps, left_arm_x, label="predicted_left_x")
    plt.plot(
        frame_steps,
        targets[:, 0][:, 0],
        label="ground_truth_left_x",
    )
    plt.plot(frame_steps, left_arm_y, label="predicted_left_y")
    plt.plot(
        frame_steps,
        targets[:, 0][:, 1],
        label="ground_truth_left_y",
    )
    plt.title("Left hand evolution")
    plt.legend()

    plt.subplot(122)
    plt.plot(frame_steps, right_arm_x, label="predicted_right_x")
    plt.plot(
        frame_steps,
        targets[:, 1][:, 0],
        label="ground_truth_right_x",
    )
    plt.plot(frame_steps, right_arm_y, label="predicted_right_y")
    plt.plot(
        frame_steps,
        targets[:, 1][:, 1],
        label="ground_truth_right_y",
    )
    plt.title("Right hand evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Results/inference_vs_groundtruth.png")

    env.close()


if __name__ == "__main__":
    main()
