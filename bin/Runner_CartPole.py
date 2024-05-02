import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
#from src.Poppy.PoppyTorsoEnv import PoppyTorsoEnv
from src.utils.movement_parser import *


def main():

    # Create paths to save tests
    testname = 'Test1'
    testdir = os.path.join(os.getcwd(), 'Tests', datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + testname) # Path for saving training data
    logger_path = os.path.join(testdir, 'Logger') # Path for logger output
    trained_models_path = os.path.join(testdir, 'TrainedModels') # Path for trained models
    os.makedirs(testdir) 
    os.makedirs(logger_path) 
    os.makedirs(trained_models_path)

    # Create vector of environments to run in parallel
    skeletons = torch.from_numpy(np.load("skeletons_sao.npy"))
    topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    targets, _ = targets_from_skeleton(skeletons, np.array(topology), 3)

    # Initialize the environment
    # env = PoppyTorsoEnv(targets.numpy())
    # env = DummyVecEnv([lambda: env])
    env = gym.make('CartPole-v1', render_mode="rgb_array")

    # Initialize the PPO model
    #policy_kwargs = dict(activation_fn= th.nn.Tanh, net_arch=[dict(pi=[4,4], vf=[4,4])])
    #model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Set up logger
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])
    #new_logger = configure(logger_path, ["stdout", "csv"])
    model.set_logger(new_logger)

    # Train and save the model
    model.learn(total_timesteps=100000)
    model.save(os.path.join(trained_models_path, "poppy_torso_ppo"))

    # Reload it
    #model = PPO.load(os.path.join(trained_models_path, "poppy_torso_ppo"), env=env)

    # Evaluate the trained model
    #mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    #print(f"Mean reward: {mean_reward} +/- {std_reward}")

    num_steps=1000
    obs, _ = env.reset()
    for i in range(num_steps):
        action, _ = model.predict(obs)
        obs, rewards, dones, _, info = env.step(action)
        if dones:
            obs, _ = env.reset()
    env.render()
    env.close()


if __name__ == "__main__":
    main()
