import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from poppy_env import PoppyTorsoEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.epoch_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Assume self.training_env is a VecEnv
            episode_rewards = self.training_env.get_attr('episode_rewards')
            mean_reward = np.mean([np.sum(rewards) for rewards in episode_rewards if rewards])
            self.epoch_rewards.append(mean_reward)

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("best_model")
            
            if self.verbose > 0:
                print(f"Step number: {self.n_calls} - Reward: {mean_reward}")

        return True


def plot_rewards(rewards, title='Training Progress'):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Sum of Rewards')
    plt.title(title)
    plt.grid(True)
    plt.show()


def main():
    # Create vector of environments to run in parallel
    env = make_vec_env(PoppyTorsoEnv, n_envs=4)

    # Create a callback to log the rewards
    reward_logger = RewardLoggerCallback(check_freq=1000)  # Adjust frequency based on your needs

    # Create PPO model that will learn the optimal policy to control the Poppy Torso
    model = PPO('CnnLstmPolicy', env, verbose=1) # using CNN (for spatial awareness) + LSTM (for temporal dynamics)

    model.learn(total_timesteps=20000, callback=reward_logger)
    model.save("poppy_torso_ppo")
    env.close()
    plot_rewards(reward_logger.epoch_rewards)


if __name__ == "__main__":
    main()