import gym
import torch
import torch.nn as nn
import torch.optim as optim
from src.Models.PolicyLSTM import PolicyLSTM

class BasicIRLAgent:
    def __init__(self, env_name='PoppyTorsoEnv', input_dim=51, hidden_dim=128, output_dim=13, num_layers=2, learning_rate=0.01):
        """
        Class of an Inverse Reinforcement Learning (IRL) agent that learns the reward function from expert demonstrations.

        Args:
            env_name (str, optional):  Environment name. Defaults to 'PoppyTorsoEnv'.
            input_dim (int, optional): Dimension of the skeleton as input. Defaults to 17 points * 3 coordinates = 51.
            hidden_dim (int, optional): Number of features in the hidden state. Defaults to 128. Can be changed.
            output_dim (int, optional): Predicting one angle per joint. Defaults to 13.
            num_layers (int, optional): Number of stacked LSTM layers. Defaults to 2. Can be changed.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.01.
        """
        self.env = gym.make(env_name)
        self.policy_net = PolicyLSTM(input_dim, hidden_dim, output_dim, num_layers)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def reward_function(self, predicted_coords, ground_truth_coords):
        # Calculate Euclidean distance between predicted and ground truth coordinates
        distance = torch.norm(predicted_coords - ground_truth_coords, dim=-1)
        return -distance.mean()  # Negative to form a reward, average over all vectors to get a single reward

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Policy Evaluation: Generate behavior and calculate the reward
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # Double unsqueeze to transform into torch(1, 1, n) shape
                action = self.policy_net(state_tensor).squeeze().detach().numpy() # Squeeze to transform into torch(n) shape, detach to remove from computation of gradient

                next_state, reward, done, info = self.env.step(action)
                ground_truth_coords = info.get('ground_truth')  # Assuming ground truth is provided in info dict

                predicted_coords = torch.tensor(next_state, dtype=torch.float32)
                ground_truth_coords = torch.tensor(ground_truth_coords, dtype=torch.float32)
                reward = self.reward_function(predicted_coords, ground_truth_coords)

                # Policy Improvement: Adjust the policy to increase the cumulative reward
                self.optimizer.zero_grad()
                loss = -reward  # Negative reward because we want to maximize reward
                loss.backward()
                self.optimizer.step()

                state = next_state
                total_reward += reward.item()

            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    def save_model(self, file_path='policy_net.pth'):
        torch.save(self.policy_net.state_dict(), file_path)
