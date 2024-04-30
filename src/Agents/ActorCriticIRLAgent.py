import gym
import torch
import torch.nn as nn
import torch.optim as optim
from src.Models.PolicyLSTM import PolicyLSTM
from src.Models.CriticLSTM import CriticLSTM

class ActorCriticIRLAgent:
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
        self.actor = PolicyLSTM(input_dim, hidden_dim, output_dim, num_layers)
        self.critic = CriticLSTM(input_dim, hidden_dim, num_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

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

                next_state, _, done, info = self.env.step(action)
                ground_truth_coords = info.get('ground_truth')  # Assuming ground truth is provided in info dict

                predicted_coords = torch.tensor(next_state, dtype=torch.float32)
                ground_truth_coords = torch.tensor(ground_truth_coords, dtype=torch.float32)
                reward = self.reward_function(predicted_coords, ground_truth_coords)

                # Evaluate and update Critic (Value function)
                state_value = self.critic(state_tensor)
                next_state_value = self.critic(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0))

                # Calculate temporal difference error
                td_error = reward + (0.99 * next_state_value.detach()) - state_value

                # Update Critic
                self.critic_optimizer.zero_grad()
                critic_loss = td_error.pow(2).mean()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Update Actor using the critic's evaluation
                self.actor_optimizer.zero_grad()
                actor_loss = -self.critic(state_tensor).mean()
                actor_loss.backward()
                self.actor_optimizer.step()

                state = next_state
                total_reward += reward.item()

            print(f"Episode {episode+1}, Total Reward: {total_reward}")

    def save_model(self, actor_path='actor_net.pth', critic_path='critic_net.pth'):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

