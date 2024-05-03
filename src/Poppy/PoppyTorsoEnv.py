import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from src.CoppeliaComs.PoppyChannel import PoppyChannel


class PoppyTorsoEnv(gym.Env):
    """A custom environment for controlling a Poppy Torso robot in a simulated environment using the Gym library.

    Attributes:
        __current_step (int): Internal counter for the current step in the environment.
        __target_positions (np.ndarray): Target positions for the robot's end effectors.
        poppy_channel (PoppyChannel): The channel through which the Poppy Torso robot is controlled.
        left_motor_names (list): List of names of the left motors.
        right_motor_names (list): List of names of the right motors.
        action_space (spaces.Box): The Space object corresponding to valid actions.
        observation_space (spaces.Box): The Space object corresponding to valid observations.
    """

    def __init__(self, targets):
        super().__init__()
        self.poppy_channel = PoppyChannel()
        self.poppy_channel.connect()

        # Targets
        self.__current_step = 0
        self.__target_positions = (
            targets.numpy() if isinstance(targets, torch.Tensor) else targets
        )

        # Motors
        self.left_motor_names = self.poppy_channel.left_motors
        self.right_motor_names = self.poppy_channel.right_motors

        # Define the action space for motor commands
        self.action_space = spaces.Box(
            low=np.array([0, -1], dtype=np.float32),
            high=np.array([1, 0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Define the observation space that the agent can expect to receive
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )

    def step(self, action):
        """Executes one time step within the environment.

        Args:
            action (array-like): The action to be taken, array of motor positions.

        Returns:
            tuple: A tuple containing the new state, reward, done flag, truncated flag, and an info dict.
        """
        # Translate actions into actual motor commands, considering the range of the motors
        scaled_action_left = np.interp(action[0], [-1, 1], [0, 180])
        scaled_action_right = np.interp(action[1], [-1, 1], [-180, 0])

        action_dict = {
            self.left_motor_names[0]: [scaled_action_left, 0.0],
            self.right_motor_names[0]: [scaled_action_right, 0.0],
        }

        # Apply and Initiate poppy mvt
        self.poppy_channel.set_poppy_position(action_dict, 0)
        self.poppy_channel.poppy_move()

        # Retrieve new state from the robot
        state = self.get_state()
        reward = self.calculate_reward(state)

        # Increment step and check for end of sim
        self.__current_step += 1
        done = self.check_if_done()
        truncated = False

        return state, float(reward), done, truncated, {}

    def get_state(self):
        """Retrieves the current state of the robot's end effectors.

        Args:
            None

        Returns:
            np.ndarray: The current state array combining both joint angles and Cartesian coordinates.
        """
        # Retrieve both joint angles and Cartesian coordinates
        _, l_end_effector_positions = self.poppy_channel.get_poppy_positions("left")
        _, r_end_effector_positions = self.poppy_channel.get_poppy_positions("right")

        return np.concatenate(
            [l_end_effector_positions, r_end_effector_positions]
        ).astype(np.float32)

    def calculate_reward(self, state):
        """Calculates the reward based on the current state.

        Args:
            state (np.ndarray): The current state from which to calculate the reward.

        Returns:
            float: The calculated reward value.
        """
        left_effector_pos = state[:3]
        right_effector_pos = state[3:]

        # Sanity check
        left_effector_pos = np.array(left_effector_pos).flatten()
        right_effector_pos = np.array(right_effector_pos).flatten()

        # Calculate distance from current position to target as a reward
        distance_left = np.linalg.norm(
            left_effector_pos - self.__target_positions[self.__current_step, 0]
        )
        distance_right = np.linalg.norm(
            right_effector_pos - self.__target_positions[self.__current_step, 1]
        )
        return -(distance_left + distance_right)

    def check_if_done(self):
        """Checks if the episode has completed.

        Returns:
            bool: True if the episode is complete, otherwise False.
        """
        return self.__current_step == self.__target_positions.shape[0]

    def reset(self, **kwargs):
        """Resets the environment to the initial state.

        Args:
            kwargs (dict): Additional arguments to be considered for resetting, not used currently.

        Returns:
            tuple: The initial state and an info dict.
        """
        # Reset the ste var
        self.__current_step = 0

        # Reset the robot to its default state
        self.poppy_channel.poppy_reset()
        initial_state = self.get_state()
        return initial_state, {}

    def close(self):
        """Closes the environment, disconnecting all external connections.

        Args:
            None

        Returns:
            None
        """
        self.poppy_channel.disconnect()
