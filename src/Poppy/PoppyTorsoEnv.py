import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from src.CoppeliaComs.PoppyChannel import PoppyChannel
from src.utils.movement_parser import *


class PoppyTorsoEnv(gym.Env):

    def __init__(self, targets):
        super().__init__()
        self.poppy_channel = PoppyChannel()
        self.poppy_channel.connect()

        # Targets
        self.__current_step = 0
        self.__target_positions = targets.numpy() if isinstance(targets, torch.Tensor) else targets

        # Motors
        self.left_motor_names = self.poppy_channel.left_motors
        self.right_motor_names = self.poppy_channel.right_motors

        # Assuming joint movement ranges
        joint_limits = np.deg2rad(180)
        self.action_space = spaces.Box(low=-joint_limits, high=joint_limits, shape=(2,), dtype=np.float32)

        # Assuming the robot provides joint positions and Cartesian positions for the end effectors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def step(self, action):
        # action is structured as [left_arm_actions, right_arm_actions]
        left_actions = action[:len(self.left_motor_names)]
        right_actions = action[len(self.left_motor_names):]

        # Create dict with motor name and for each position & velocity
        action_dict = {**dict(zip(self.left_motor_names, left_actions)),
                       **dict(zip(self.right_motor_names, right_actions))}

        # Set Poppy position
        self.poppy_channel.set_poppy_position(action_dict, 0)
        self.poppy_channel.poppy_move()

        # Retrieve the new state from the robot
        state = self.get_state()
        reward = self.calculate_reward(action, state)
        done = self.check_if_done(state)

        self.__current_step += 1
        return state, reward, done, {}

    def get_state(self):
        # Retrieve both joint angles and Cartesian coordinates
        _, l_end_effector_positions = self.poppy_channel.get_poppy_positions("left")
        _, r_end_effector_positions = self.poppy_channel.get_poppy_positions("right")

        # Debug
        print("Left positions shape:", l_end_effector_positions.shape)
        print("Right positions shape:", r_end_effector_positions.shape)

        return np.concatenate([l_end_effector_positions, r_end_effector_positions])

    def calculate_reward(self, state):
        left_effector_pos = state[:3]
        right_effector_pos = state[3:]

        # Sanity check
        left_effector_pos = np.array(left_effector_pos).flatten()
        right_effector_pos = np.array(right_effector_pos).flatten()

        # Simple reward
        distance_left = np.linalg.norm(left_effector_pos - self.__target_positions[self.__current_step, 0])
        distance_right = np.linalg.norm(right_effector_pos - self.__target_positions[self.__current_step, 1])

        return -(distance_left + distance_right)

    def check_if_done(self, state):
        return self.__current_step == self.__target_positions.shape[0]

    def reset(self, **kwargs):
        self.__current_step = 0
        self.poppy_channel.poppy_reset()

    def close(self):
        self.poppy_channel.disconnect()

    def test(self):
        import time
        self.poppy_channel.poppy_reset()
        timestamps = np.linspace(0.02, 3, 10)
        a = {'l_shoulder_x': [20, 0.0], 'r_shoulder_x': [-20, 0.0]}
        b = {'l_shoulder_x': [40, 0.0], 'r_shoulder_x': [-40, 0.0]}
        c = {'l_shoulder_x': [60, 0.0], 'r_shoulder_x': [-60, 0.0]}
        d = {'l_shoulder_x': [90, 0.0], 'r_shoulder_x': [-90, 0.0]}
        e = {'l_shoulder_x': [110, 0.0], 'r_shoulder_x': [-110, 0.0]}
        f = {'l_shoulder_x': [130, 0.0], 'r_shoulder_x': [-130, 0.0]}
        self.poppy_channel.set_poppy_position(a, 0.02)
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(b, timestamps[0])
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(c, timestamps[0])
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(d, timestamps[0])
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(d, timestamps[0])
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(e, timestamps[0])
        self.poppy_channel.poppy_move()
        time.sleep(2)
        self.poppy_channel.set_poppy_position(f, timestamps[0])
        self.poppy_channel.poppy_move()

        print(f"LEFT: {self.poppy_channel.get_poppy_positions('left')}")
        print(f"RIGHT: {self.poppy_channel.get_poppy_positions('left')}")
        self.poppy_channel.set_poppy_position({'l_shoulder_x': [0, 0.0], 'r_shoulder_x': [0, 0.0]}, timestamps[0])
        self.poppy_channel.poppy_move()


# skeletons = torch.from_numpy(np.load("skeletons_sao.npy"))
# topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
#
# targets, _ = targets_from_skeleton(skeletons, np.array(topology), 3)
# To create an instance of the environment:
# env = PoppyTorsoEnv([0])
# env.test()
