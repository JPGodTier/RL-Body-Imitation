import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.CoppeliaComs.PoppyChannel import PoppyChannel


class PoppyTorsoEnv(gym.Env):

    def __init__(self):
        self.poppy_channel = PoppyChannel()
        self.poppy_channel.connect()

        # Motors
        self.left_motor_names = self.poppy_channel.left_motors
        self.right_motor_names = self.poppy_channel.right_motors

        # Assuming joint movement ranges
        joint_limits = np.deg2rad(180)
        self.action_space = spaces.Box(low=-joint_limits, high=joint_limits, shape=(13,), dtype=np.float32)

        # Assuming the robot provides joint positions and Cartesian positions for the end effectors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def step(self, action):
        # action is structured as [left_arm_actions, right_arm_actions]
        left_actions = action[:len(self.left_motor_names)]
        right_actions = action[len(self.left_motor_names):]

        # Create dict with motor name and for each position & velocity
        action_dict = {**dict(zip(self.left_motor_names, left_actions)),
                       **dict(zip(self.right_motor_names, right_actions))}

        # Set Poppy position
        self.poppy_channel.set_poppy_position(action_dict)

        # Retrieve the new state from the robot
        state = self.get_state()
        reward = self.calculate_reward(action, state)
        done = self.check_if_done(state)

        return state, reward, done, {}

    def get_state(self):
        # Retrieve both joint angles and Cartesian coordinates
        l_joint_positions, l_end_effector_positions = self.poppy_channel.get_poppy_positions("left")
        r_joint_positions, r_end_effector_positions = self.poppy_channel.get_poppy_positions("right")

        return (np.concatenate([l_joint_positions, l_end_effector_positions]),
                np.concatenate([r_joint_positions, r_end_effector_positions]))

    def calculate_reward(self, action, state):
        # simple test reward
        # TODO: add target as list of points that must be reached
        return -np.linalg.norm(state[-3:] - self.target_position)

    def check_if_done(self, state):
        return np.linalg.norm(state[-3:] - self.target_position) < 0.01

    def poppy_default_pos(self):
        self.poppy_channel.poppy_reset()
        return self.get_state()

    def close(self):
        self.poppy_channel.disconnect()

    def test(self):
        import time
        self.poppy_default_pos()
        a = {'r_shoulder_x': [10.176048262430385, 0.0],
             'r_shoulder_y': [10.176048262430385, 0.0]}
        b = {'r_shoulder_x': [0.0, 0.0],
             'r_shoulder_y': [0.0, 0.0]}
        c = {'l_shoulder_x': [40, 0.0],
             'l_shoulder_y': [40, 0.0]}
        d = {'l_shoulder_x': [0.0, 0.0],
             'l_shoulder_y': [0.0, 0.0]}
        while True:
            self.poppy_channel.set_poppy_position(a)
            self.poppy_channel.poppy_move()
            print(self.poppy_channel.get_poppy_positions())
            time.sleep(3)
            self.poppy_channel.set_poppy_position(c)
            self.poppy_channel.poppy_move()
            print(self.poppy_channel.get_poppy_positions())
            time.sleep(3)
            self.poppy_channel.set_poppy_position(b)
            self.poppy_channel.poppy_move()
            print(self.poppy_channel.get_poppy_positions())
            time.sleep(3)
            self.poppy_channel.set_poppy_position(d)
            self.poppy_channel.poppy_move()
            print(self.poppy_channel.get_poppy_positions())
            time.sleep(3)


# To create an instance of the environment:
env = PoppyTorsoEnv()
env.test()
