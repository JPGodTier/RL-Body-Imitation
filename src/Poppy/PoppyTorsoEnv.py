import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.CoppeliaComs.PoppyChannel import PoppyChannel


class PoppyTorsoEnv(gym.Env):

    def __init__(self, targets):
        self.poppy_channel = PoppyChannel()
        self.poppy_channel.connect()

        # Targets
        self.__current_step = 0
        self.__target_positions = targets

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

        return (np.array(l_end_effector_positions),
                np.array(r_end_effector_positions))

    def calculate_reward(self, state):
        # simple test reward
        # TODO: add target as list of points that must be reached
        left_effector_pos, right_effector_pos = state
        distance_left = np.linalg.norm(left_effector_pos - self.__target_positions[self.__current_step, 0])
        distance_right = np.linalg.norm(right_effector_pos - self.__target_positions[self.__current_step, 1])

        return -np.sum(distance_left + distance_right)

    def check_if_done(self, state):
        return self.__current_step == self.__target_positions.shape[0]

    def poppy_default_pos(self):
        self.poppy_channel.poppy_reset()
        return self.get_state()

    def close(self):
        self.poppy_channel.disconnect()

    def test(self):
        import time
        self.poppy_default_pos()
        timestamps = np.linspace(0.02, 3, 10)
        a = {'l_shoulder_x': [20, 0.0], 'r_shoulder_x': [-20, 0.0]}
        b = {'l_shoulder_x': [40, 0.0], 'r_shoulder_x': [-40, 0.0]}
        c = {'l_shoulder_x': [60, 0.0], 'r_shoulder_x': [-60, 0.0]}
        d = {'l_shoulder_x': [90, 0.0], 'r_shoulder_x': [-90, 0.0]}
        e = {'l_shoulder_x': [110, 0.0], 'r_shoulder_x': [-110, 0.0]}
        f = {'l_shoulder_x': [130, 0.0], 'r_shoulder_x': [-130, 0.0]}
        self.poppy_channel.set_poppy_position(a, timestamps[0])
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


# To create an instance of the environment:
env = PoppyTorsoEnv()
env.test()
