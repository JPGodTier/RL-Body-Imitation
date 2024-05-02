from pypot.creatures import PoppyTorso
from pypot import vrep
from pypot.primitive.move import MovePlayer, Move
import numpy as np


class PoppyChannel:
    def __init__(self):
        vrep.close_all_connections()
        self.__poppy = None
        self.__moves = None
        self.__fps = 10
        self.__default_pose = {
            "head_y": 0.0,
            "head_z": 0.0,
            "abs_z": 0.0,
            "bust_x": 0.0,
            "bust_y": 0.0,
            "r_elbow_y": 90.0,
            "r_arm_z": 0.0,
            "r_shoulder_x": 0.0,
            "r_shoulder_y": 0.0,
            "l_elbow_y": 90.0,
            "l_arm_z": 0.0,
            "l_shoulder_x": 0.0,
            "l_shoulder_y": 0.0,
        }
        self.left_motors = ["l_shoulder_x"]
        self.right_motors = ["r_shoulder_x"]

    def connect(self):
        self.disconnect()
        self.__poppy = PoppyTorso(
            simulator="vrep", base_path="./src/scene", scene="poppy_torso.ttt"
        )
        self.__moves = Move(freq=self.__fps)

    def disconnect(self):
        vrep.close_all_connections()

    def set_poppy_position(self, positions, timestamps):
        self.__moves.add_position(positions, timestamps)

    def get_poppy_positions(self, arm):
        if arm == "left":
            joint_positions = self.__poppy.l_arm_chain.joints_position
            end_effector_position = self.__poppy.l_arm_chain.position
        elif arm == "right":
            joint_positions = self.__poppy.r_arm_chain.joints_position
            end_effector_position = self.__poppy.r_arm_chain.position
        else:
            raise Exception("Unknown arm")

        return joint_positions, end_effector_position

    def poppy_move(self):
        move_player = MovePlayer(self.__poppy, self.__moves, player_id=1)
        move_player.start()

    def poppy_reset(self):
        if self.__poppy is not None:
            for m in self.__poppy.motors:
                m.goto_position(self.__default_pose[m.name], 5)
