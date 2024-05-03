from pypot.creatures import PoppyTorso
from pypot import vrep
from pypot.primitive.move import MovePlayer, Move


class PoppyChannel:
    """Manager and controller of Poppy Torso robot within a V-REP simulation.

    Attributes:
        __poppy (PoppyTorso): Instance of the Poppy Torso robot.
        __moves (Move): Movement controller for the robot.
        __fps (int): Frame rate for the movement commands.
        __default_pose (dict): Default positions for the robot's joints.
        left_motors (list): List of left arm motor identifiers.
        right_motors (list): List of right arm motor identifiers.
    """
    def __init__(self):
        vrep.close_all_connections()
        self.__poppy = None
        self.__moves = None
        self.__fps = 10
        self.__default_pose = {
            "head_y": [0.0, 0.0],
            "head_z": [0.0, 0.0],
            "abs_z": [0.0, 0.0],
            "bust_x": [0.0, 0.0],
            "bust_y": [0.0, 0.0],
            "r_elbow_y": [90.0, 0.0],
            "r_arm_z": [0.0, 0.0],
            "r_shoulder_x": [0.0, 0.0],
            "r_shoulder_y": [0.0, 0.0],
            "l_elbow_y": [90.0, 0.0],
            "l_arm_z": [0.0, 0.0],
            "l_shoulder_x": [0.0, 0.0],
            "l_shoulder_y": [0.0, 0.0],
        }
        self.left_motors = ["l_shoulder_x"]
        self.right_motors = ["r_shoulder_x"]
        self.poppy_reset()

    def connect(self):
        """Establishes a connection with the V-REP simulator and initializes the Poppy Torso.

        Args:
            None

        Returns:
            None
        """
        self.disconnect()
        self.__poppy = PoppyTorso(
            simulator="vrep", base_path="./src/scene", scene="poppy_torso.ttt"
        )
        self.__moves = Move(freq=self.__fps)

    def disconnect(self):
        """Closes all existing connections to the V-REP simulator.

        Args:
            None

        Returns:
            None
        """
        vrep.close_all_connections()

    def set_poppy_position(self, positions, timestamps):
        """Adds a position to the move queue of the robot.

        Args:
            positions (dict): A dictionary of joint names and target positions.
            timestamps (int or list): Time(s) at which the positions should be set.

        Returns:
            None
        """
        self.__moves.add_position(positions, timestamps)

    def get_poppy_positions(self, arm):
        """Retrieves the current joint positions and end-effector position for a specified arm.

        Args:
            arm (str): The arm for which to retrieve positions ('left' or 'right').

        Returns:
            tuple: A tuple containing joint positions and the end-effector position.

        Raises:
            Exception: If an unknown arm is specified.
        """
        # Retrieve poppy joint & effector positions
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
        """Starts the movement sequence using the queued positions.

        Args:
            None

        Returns:
            None
        """
        move_player = MovePlayer(self.__poppy, self.__moves, player_id=1)
        move_player.start()

    def poppy_reset(self):
        """Resets the Poppy robot to the default pose.

        Args:
            None

        Returns:
            None
        """
        if self.__poppy is not None:
            self.set_poppy_position(self.__default_pose, 0)
            self.poppy_move()
