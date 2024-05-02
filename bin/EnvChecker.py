from stable_baselines3.common.env_checker import check_env
from src.Poppy.PoppyTorsoEnv import PoppyTorsoEnv
from src.utils.movement_parser import *

# This amazing scripts checks that the env is well done like a steak
skeletons = torch.from_numpy(np.load("skeletons_sao.npy"))
topology = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

targets, _ = targets_from_skeleton(skeletons, np.array(topology), 3)
env = PoppyTorsoEnv(targets.numpy())
check_env(env)