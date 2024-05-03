import numpy as np
import torch
from src.utils.skeleton import forward_kinematics, find_quaternions
from src.utils.quaternion import (
    batch_quat_left_multiply,
    batch_quat_inverse,
)

POPPY_LENGTHS = torch.Tensor(
    [
        0.0,
        0.07,
        0.18,
        0.19,
        0.07,
        0.18,
        0.19,
        0.12,
        0.08,
        0.07,
        0.05,
        0.1,
        0.15,
        0.13,
        0.1,
        0.15,
        0.13,
    ]
)


def normalize(skeletons, topology):
    """
    Normalises the skeleton

    Args:
        skeletons (np.ndarray, torch.Tensor): Shape [Nb of frames within video, Nb of joints, 3 (x,y,z)]
        topology (np.ndarray): Skeleton topology array
    """
    rota_skeletons_A = skeletons.clone()
    rota_skeletons_A[:, :, 2] = -skeletons[:, :, 1]
    rota_skeletons_A[:, :, 1] = skeletons[:, :, 2]
    center_A = rota_skeletons_A[:, 0, :].unsqueeze(1).repeat(1, len(topology), 1)
    rota_skeletons_A = rota_skeletons_A - center_A

    return rota_skeletons_A


def change_frame(skeletons, topology, alpha=np.pi / 4.0):
    """
    Normalizes and allows to change the reference frame

    Args:
        skeletons (np.ndarray, torch.Tensor): Shape [Nb of frames within video, Nb of joints, 3 (x,y,z)]
        topology (np.ndarray): Skeleton topology array
        alpha (float): Used to define the rotation, defaults to np.pi/4.0

    """
    rota_skeletons_A = normalize(skeletons, topology)

    batch_size, n_joints, _ = rota_skeletons_A.shape

    # Measure skeleton bone lengths
    lengths = torch.Tensor(batch_size, n_joints)
    for child, parent in enumerate(topology):
        lengths[:, child] = torch.sqrt(
            torch.sum(
                (rota_skeletons_A[:, child] - rota_skeletons_A[:, parent]) ** 2, axis=-1
            )
        )

    # Find the corresponding angles
    offsets = torch.zeros(batch_size, n_joints, 3)
    offsets[:, :, -1] = lengths
    quaternions = find_quaternions(topology, offsets, rota_skeletons_A)

    # Rotate of alpha
    # define the rotation by its quaternion
    rotation = (
        torch.Tensor([np.cos(alpha / 2), np.sin(alpha / 2), 0, 0])
        .unsqueeze(0)
        .repeat(batch_size * n_joints, 1)
    )
    quaternions = quaternions.reshape(batch_size * n_joints, 4)
    quaternions = batch_quat_left_multiply(batch_quat_inverse(rotation), quaternions)
    quaternions = quaternions.reshape(batch_size, n_joints, 4)

    # Use these quaternions in the forward kinematics with the Poppy skeleton
    skeleton = forward_kinematics(
        topology, torch.zeros(batch_size, 3), offsets, quaternions
    )[0]

    outputs = skeleton.clone()

    return outputs


def targets_from_skeleton(source_positions, topology, end_effect_precision):
    """
    Extracts targets from skeleton input

    Args:
        source_positions (np.ndarray, torch.Tensor): Shape [Nb of frames within video, Nb of joints, 3 (x,y,z)]
        topology (np.ndarray): Skeleton topology array
        end_effect_precision (int): Precision of the end-effect selection, input from 1 to 3
                                    Defaults to (3) the less precise [13, 16] i.e. left/right hand

    """
    # Works in batched
    batch_size, n_joints, _ = source_positions.shape

    # Measure skeleton bone lengths
    source_lengths = torch.Tensor(batch_size, n_joints)
    for child, parent in enumerate(topology):
        source_lengths[:, child] = torch.sqrt(
            torch.sum(
                (source_positions[:, child] - source_positions[:, parent]) ** 2, axis=-1
            )
        )

    # Find the corresponding angles
    source_offsets = torch.zeros(batch_size, n_joints, 3)
    source_offsets[:, :, -1] = source_lengths
    quaternions = find_quaternions(topology, source_offsets, source_positions)

    # Re-orient according to the pelvis->chest orientation
    base_orientation = (
        quaternions[:, 7:8].repeat(1, n_joints, 1).reshape(batch_size * n_joints, 4)
    )
    base_orientation += 1e-3 * torch.randn_like(base_orientation)
    quaternions = quaternions.reshape(batch_size * n_joints, 4)
    quaternions = batch_quat_left_multiply(
        batch_quat_inverse(base_orientation), quaternions
    )
    quaternions = quaternions.reshape(batch_size, n_joints, 4)

    # Use these quaternions in the forward kinematics with the Poppy skeleton
    target_offsets = torch.zeros(batch_size, n_joints, 3)
    target_offsets[:, :, -1] = POPPY_LENGTHS.unsqueeze(0).repeat(batch_size, 1)
    target_positions = forward_kinematics(
        topology, torch.zeros(batch_size, 3), target_offsets, quaternions
    )[0]

    # Measure the hip orientation
    alpha = np.arctan2(
        target_positions[0, 1, 1] - target_positions[0, 0, 1],
        target_positions[0, 1, 0] - target_positions[0, 0, 0],
    )

    # Rotate by alpha around z
    alpha = alpha
    rotation = (
        torch.Tensor([np.cos(alpha / 2), 0, 0, np.sin(alpha / 2)])
        .unsqueeze(0)
        .repeat(batch_size * n_joints, 1)
    )
    quaternions = quaternions.reshape(batch_size * n_joints, 4)
    quaternions = batch_quat_left_multiply(batch_quat_inverse(rotation), quaternions)
    quaternions = quaternions.reshape(batch_size, n_joints, 4)

    # Use these quaternions in the forward kinematics with the Poppy skeleton
    target_positions = forward_kinematics(
        topology, torch.zeros(batch_size, 3), target_offsets, quaternions
    )[0]

    # Return only target positions for the end-effector of the 6 kinematic chains:
    # Chest, head, left hand, left elbow, left shoulder, right hand, right elbow
    # 8 , 10 , 13, 12, 11, 16, 15
    end_effector_dict = {
        1: [8, 10, 13, 12, 11, 16, 15],
        2: [13, 12, 16, 15],
        3: [13, 16],
    }

    end_effector_indices = end_effector_dict.get(end_effect_precision, [13, 16])

    return target_positions[:, end_effector_indices], target_positions


def interpolate_targets(targets, factor=1):
    """
    Allows to interpolate data points to create more intermediary points in the target array.

    Args:
        targets (np.ndarray): target array
        factor (int, optional): Data points multiplying factor. Defaults to 1 i.e no interpolation.

    """
    length, joints, _ = targets.shape

    new_targets = torch.zeros((length - 1) * factor + 1, joints, 3)

    for i in range(new_targets.shape[0]):
        target_id = float(i / factor)
        before_id = int(np.floor(target_id))
        after_id = int(np.floor(target_id + 1))

        before_coef = 1 - (target_id - before_id)
        after_coef = 1 - (after_id - target_id)

        if after_id > length - 1:
            after_id = length - 1

        new_targets[i] = (
            before_coef * targets[before_id] + after_coef * targets[after_id]
        )

    return new_targets


def moving_average(a, n=3):
    """
    Enables smoother target movements, by computing a moving average of the positions

    Args:
        a (np.ndarray): target array
        n (int, optional): Moving average factor. Defaults to 3.

    """
    repeat_shape = list(a.shape)
    repeat_shape[1:] = [1 for _ in range(len(repeat_shape) - 1)]
    repeat_shape[0] = n // 2
    a = torch.cat([a[:1].repeat(*repeat_shape), a, a[-2:].repeat(*repeat_shape)])
    ret = torch.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
