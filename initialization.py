import numpy as np


def graph_slam_initialize(controls: np.ndarray, time_controls: np.ndarray, origin_pose=np.zeros((3, ))):
    """
    Initial estimate is provided by chaining together the motion model
    :param origin_pose:
    :param time_controls:
    :param controls:
    :return:
    """

    pose = np.zeros((len(time_controls), 3))
    pose[0, :] = origin_pose

    # Index of time
    t = 1

    # Array of controls: speed and steering
    for control in controls:

        # Parameters definition
        delta_t = (time_controls[t] - time_controls[t-1]) / 1000
        speed, angular_vel = control[0], control[1] / delta_t
        v_over_w = speed / angular_vel

        # Previous angular velocity omega_t
        pr_steer = pose[t - 1][2]

        # Compute pose_x
        pose[t][0] = pose[t - 1][0] - v_over_w * np.sin(pr_steer) + v_over_w * np.sin(pr_steer + angular_vel * delta_t)

        # Compute pose_y
        pose[t][1] = pose[t - 1][1] + v_over_w * np.cos(pr_steer) - v_over_w * np.cos(pr_steer + angular_vel * delta_t)

        # Compute pose_omega
        pose[t][2] = angular_vel * delta_t

        # Updating time
        t += 1

    return pose
