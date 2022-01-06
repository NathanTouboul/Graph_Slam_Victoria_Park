import numpy as np

# Disturbance covariance matrix (Rt)
DISTURBANCE_COV = np.diag([0.01**2, 0.01**2, (np.pi/180)**2])
DISTURBANCE_COV_INV = np.linalg.inv(DISTURBANCE_COV)

# Measurement covariance matrix (Qt)
MEASUREMENT_COV = np.diag([0.01**2, (np.pi/180)**2, 0.01 ** 2])
MEASUREMENT_INV = np.linalg.inv(MEASUREMENT_COV)


def graph_slam_linearizing(controls, time_controls, lidars, nb_landmarks, correspondences,  initial_mean_pose):

    # mean pose argument from 0 to t_end
    # other arguments from 1 to t_end

    # composed of 4466 (len(time)) pose * 3 coord + total number of landmarks detected
    n_pose, n_coord = initial_mean_pose.shape[0], initial_mean_pose.shape[1]
    dim_info = n_pose * n_coord + nb_landmarks

    information_matrix = np.zeros((dim_info, dim_info))
    information_vector = np.zeros((dim_info, 1))

    information_matrix[0:3, 0:3] = np.diag([np.inf] * 3)

    # Index of time
    t = 1

    # for all control u_t = (v_t, w_t)
    for control in controls:

        print(f"\n {t = }")
        # Parameters definition
        delta_t = (time_controls[t] - time_controls[t - 1]) / 1000
        speed, angular_vel = control[0], control[1] / delta_t
        v_over_w = speed / angular_vel

        # Previous angular velocity omega_t
        pr_steer = initial_mean_pose[t - 1][2]

        # Linear approximation of the non-linear control function g
        x_hat = np.zeros(np.shape(initial_mean_pose[t - 1]))
        x_hat[0] = initial_mean_pose[t - 1][0] - v_over_w * np.sin(pr_steer) + v_over_w * np.sin(pr_steer + angular_vel * delta_t)
        x_hat[1] = initial_mean_pose[t - 1][1] + v_over_w * np.cos(pr_steer) - v_over_w * np.cos(pr_steer + angular_vel * delta_t)
        x_hat[1] = angular_vel * delta_t

        g_matrix = np.identity(3)
        g_matrix[0][2] = - v_over_w * np.cos(pr_steer) + v_over_w * np.cos(pr_steer + angular_vel * delta_t)
        g_matrix[1][2] = - v_over_w * np.sin(pr_steer) + v_over_w * np.sin(pr_steer + angular_vel * delta_t)

        # Inclusion of a new constraint into the information matrix and vector
        locations_to_update = [t - 1, t]
        constraint_0 = np.vstack([-g_matrix.T, np.identity(3)]) @ DISTURBANCE_COV_INV
        constraint_1 = constraint_0 @ np.hstack([-g_matrix, np.identity(3)])
        constraint_2 = constraint_0 @ (x_hat - g_matrix @ initial_mean_pose[t - 1])

        print(f"{constraint_1.shape = }")
        print(f"{constraint_2.shape = }")

        information_matrix = update_information(information_matrix, constraint_1, locations_to_update)
        information_vector = update_information(information_vector, np.expand_dims(constraint_2, axis=1),
                                                locations_to_update)
        # Updating time
        t += 1

    # Index of time
    t = 1
    # for all measurement z_t
    for _, ranges in enumerate(lidars[1]):

        for j, _ in enumerate(ranges):

            pass




    return information_matrix, information_vector


def update_information(matrix_to_transform, matrix_to_add, location_to_update: list):

    for cpt, loc in enumerate(location_to_update):

        j = 3 * loc
        k = 3 * cpt

        if matrix_to_add.shape[1] > 1:

            print(f"{matrix_to_transform[j:j+3, j:j+3].shape = }")
            print(f"{matrix_to_add[k:k+3, k:k+3].shape = }")

            # Update matrix
            matrix_to_transform[j:j+3, j:j+3] += matrix_to_add[k:k+3, k:k+3]

        if matrix_to_add.shape[1] == 1:
            # Update vector
            matrix_to_transform[j:j + 3, ] +=  matrix_to_add[k:k+3, ]

    return matrix_to_transform
