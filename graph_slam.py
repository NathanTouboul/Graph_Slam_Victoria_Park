import numpy as np

# Disturbance noise covariance matrix (Rt)
DISTURBANCE_COV = np.diag([0.01**2, 0.01**2, (np.pi/180)**2])
DISTURBANCE_COV_INV = np.linalg.inv(DISTURBANCE_COV)

# Measurement noise covariance matrix (Qt)
MEASUREMENT_COV = np.diag([0.01**2, (np.pi/180)**2, 0.01 ** 2])
MEASUREMENT_COV_INV = np.linalg.inv(MEASUREMENT_COV)


def graph_slam_initialize(controls: np.ndarray, time_controls: np.ndarray, origin_pose=np.zeros((3, 1))):

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


def graph_slam_linearizing(controls, time, lidars, correspondences,  initial_mean_pose):

    # mean pose argument from 0 to t_end
    # other arguments from 1 to t_end

    nb_landmarks = len(correspondences)

    # composed of 4466 (len(time)) pose * 3 coord + total number of landmarks detected
    n_pose, n_coord = initial_mean_pose.shape[0], initial_mean_pose.shape[1]

    # 3 coordinates for vehicle: x, y and theta and tree coordinates for landmarks: x, y and signature
    dim_info = n_pose * n_coord + nb_landmarks * 3

    information_matrix = np.zeros((dim_info, dim_info))
    information_vector = np.zeros((dim_info, 1))

    # Adding infinite information to fixe the first pose at the origin (otherwise matrix might become singular)
    information_matrix[0:3, 0:3] = np.diag([np.inf] * 3)

    # Initialization pose: vehicle: x, y, theta then landmarks: x, y, signature
    pose = np.zeros((n_pose + nb_landmarks, n_coord))
    pose[:len(initial_mean_pose), :] = initial_mean_pose

    # Index of time
    t = 1

    # for all control u_t = (v_t, w_t)
    for control in controls:

        print(f"\n {t = }")
        # Parameters definition
        delta_t = (time[t] - time[t - 1]) / 1000
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
        location_to_update = [t - 1, t]
        constraint_0 = np.vstack([-g_matrix.T, np.identity(3)]) @ DISTURBANCE_COV_INV
        constraint_1 = constraint_0 @ np.hstack([-g_matrix, np.identity(3)])
        constraint_2 = constraint_0 @ (x_hat - g_matrix @ initial_mean_pose[t - 1])

        # print(f"{constraint_1.shape = }")
        # print(f"{constraint_2.shape = }")

        information_matrix = update_matrix(information_matrix, constraint_1, location_to_update)
        information_vector = update_matrix(information_vector, np.expand_dims(constraint_2, axis=1),
                                           location_to_update)
        # Updating time
        t += 1

    # for all measurement z_t
    already_seen = []
    # print("\n For all measurements")

    for t, (ranges, bearings, signatures) in enumerate(zip(lidars[0], lidars[1], lidars[2])):

        # Index of time
        t += 1

        print(f"\n {t = }")

        for s, signature in enumerate(signatures):

            # At each time step, we do not necessarily have the same number of perceived landmarks
            if np.isnan(signature):
                break

            """ Computing the Taylor Expansion of the measurement function """

            # Obtaining the corresponding index of a particular signature
            j = correspondences[signature]

            # If the landmarks has never been seen before, we initialize the position of the landmark as if the position
            # of the vehicle was perfectly known
            if j not in already_seen:
                range_to_landmark = ranges[s]
                new_position_landmark = pose[t, :2] + range_to_landmark
                pose[j, :] = np.hstack([new_position_landmark, signature])
                already_seen.append(j)

            # Delta vector
            delta_x = pose[j, 0] - pose[t, 0]
            delta_y = pose[j, 1] - pose[t, 1]
            delta = np.array([[delta_x], [delta_y]])

            # q
            q = delta.T @ delta

            # Measurement and revised measurements
            sqrt_q = np.sqrt(q)
            z, z_hat = np.zeros((3, 1)), np.zeros((3, 1))
            z[0], z[1], z[2] = ranges[s], bearings[s], signatures[s]
            z_hat[0], z_hat[1], z_hat[2] = sqrt_q, np.arctan2(delta_y, delta_x) - pose[t, 2], signature

            # h matrix
            sqrt_q_time_delta_x, sqrt_q_time_delta_y = sqrt_q * delta_x, sqrt_q * delta_y
            h = np.zeros((3, 6))
            h[0, :] = [-sqrt_q_time_delta_x, -sqrt_q_time_delta_y, 0, sqrt_q_time_delta_x, sqrt_q_time_delta_y, 0]
            h[1, :] = [delta_y, -delta_x, -q, -delta_y, delta_x, 0]
            h[2, 5] = q

            h = h / q if q != 0 else h

            # Adding information into the matrices
            location_to_update = [t, j]

            mu = np.zeros((6, 1))
            mu[0], mu[1], mu[2] = pose[t, :]
            mu[3], mu[4], mu[5] = pose[j, :]

            ht_q_inv = h.T @ MEASUREMENT_COV_INV
            ht_q_inv_h = ht_q_inv @ h
            measurement_error = z - z_hat

            ht_q_inv_z_plus_h_mu = ht_q_inv @ (measurement_error + h @ mu)

            information_matrix = update_matrix(information_matrix, ht_q_inv_h, location_to_update)
            information_vector = update_matrix(information_vector, ht_q_inv_z_plus_h_mu, location_to_update)

    return information_matrix, information_vector


def graph_slam_reduce(information_matrix: np.ndarray, information_vector: np.ndarray, correspondences: dict) -> tuple:

    reduced_matrix = information_matrix[:]
    reduced_vector = information_vector[:]

    # The set of poses at which a feature is observed is given by the non zero rows/columns of the information matrices
    # for each feature row.
    # The correspondences dictionary is used to iterate over all features in the information matrices

    for signature, j in correspondences.items():

        pass

    return reduced_matrix, reduced_vector

def graph_slam_solve(reduced_info_matrix, reduced_info_vector, information_matrix, information_vector) -> tuple:
    pass


def update_matrix(matrix_to_transform: np.ndarray, matrix_to_add: np.ndarray, location_to_update: list) -> np.ndarray:

    for cpt, loc in enumerate(location_to_update):

        j = 3 * loc
        k = 3 * cpt

        if matrix_to_add.shape[1] > 1:

            # print(f"{matrix_to_transform[j:j+3, j:j+3].shape = }")
            # print(f"{matrix_to_add[k:k+3, k:k+3].shape = }")

            # Update matrix
            matrix_to_transform[j:j+3, j:j+3] += matrix_to_add[k:k+3, k:k+3]

        elif matrix_to_add.shape[1] == 1:
            # Update vector
            matrix_to_transform[j:j + 3, ] += matrix_to_add[k:k+3, ]

    return matrix_to_transform
