import numpy as np

# Disturbance noise covariance matrix (Rt)
DISTURBANCE_COV = np.diag([0.01**2, 0.01**2, (np.pi/180)**2])
DISTURBANCE_COV_INV = np.linalg.pinv(DISTURBANCE_COV)

# Measurement noise covariance matrix (Qt)
MEASUREMENT_COV = np.diag([0.01**2, (np.pi/180)**2, 0.01 ** 2])
MEASUREMENT_COV_INV = np.linalg.pinv(MEASUREMENT_COV)

# Infinity value
INF = 1E15

# Number of coordinates for pose (x, y, theta) and landmarks (x, y and s)
n_coord_pose, n_coord_landmarks = 3, 3

# Storing inverse calculation of the information matrix (j: j:j_end)
sub_matrices_inv = dict()

# Distance between wheels
L = 2.83   # meters - span between wheels
A = 0.95   # meters - distance from center of mass to laser - x
B = 0.5    # meters - distance from center of mass to laser - y
H = 0.76   # meters - distance from center of mass to encoder - y


def graph_slam_initialize(controls: np.ndarray, time: np.ndarray, origin_pose=np.zeros((3, 1))):
    """
    Initial estimate is provided by chaining together the motion model
    :param controls:
    :param time:
    :param origin_pose:
    :return:
    """

    global n_coord_pose, L

    pose = np.zeros((len(time), n_coord_pose), )
    pose[0, :] = origin_pose

    # Array of controls: speed and steering
    for t, control in enumerate(controls):

        # Updating time
        t += 1

        # Parameters definition
        delta_t = (time[t] - time[t - 1]) / 1000    # milliseconds to seconds
        speed, steering = control[0], control[1]
        heading = pose[t - 1][2]

        # Compute pose_x
        pose[t][0] = pose[t - 1][0] + delta_t * (speed * np.cos(heading) - (speed / L) * np.tan(steering)
                                                 * (A * np.sin(heading) + B * np.cos(heading)))

        # Compute pose_y
        pose[t][1] = pose[t - 1][1] + delta_t * (speed * np.sin(heading) + (speed / L) * np.tan(steering)
                                                 * (A * np.cos(heading) - B * np.sin(heading)))

        # Compute heading
        pose[t][2] = pi_to_pi(heading + (delta_t * speed / L) * np.tan(steering))

    return pose


def graph_slam_linearizing(controls, time, lidars, correspondences,  initial_mean_pose):
    """

    :param controls:
    :param time:
    :param lidars:
    :param correspondences:
    :param initial_mean_pose:
    :return:
    """
    print(f"Linearizing step")

    # mean pose argument from 0 to t_end - other arguments from 1 to t_end
    nb_landmarks = len(correspondences)

    # composed of 4466 (len(time)) pose * 3 coord + total number of landmarks detected
    n_pose, n_coord = initial_mean_pose.shape[0], initial_mean_pose.shape[1]

    # 3 coordinates for vehicle: x, y and theta and tree coordinates for landmarks: x, y and signature
    dim_info = n_pose * n_coord + nb_landmarks * n_coord_landmarks

    information_matrix = np.zeros((dim_info, dim_info))
    information_vector = np.zeros((n_pose + nb_landmarks, n_coord))

    # Adding infinite information to fixe the first pose at the origin (otherwise matrix might become singular)
    information_matrix[0:3, 0:3] = np.diag([INF] * 3)

    # Initialization pose: vehicle: x, y, theta then landmarks: x, y, signature
    pose = np.zeros((n_pose + nb_landmarks, n_coord))
    pose[:len(initial_mean_pose), :] = initial_mean_pose

    # Index of time
    t = 1

    # for all control u_t = (v_t, w_t)
    for control in controls:

        # Parameters definition
        delta_t = (time[t] - time[t - 1]) / 1000  # milliseconds to seconds
        speed, steering = control[0], control[1]
        heading = initial_mean_pose[t - 1][2]

        # Linear approximation of the non-linear control function g
        x_hat = np.zeros((np.shape(initial_mean_pose[t - 1])[0], 1))
        x_hat[0] = initial_mean_pose[t - 1][0] + delta_t * (speed * np.cos(heading) - (speed / L) * np.tan(steering)
                                                            * (A * np.sin(heading) + B * np.cos(heading)))
        x_hat[1] = initial_mean_pose[t - 1][1] + delta_t * (speed * np.sin(heading) + (speed / L) * np.tan(steering)
                                                            * (A * np.cos(heading) - B * np.sin(heading)))
        x_hat[2] = pi_to_pi(heading + (delta_t * speed / L) * np.tan(steering))

        g_matrix = np.identity(3)
        g_matrix[0][2] = - delta_t * speed * np.sin(heading) - \
            (delta_t * speed / L) * np.tan(steering) * (A * np.cos(heading) - B * np.sin(heading))
        g_matrix[1][2] = + delta_t * speed * np.cos(heading) - \
            (delta_t * speed / L) * np.tan(steering) * (- A * np.sin(heading) - B * np.cos(heading))

        # Inclusion of a new constraint into the information matrix and vector
        location_to_update = [t - 1, t]
        constraint_0 = np.vstack([-g_matrix.T, np.identity(3)]) @ DISTURBANCE_COV_INV
        constraint_1 = constraint_0 @ np.hstack([-g_matrix, np.identity(3)])
        constraint_2 = constraint_0 @ (x_hat - g_matrix @ np.expand_dims(initial_mean_pose[t - 1, :], 1))

        information_matrix = update_matrix(information_matrix, constraint_1, location_to_update)
        information_vector = update_matrix(information_vector, constraint_2.T, location_to_update)

        # Updating time
        t += 1

    # for all measurement z_t
    already_seen = list()

    for t, (ranges, bearings, signatures) in enumerate(zip(lidars[0], lidars[1], lidars[2])):

        # Index of time
        t += 1

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

            #
            q = delta.T @ delta

            # Raw and revised measurements
            sqrt_q = np.sqrt(q)
            z, z_hat = np.zeros((3, 1)), np.zeros((3, 1))
            z[0], z[1], z[2] = ranges[s], bearings[s], signatures[s]
            z_hat[0], z_hat[1], z_hat[2] = sqrt_q, np.arctan2(delta_y, delta_x) - pose[t, 2] - np.pi / 2, signature

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
            information_vector = update_matrix(information_vector, ht_q_inv_z_plus_h_mu.T, location_to_update)

    return information_matrix, information_vector


def graph_slam_reduce(information_matrix: np.ndarray, information_vector: np.ndarray, correspondences: dict) -> tuple:
    """

    :param information_matrix:
    :param information_vector:
    :param correspondences:
    :return:
    """

    global n_coord_landmarks
    global sub_matrices_inv

    print(f"Reducing step")

    # Initializing
    sub_matrices_inv = dict()

    # The last time t (t + 1 actually) is given by the first index of feature: j
    # Obtaining the first value of index (the dictionary was constructed sorted)
    j_initial = next(iter(correspondences.items()))[1]

    # We initialize the reduce matrix as information_matrix[0:tend, 0:tend]
    reduced_matrix = information_matrix[:j_initial, :j_initial]
    reduced_vector = information_vector[:j_initial, :]

    # The set of poses at which a feature is observed is given by the non zero rows/columns of the information matrices
    # for each feature row.
    # The correspondences dictionary is used to iterate over all features in the information matrices

    matrix_subtract = np.zeros_like(reduced_matrix, )
    vector_subtract = np.zeros_like(reduced_vector, )

    # Sequential update
    for _, j in correspondences.items():

        # omega j,j
        sub_matrix = information_matrix[j: j + 3, j: j + 3]
        sub_matrix_inv = np.linalg.pinv(sub_matrix)
        temp_matrix = information_matrix[:j_initial, j: j + 3] @ sub_matrix_inv

        # Storing sub matrix inverse for using in the solver
        sub_matrices_inv[j] = sub_matrix_inv

        matrix_subtract += temp_matrix @ information_matrix[j: j + 3, :j_initial]
        vector_subtract += temp_matrix @ np.expand_dims(information_vector[j, :], 1)

    # Final calculations
    reduced_matrix -= matrix_subtract   # (11.35)
    reduced_vector -= vector_subtract   # (11.36)

    return reduced_matrix, reduced_vector


def graph_slam_solve(reduced_info_m: np.ndarray, reduced_info_v: np.ndarray, information_m: np.ndarray,
                     information_v: np.ndarray, correspondences: dict) -> tuple:
    """

    :param reduced_info_m:
    :param reduced_info_v:
    :param information_m:
    :param information_v:
    :param correspondences:
    :return:
    """

    print(f"Solving step")
    global n_coord_pose
    global sub_matrices_inv

    # First index of feature in the information arrays
    j_initial = next(iter(correspondences.items()))[1]

    # Compute covariance posterior paths
    covariance_path_posterior = np.linalg.inv(reduced_info_m)

    # Initialization: posterior pose mu
    pose_posterior = np.zeros((len(information_m), n_coord_pose), )

    # Compute path estimates - (11.38)
    pose_posterior[:j_initial, :] = covariance_path_posterior @ reduced_info_v

    """
    # Computing posterior pose of each landmarks - (11.44)
    for _, j in correspondences.items():

        covariance_feature = sub_matrices_inv[j]
        information_vector_feature = np.expand_dims(information_v[j, :], 1)
        pose_posterior_j = covariance_feature \
            @ (information_vector_feature + information_m[j: j + 3, :j_initial] @ pose_posterior[:j_initial, :])

        pose_posterior[j, :] = pose_posterior_j.T
    
    """

    return pose_posterior, covariance_path_posterior


def graph_slam_is_converging(prior_pose: np.ndarray, posterior_pose: np.ndarray, truth) -> bool:
    """
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(prior_pose[:, 0])
    axes.plot(prior_pose[:, 1])
    axes.plot(posterior_pose[:, 0])
    axes.plot(posterior_pose[:, 1])
    plt.legend(["Prior x", "Prior y", "Posterior x", "Posterior y"])
    plt.show()
    """
    root_sum_square_pp = np.sqrt(np.square(prior_pose[:, 0] - posterior_pose[:, 0]) + np.square(prior_pose[:, 1]
                                                                                                - posterior_pose[:, 1]))

    root_sum_square_truth = np.sqrt(np.square(truth[:, 0] - posterior_pose[:, 0]) + np.square(truth[:, 1]
                                                                                            - posterior_pose[:, 1]))
    print(f"{np.mean(root_sum_square_truth) = }")
    if np.all(root_sum_square_pp) < 0.001:
        return True

    return False


def update_matrix(matrix_to_transform: np.ndarray, matrix_to_add: np.ndarray, location_to_update: list) -> np.ndarray:

    # Update matrix
    if matrix_to_add.shape[0] > 1:
        loc1, loc2 = 3 * location_to_update[0], 3 * location_to_update[1]
        matrix_to_transform[loc1:loc1+3, loc1:loc1+3] += matrix_to_add[0:3, 0:3]
        matrix_to_transform[loc1:loc1+3, loc2:loc2+3] += matrix_to_add[0:3, 3:6]
        matrix_to_transform[loc2:loc2+3, loc1:loc1+3] += matrix_to_add[3:6, 0:3]
        matrix_to_transform[loc2:loc2+3, loc2:loc2+3] += matrix_to_add[3:6, 3:6]

    # Update vector
    else:
        loc1, loc2 = location_to_update[0], location_to_update[1]
        matrix_to_transform[loc1, :] += matrix_to_add[0, 0:3]
        matrix_to_transform[loc2, :] += matrix_to_add[0, 3:6]

    return matrix_to_transform


def get_feature_matrix(n_rows: int, n_cols: int, j: int) -> np.ndarray:

    feature_matrix = np.zeros((n_rows, n_cols))
    feature_matrix[:, j:j+3] = np.identity(n_rows)  # (11.34)

    return feature_matrix


def pi_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
