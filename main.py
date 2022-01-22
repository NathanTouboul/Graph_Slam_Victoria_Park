import matplotlib.pyplot as plt

import numpy as np
from preprocessing_victoria_park import common_time_dataset
from plotting_saving import animate_lidar, animate_vehicle_pose, plot_parameters
from graph_slam import graph_slam_initialize, graph_slam_linearizing, graph_slam_reduce, graph_slam_solve


def main():

    dimension_trial = 25

    # Victoria Dataset
    controls, lidars, correspondences, gps, simulation_time = common_time_dataset(dimension_trial)

    # Animation lidars
    # animate_lidar(lidars[0], lidars[1], lidars[2])

    # Initialization of the Graph Slam Algorithm
    # Initial pose known
    origin_pose = np.array([gps[0][0], gps[0][1], 0])
    initial_mean_pose = graph_slam_initialize(controls[1:, ], simulation_time, origin_pose=origin_pose)

    print(f"{initial_mean_pose.shape = }")

    # Linearizing step of the Graph Slam Algorithm
    controls_1_t = controls[1:, ]
    lidars_1_t = [dimension[1:, ] for dimension in lidars]

    information_matrix, information_vector = graph_slam_linearizing(controls_1_t, simulation_time, lidars_1_t,
                                                                    correspondences,
                                                                    initial_mean_pose)

    # Dimension reduction step of the Graph Slam Algorithm
    information_matrix_red, information_vector_red = graph_slam_reduce(information_matrix, information_vector,
                                                                       correspondences)

    # Solving step of the Graph Slam Algorithm
    posterior_pose, covariance_path = graph_slam_solve(information_matrix_red, information_vector_red,
                                                       information_matrix, information_vector, correspondences)

    # Plotting Parameters
    #plot_parameters(controls, initial_mean_pose, gps, simulation_time)

    # Animation Vehicle poses and landmarks
    j_initial = next(iter(correspondences.items()))[1]
    posterior_path, landmarks = posterior_pose[:j_initial, :], posterior_pose[j_initial:, :]

    animate_vehicle_pose(gps, initial_mean_pose, posterior_path, landmarks, simulation_time)


if __name__ == '__main__':

    main()
