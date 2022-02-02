import cProfile
import pstats

import numpy as np
from preprocessing_victoria_park import common_time_dataset
from plotting_saving import animate_lidar, animate_vehicle_pose, plot_parameters

from graph_slam import graph_slam_initialize, graph_slam_linearizing, graph_slam_reduce, graph_slam_solve, \
    graph_slam_is_converging


def main():

    # Victoria Dataset
    controls, lidars, correspondences, gps, simulation_time = common_time_dataset(timeframe=1000)

    # Animation lidars
    # animate_lidar(lidars[0], lidars[1], lidars[2])

    # Initialization of the Graph Slam Algorithm
    # Initial pose known
    origin_pose = np.array([gps[0][0], gps[0][1], np.pi / 4])
    prior_pose = graph_slam_initialize(controls[1:, ], simulation_time, origin_pose=origin_pose)

    # Copying dead reckoning for plotting
    initial_mean_pose = prior_pose[:]

    """
    # Plotting Parameters
    plot_parameters(prior_pose, gps, simulation_time)
    """

    # animate_vehicle_pose(gps, initial_mean_pose, simulation_time)

    # Linearizing step of the Graph Slam Algorithm
    controls_1_t = controls[1:, ]
    lidars_1_t = [dimension[1:, ] for dimension in lidars]

    convergence: bool = False

    while not convergence:

        information_matrix, information_vector = graph_slam_linearizing(controls_1_t, simulation_time, lidars_1_t,
                                                                        correspondences,
                                                                        prior_pose)

        # Dimension reduction step of the Graph Slam Algorithm
        information_matrix_red, information_vector_red = graph_slam_reduce(information_matrix, information_vector,
                                                                           correspondences)

        # Solving step of the Graph Slam Algorithm
        posterior_pose, covariance_path = graph_slam_solve(information_matrix_red, information_vector_red,
                                                           information_matrix, information_vector, correspondences)

        # Does it converge
        convergence = graph_slam_is_converging(prior_pose, posterior_pose)

        # Next iteration is based on the posterior
        prior_pose = posterior_pose

    """
    # Animation Vehicle poses and landmarks
    j_initial = next(iter(correspondences.items()))[1]
    posterior_path, landmarks = posterior_pose[:j_initial, :], posterior_pose[j_initial:, :]

    animate_vehicle_pose(gps, initial_mean_pose, simulation_time, posterior_path, landmarks)
    """


if __name__ == '__main__':

    with cProfile.Profile() as profile:
        main()

    stats = pstats.Stats(profile).sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename="profiling_graph_slam.prof")

