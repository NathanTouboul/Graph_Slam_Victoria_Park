import matplotlib.pyplot as plt

import numpy as np

from preprocessing_victoria_park import common_time_dataset
from plotting_saving import animate_lidar, animate_vehicle_pose, plot_parameters
from initialization import graph_slam_initialize
from linearize import graph_slam_linearizing


def main():

    # Victoria Dataset
    controls, lidars, gps, simulation_time, nb_total_landmarks = common_time_dataset()

    # Animation lidars
    #animate_lidar(lidars[0], lidars[1], lidars[2])

    # Initialization of the Graph Slam Algorithm
    initial_mean_pose = graph_slam_initialize(controls[1:, ], simulation_time, origin_pose=np.zeros((3,)))
    print(f"{initial_mean_pose.shape = }")

    # Linearizing step of the Graph Slam Algorithm
    corr = np.array([])
    lidars_1_t = [dimension[1:, ] for dimension in lidars]
    information_matrix, information_vector = graph_slam_linearizing(controls[1:, ], simulation_time, lidars_1_t,
                                                                    nb_total_landmarks, corr, initial_mean_pose[1:, ])


    # Plotting Parameters
    #plot_parameters(controls, initial_pose, gps, simulation_time)

    #animate_vehicle_pose(gps, initial_pose, simulation_time)


if __name__ == '__main__':

    main()
