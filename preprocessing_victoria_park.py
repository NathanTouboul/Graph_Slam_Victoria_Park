from scipy import io
import os
import numpy as np
import matplotlib.pyplot as plt

"""
This python program allows the importation of the necessary data from the Victoria Park dataset. Most of the work is
in the matlab program, we only need to import the following:
- lidar: range, bearing and signature (known correspondence problem) of landmarks detected by the Lidar
    -> needs to be imported through the variable xra created in the matlab code
    -> preprocessing.py stage there: converts lidar raw measurements to landmarks measurements with (approximate) diameters
        signatures
    -> the signatures of the landmarks are given by the diameters of the tree perceived (quite a challenge for a graph 
        slam algorithm as there is a mis association possibility)
- dead_reckoning: speed, steering, (time): motion commands
- gps: latitude, longitude, (time): considered as the position truth
"""

# Obtaining range, bearing and signature of landmarks localization
DATABASE: str = "VictoriaParkDataset"
LIDAR_MEASURES: str = os.path.join(DATABASE, "lidar_measurements.mat")
LIDAR_DATASET: str = os.path.join(DATABASE, "aa3_lsr2.mat")
DEAD_RECKONING_DATASET: str = os.path.join(DATABASE, "aa3_dr.mat")
GPS_DATASET: str = os.path.join(DATABASE, "aa3_gpsx.mat")


def lidar_measurements():

    ranges = io.loadmat(LIDAR_MEASURES)["lidar_ranges"]
    bearings = io.loadmat(LIDAR_MEASURES)["lidar_bearings"]

    diameters = io.loadmat(LIDAR_MEASURES)["lidar_signatures"]

    # The signatures are given by the diameters perceived for each tree
    signatures = 1000 * np.round(diameters, 3)

    # The zero diameters are obviously not real landmarks: we replace it by nan for clarity
    signatures[signatures == 0.] = np.nan

    lidar = [ranges, bearings, signatures]
    lidar_time = io.loadmat(LIDAR_DATASET)["TLsr"][:len(ranges)].flatten()

    return lidar, lidar_time


def dead_reckoning_controls():

    speed = io.loadmat(DEAD_RECKONING_DATASET)["speed"]
    steering = io.loadmat(DEAD_RECKONING_DATASET)["steering"]
    controls = np.stack((speed.flatten(), steering.flatten())).transpose()
    control_time = io.loadmat(DEAD_RECKONING_DATASET)["time"].flatten()

    return controls, control_time


def gps_measurements():

    longitude = io.loadmat(GPS_DATASET)["Lo_m"]
    latitude = io.loadmat(GPS_DATASET)["La_m"]

    gps = np.stack((longitude.flatten(), latitude.flatten())).transpose()

    gps_time = io.loadmat(GPS_DATASET)["timeGps"].flatten()

    return gps, gps_time


def common_time_dataset():

    # Retrieving lidar measurements
    lidars, lidar_time = lidar_measurements()

    # plot_lidar(ranges, bearings, signatures, title="Lidar Measurements")

    # Retrieving dead reckoning
    controls, controls_time = dead_reckoning_controls()

    # Retrieving GPS dataset (truth localisation)
    gps, gps_time = gps_measurements()

    # Creating common vectors for simulating and plotting -> imposed by the smallest time vector
    simulation_time = np.round(gps_time).astype(int)

    # DEBUGGING: n first everything
    n = 11
    controls = controls[:n, :]
    controls_time = controls_time[:n]
    lidars = [dimension[:n, :] for dimension in lidars]
    lidar_time = lidar_time[:n]
    gps = gps[:n, :]
    simulation_time = simulation_time[:n]

    # Computing total number of landmark perceived in the studied timeframe
    landmarks_signatures = []

    for lidar_t in lidars[2]:
        for j, _ in enumerate(lidar_t):
            if lidar_t[j] != 0.:
                landmarks_signatures.append(lidar_t[j])

    landmarks_signatures = sorted(set(landmarks_signatures))
    nb_total_landmarks: int = len(landmarks_signatures)
    print(f"{nb_total_landmarks = }")

    # Simple down sampling of control and lidar -> Could be optimized through interpolation
    step_sim_control = np.floor(len(controls_time) / len(simulation_time)).astype(int)
    step_sim_lidar = np.floor(len(lidar_time) / len(simulation_time)).astype(int)

    gps_sim = gps
    controls_sim = controls[::step_sim_control][:len(simulation_time)]

    lidar_sim = []
    for dimension in lidars:
        lidar_sim.append(dimension[::step_sim_lidar][:len(simulation_time)])

    print(f"\n Simulation: "
          f"Simulation time: {len(simulation_time)} \n"
          f"Shape of controls: {controls_sim.shape} \n"
          f"Shape of lidar: {[dim.shape for dim in lidar_sim]} \n"
          f"Shape of GPS: {gps_sim.shape} \n")

    return controls_sim, lidar_sim, gps_sim, simulation_time, nb_total_landmarks
