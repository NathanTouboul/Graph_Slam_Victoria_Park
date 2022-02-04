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


def common_time_dataset(timeframe=-1):

    # Retrieving lidar measurements
    lidars, lidar_time = lidar_measurements()

    # plot_lidar(ranges, bearings, signatures, title="Lidar Measurements")

    # Retrieving dead reckoning
    controls, controls_time = dead_reckoning_controls()

    # Retrieving GPS dataset (truth localisation)
    gps, gps_time = gps_measurements()

    # Simulating time -> imposed by the smallest frequency between controls and measurements (lidar)
    simulation_time = np.round(lidar_time)

    # Timeframe studied
    controls = controls[:timeframe:, :]
    controls_time = controls_time[:timeframe:]
    lidars = [dimension[:timeframe:, :] for dimension in lidars]
    gps = gps[:timeframe:, :]
    simulation_time = simulation_time[:timeframe:]

    # Simple down sampling of control -> Can be optimized through interpolation
    controls_sim = down_sampling(controls, lidars[0])

    lidar_sim = []
    for dimension in lidars:
        lidar_sim.append(dimension[::][:len(simulation_time)])

    # Obtaining unique signatures
    landmarks_signatures = set()
    for signatures in lidar_sim[2]:
        for signature in signatures:
            if np.isnan(signature):
                break
            landmarks_signatures.add(signature)

    # Sorting the landmark signatures
    landmarks_signatures_sorted = sorted(landmarks_signatures)

    # Building a correspondences dictionary: from a signature to an index
    correspondences = {signature: j + len(simulation_time) for j, signature in enumerate(landmarks_signatures_sorted)}

    print(f"Simulation parameters: \n"
          f"\tSimulation time: {len(simulation_time)} \n"
          f"\tShape of controls: {controls_sim.shape} \n"
          f"\tShape of lidar: {[dim.shape for dim in lidar_sim]} \n"
          f"\tShape of GPS: {gps.shape}\n")

    return controls_sim, lidar_sim, correspondences,  gps, simulation_time


def down_sampling(high_frequency_signal1, low_frequency_signal2):

    step = np.floor(len(high_frequency_signal1) / len(low_frequency_signal2)).astype(int)
    return high_frequency_signal1[::step][:len(low_frequency_signal2)]
