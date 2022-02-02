import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import numpy as np

BLOCK = False
FIGURES_DIRECTORY = f"figures"
if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def animate_lidar(ranges, bearings, signatures, title="Lidar Measurements of the vehicle"):

    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    axes.grid()
    plt.xlim(-50, 50)
    plt.ylim(-0.5, 50)
    t_end = len(ranges)

    # Each time-step plotting the several lasers perceived
    # Maximum landmarks seen at any given time is 18 = ranges.shape[1]
    landmarks_pool = tuple([axes.plot([], [], marker='o', ls='--', markersize=5)[0] for _ in range(ranges.shape[1])])
    signatures_pool = tuple([axes.text([], [], '') for _ in range(ranges.shape[1])])

    def init():

        for laser in landmarks_pool:
            laser.set_data([], [])

        for signature in signatures_pool:
            signature.set_position((0, 0))
            signature.set_text("")

        return landmarks_pool

    def animate(f, ranges_, bearings_, signatures_):

        x_coord_landmarks = ranges_[f, :] * np.cos(bearings_[f, :])
        y_coord_landmarks = ranges_[f, :] * np.sin(bearings_[f, :])

        # the position of the vehicle is the origin as we work in its referential
        x1, y1 = 0, 0

        for s, laser in enumerate(landmarks_pool):

            if signatures_[f, s] != 0:
                laser.set_data([x1, x_coord_landmarks[s]], [y1, y_coord_landmarks[s]])
                signatures_pool[s].set_position((x_coord_landmarks[s] + 0.5, y_coord_landmarks[s] + 0.5))
                signatures_pool[s].set_text(f"{signatures_[f, s]}")

            else:
                break

        return landmarks_pool

    anim = FuncAnimation(fig, animate, frames=t_end, fargs=(ranges, bearings, signatures), init_func=init, interval=200)

    plt.draw()
    plt.show()


def animate_vehicle_pose(gps: list, initial: list, time: np.ndarray, posterior=np.array([]),
                         landmarks=np.array([]), title="Position vehicle"):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    ax.grid()
    #plt.xlim(np.min(initial[:, 0]) - 1, np.max(initial[:, 0]) + 1)
    #plt.ylim(np.min(initial[:, 1]) - 1, np.max(initial[:, 1]) + 1)

    plt.xlim(np.min(initial[:, 0]) - 1, np.max(initial[:, 0]) + 1)
    plt.ylim(np.min(initial[:, 1]) - 1, np.max(initial[:, 1]) + 1)

    # Vehicle truth - initial pose estimate
    vehicle_gps = ax.plot([], [], 'bx', markersize=3)[0]
    path_gps = ax.plot([], [], '--b', label='_nolegend_')[0]
    path_gps_x, path_gps_y = [], []

    # Vehicle initial mean pose - dead reckoning
    vehicle_initial = ax.plot([], [], 'go', markersize=3)[0]
    path_init = ax.plot([], [], 'g', label='_nolegend_')[0]
    path_init_x, path_init_y = [], []

    # Vehicle Posterior
    if len(posterior) > 0:
        vehicle_posterior = ax.plot([], [], 'rX', markersize=3)[0]
        path_post = ax.plot([], [], ':r', label='_nolegend_')[0]
        path_post_x, path_post_y = [], []

    # Landmarks
    if len(landmarks) > 0:
        ax.scatter(landmarks[:, 0], landmarks[:, 1])
        signatures_pool = tuple([ax.text([], [], '') for _ in range(len(landmarks))])
        for s, _ in enumerate(signatures_pool):
            signatures_pool[s].set_position((landmarks[s, 0] + 0.5, landmarks[s, 1] + 0.5))
            #signatures_pool[s].set_text(f"{int(landmarks[s, 2])}")

    # Information to display
    pose_gps_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    time_viz = ax.text(0.02, 0.9, '', transform=ax.transAxes)

    def init():

        # Vehicle GPS
        vehicle_gps.set_data([], [])
        path_gps.set_data([], [])

        # Dead Reckoning
        vehicle_initial.set_data([], [])
        path_init.set_data([], [])

        # Texts
        pose_gps_text.set_text('')
        time_viz.set_text('')

        # Vehicle posterior pose
        if len(posterior) > 0:
            vehicle_posterior.set_data([], [])
            path_post.set_data([], [])
            return vehicle_gps, path_gps, vehicle_initial, path_init, vehicle_posterior, path_post, pose_gps_text, time_viz

        return vehicle_gps, path_gps, vehicle_initial, path_init, pose_gps_text, time_viz

    def animate(f, gps_, initial_, posterior_, time_):

        # Vehicle GPS
        vehicle_gps.set_data(gps_[f][0], gps_[f][1])
        path_gps_x.append(gps_[f][0])
        path_gps_y.append(gps_[f][1])
        path_gps.set_data(path_gps_x, path_gps_y)

        # Dead Reckoning
        vehicle_initial.set_data(initial_[f][0], initial_[f][1])
        path_init_x.append(initial_[f][0])
        path_init_y.append(initial_[f][1])
        path_init.set_data(path_init_x, path_init_y)

        # Texts
        time_viz.set_text(f"Time {datetime.timedelta(milliseconds=int(time_[f]))} s \n")
        pose_gps_text.set_text(f"Latitude: {np.round(gps_[f][0], 2)} m \n"
                               f"Longitude: {np.round(gps_[f][1], 2)} m")
        # Vehicle posterior pose
        if len(posterior) > 0:
            vehicle_posterior.set_data(posterior_[f][0], posterior_[f][1])
            path_post_x.append(posterior_[f][0])
            path_post_y.append(posterior_[f][1])
            path_post.set_data(path_post_x, path_post_y)

            return vehicle_gps, path_gps, vehicle_initial, path_init, vehicle_posterior, path_post, pose_gps_text, time_viz

        return vehicle_gps, path_gps, vehicle_initial, path_init, pose_gps_text, time_viz

    anim_gps = FuncAnimation(fig, animate, frames=len(time), fargs=(gps, initial, posterior, time),
                             init_func=init,  blit=True,
                             repeat=False)

    plt.legend(["GPS Pose", "Initial mean pose", "Posterior pose"], loc='lower right')

    plt.draw()
    plt.show()


def plot_parameters(initial_pose, gps, simulation_time, title="Parameters"):

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    fig.suptitle(title)

    axes[0].set_title(f"Initial pose estimate: x and y")
    axes[0].plot(simulation_time, initial_pose[:, 0])

    axes[0].plot(simulation_time, gps[:, 0])
    axes[0].legend(["x initial estimate", "x gps"])
    axes[0].grid()

    axes[1].plot(simulation_time, initial_pose[:, 1])
    axes[1].plot(simulation_time, gps[:, 1])
    axes[1].legend(["y initial estimate", "y gps"])
    axes[1].grid()

    plt.show()
