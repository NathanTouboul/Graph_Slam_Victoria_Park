import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import datetime
import numpy as np
from preprocessing_victoria_park import down_sampling

BLOCK = False
FIGURES_DIRECTORY = f"figures"
if FIGURES_DIRECTORY not in os.listdir():
    os.mkdir(FIGURES_DIRECTORY)


def animate_lidar(ranges, bearings, signatures, title="Victoria Park\nLidar Measurements", plotting=False, saving=False):

    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title, size=15)
    axes.set_xlim(-50, 50)
    axes.set_ylim(-0.5, 50)
    axes.set_xlabel("Longitude [m]")
    axes.set_ylabel("Latitude [m]")

    # Each time-step plotting the several lasers perceived
    # Maximum landmarks seen at any given time is 18 = ranges.shape[1]
    landmarks_pool = tuple([axes.plot([], [], 'g^', markersize=10)[0]for _ in range(ranges.shape[1])])
    lasers_pool = tuple([axes.plot([], [], '--r', )[0] for _ in range(ranges.shape[1])])
    signatures_pool = tuple([axes.text([], [], '', size=8) for _ in range(ranges.shape[1])])

    def init():
        for landmark in landmarks_pool:
            landmark.set_data((None, None))

        for laser in lasers_pool:
            laser.set_data([], [])

        for signature in signatures_pool:
            signature.set_position((0, 0))
            signature.set_text("")

        return landmarks_pool + lasers_pool + signatures_pool

    def animate(f, ranges_, bearings_, signatures_):

        # The position of the vehicle is the origin as we work in its referential
        x_vehicle, y_vehicle = 0, 0

        # Coordinates of landmarks relative to the vehicle
        x_coord_landmarks = ranges_[f, :] * np.cos(bearings_[f, :])
        y_coord_landmarks = ranges_[f, :] * np.sin(bearings_[f, :])
        signature_plot = sorted([sign for sign in signatures_[f, :]])

        for s, laser in enumerate(lasers_pool):

            if not np.isnan(signature_plot[s]):
                # Laser from vehicle to landmark
                laser.set_data([x_vehicle, x_coord_landmarks[s]], [y_vehicle, y_coord_landmarks[s]])
                # Position landmark
                landmarks_pool[s].set_data(x_coord_landmarks[s], y_coord_landmarks[s])
                # Position text landmarks
                signatures_pool[s].set_position((x_coord_landmarks[s] + 0.5, y_coord_landmarks[s] + 0.5))
                signatures_pool[s].set_text(f"{int(signature_plot[s])}")
            else:
                break

        return landmarks_pool + lasers_pool + signatures_pool

    if plotting or saving:

        anim = FuncAnimation(fig, animate, frames=len(ranges), fargs=(ranges, bearings, signatures), init_func=init,
                             blit=True, interval=200)

        # Plotting animation
        if plotting:
            plt.draw()
            plt.show()

        # Saving the animation as a gif file
        if saving:
            filepath_animation = os.path.join(FIGURES_DIRECTORY, "lidar_measurements.gif")
            anim.save(filepath_animation, writer=PillowWriter(fps=5))

    plt.close()


def animate_vehicle_pose(gps: np.ndarray, initial: np.ndarray, time: np.ndarray, posterior=np.array([]),
                         landmarks=np.array([]), title="Position vehicle", plotting=False, saving=False):

    if not plotting and not saving:
        return

    # Down sampling pose to match gps frequency
    if len(initial) > len(gps):
        initial = down_sampling(initial, gps)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle(title)
    ax.grid()
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
    #if len(landmarks) > 0:
        #ax.scatter(landmarks[:, 0], landmarks[:, 1])
        #signatures_pool = tuple([ax.text([], [], '') for _ in range(len(landmarks))])

        #for s, _ in enumerate(signatures_pool):
            #signatures_pool[s].set_position((landmarks[s, 0] + 0.5, landmarks[s, 1] + 0.5))
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
            return vehicle_gps, path_gps, vehicle_initial, path_init, vehicle_posterior, path_post, pose_gps_text, \
                   time_viz

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

            return vehicle_gps, path_gps, vehicle_initial, path_init, vehicle_posterior, path_post, pose_gps_text, \
                   time_viz

        return vehicle_gps, path_gps, vehicle_initial, path_init, pose_gps_text, time_viz

    if plotting or saving:
        anim = FuncAnimation(fig, animate, frames=len(time), fargs=(gps, initial, posterior, time),
                             init_func=init,  blit=True,
                             repeat=False)
        plt.legend(["GPS Pose", "Initial mean pose", "Posterior pose"], loc='lower right')

        # Plotting animation
        if plotting:
            plt.draw()
            plt.show()

        # Saving the animation as a gif file
        if saving:
            filepath_animation = os.path.join(FIGURES_DIRECTORY, title + ".gif")
            anim.save(filepath_animation, writer=PillowWriter(fps=2))

    plt.draw()
    plt.show(block=BLOCK)


def plot_parameters(initial_pose, gps, simulation_time, title="Parameters", plotting=False, saving=True):

    if not plotting and not saving:
        return

    simulation_time_gps = simulation_time
    # Down sampling pose to match gps frequency
    if len(initial_pose) > len(gps):
        simulation_time_gps = down_sampling(simulation_time, gps)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    fig.suptitle(title)

    axes[0].set_title(f"Initial pose estimate: x and y")
    axes[0].plot(simulation_time, initial_pose[:, 0])

    axes[0].plot(simulation_time_gps, gps[:, 0])
    axes[0].legend(["x initial estimate", "x gps"])
    axes[0].grid()

    axes[1].plot(simulation_time, initial_pose[:, 1])
    axes[1].plot(simulation_time_gps, gps[:, 1])
    axes[1].legend(["y initial estimate", "y gps"])
    axes[1].grid()

    if plotting:
        plt.show(block=BLOCK)

    if saving:
        filepath_figure = os.path.join(FIGURES_DIRECTORY, title)
        plt.savefig(filepath_figure)
        plt.close()


def plot_paths(initial_path, posterior_path, title="Paths", plotting=True, saving=False):

    if not plotting and not saving:
        return

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    fig.suptitle(title)

    axes.set_title(f"Initial pose estimate: x and y")
    axes.plot(initial_path[:, 0], initial_path[:, 1])
    axes.plot(posterior_path[:, 0], posterior_path[:, 1])

    if plotting:
        plt.show()