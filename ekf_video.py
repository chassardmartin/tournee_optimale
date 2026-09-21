import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Ellipse

### my own imports
from grid import Grid2D
from particle_dynamics import GridParticle
from kalman_filters import EKF
from ekf_particle import h, dh, observe

### functions
def covariance_ellipse(mean, cov, n_std=2.0):
    """
    parameters of the n_std-sigma ellipse of a 2D covariance
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(eigvals[::-1])
    return mean, width, height, angle

def make_video(grid, states, observations, ekf, tau, filename, fps=10, hold_seconds=2):
    """
    animate the true trajectory, the observations and the EKF estimate (with its 2-sigma position ellipse).
    Frame 0 shows the prior, frame k >= 1 the estimate after the k-th correction.
    """
    estimates = np.array([ekf.states_minus[0]] + ekf.states_plus)
    covs = np.array([ekf.Pi_minus_list[0]] + ekf.Pi_plus_list)[:, :2, :2]
    n_frames = len(estimates)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    grid.plot_grid(ax=ax)
    ax.set_xlim(grid.x_min - 0.5, grid.x_max + 0.5)
    ax.set_ylim(grid.y_min - 0.5, grid.y_max + 0.5)

    obs_past = ax.scatter([], [], s=6, color='gray', alpha=0.3, zorder=2, label='observations')
    true_line, = ax.plot([], [], color='black', lw=1.2, zorder=3, label='true trajectory')
    true_dot, = ax.plot([], [], 'o', color='black', ms=6, zorder=5)
    est_line, = ax.plot([], [], color=ekf.color, lw=1.2, zorder=4, label='EKF estimate')
    est_dot, = ax.plot([], [], 'o', color=ekf.color, ms=6, zorder=6)
    ellipse = Ellipse((0, 0), 0, 0, facecolor=ekf.color, edgecolor=ekf.color, alpha=0.15, zorder=4,
                      label=r'EKF $2\sigma$ region')
    ax.add_patch(ellipse)
    title = ax.set_title('')
    ax.legend(loc='upper right')

    def update(frame):
        k = frame
        n = max(k - 1, 0) # time index of the true state
        obs_past.set_offsets(observations[:k].reshape(-1, 2))
        true_line.set_data(states[:n+1, 0], states[:n+1, 1])
        true_dot.set_data([states[n, 0]], [states[n, 1]])
        est_line.set_data(estimates[:k+1, 0], estimates[:k+1, 1])
        est_dot.set_data([estimates[k, 0]], [estimates[k, 1]])

        center, width, height, angle = covariance_ellipse(estimates[k, :2], covs[k])
        ellipse.set_center(center)
        ellipse.set_width(width)
        ellipse.set_height(height)
        ellipse.set_angle(angle)

        err = np.linalg.norm(estimates[k, :2] - states[n, :2])
        label = 'prior' if k == 0 else f't = {n * tau:5.1f}'
        title.set_text(f'{label}    position error = {err:.3f}')
        return obs_past, true_line, true_dot, est_line, est_dot, ellipse, title

    # hold the prior for 1s and the last frame for hold_seconds
    frames = [0] * fps + list(range(n_frames)) + [n_frames - 1] * (hold_seconds * fps)
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(filename, writer=FFMpegWriter(fps=fps))
    plt.close(fig)
    print(f"video saved to {filename}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ### true trajectory: no loop, stops on node (6, 8)
    grid = Grid2D(x_min=0, x_max=10, y_min=0, y_max=10, Nx=10, Ny=10)
    L, R = np.pi/2, -np.pi/2
    turns = {(5, 1): L, (5, 4): R, (8, 4): L, (8, 8): L}

    # tau = 0.1 requires eps > 2 v tau (bumps resolved by the Euler steps) and eps < dx/2
    particle = GridParticle(grid, turns, v=1.0, eps=0.25, tau=0.1)
    tau = particle.tau
    states = particle.simulate(x0=1, y0=1, theta0=0, T=40, stop_node=(6, 8))

    ### observations
    sigma_obs = 0.2                          # std of one position measurement
    W = sigma_obs**2 * tau * np.eye(2)       # observation noise, scaled by 1/tau in the EKF
    observations = observe(states, W, tau, rng)

    ### filter, with a prior far from the trajectory (true start: (1, 1) heading east)
    Q = np.diag([1e-2, 1e-2, 0.3])           # model noise, scaled by tau in the EKF
    x0 = np.array([2.0, 7.0, -np.pi/2])      # prior: (2, 7) heading south
    Pi0 = 0.1 * np.eye(3)                    # confident prior, so the catch is gradual

    ekf = EKF(
        x0=x0, Pi0=Pi0, Q=Q, W=W, tau=tau,
        Phi=particle.phi, dPhi=particle.dphi, h=h, dh=dh,
        observed_trajectory=observations,
        initialize_with_plus=False
    )
    ekf.run_full(len(observations))

    make_video(grid, states, observations, ekf, tau, filename='ekf_catch.mp4')
