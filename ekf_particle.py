import numpy as np
import matplotlib.pyplot as plt

### my own imports
from grid import Grid2D
from particle_dynamics import GridParticle
from kalman_filters import EKF

### functions
def h(state):
    """
    observation: position r = (x, y)
    """
    return state[:2]

def dh(state):
    return np.array([[1., 0., 0.], [0., 1., 0.]])

def observe(states, Q, tau, rng):
    """
    noisy observations of the positions, with covariance Q / tau (same scaling as in the EKF)
    """
    noise = rng.multivariate_normal(np.zeros(2), Q / tau, size=len(states))
    return np.array([h(s) for s in states]) + noise

def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    ### true trajectory: closed loop of length 20, 5 left turns and 1 right turn
    grid = Grid2D(x_min=0, x_max=10, y_min=0, y_max=10, Nx=10, Ny=10)
    L, R = np.pi/2, -np.pi/2
    turns = {(6, 1): L, (6, 3): L, (4, 3): R, (4, 6): L, (1, 6): L, (1, 1): L}

    particle = GridParticle(grid, turns, v=1.0, eps=0.05)
    tau = particle.tau
    states = particle.simulate(x0=2, y0=1, theta0=0, T=3*20)

    ### observations
    sigma_obs = 0.2                          # std of one position measurement
    Q = sigma_obs**2 * tau * np.eye(2)       # observation noise, scaled by 1/tau in the EKF
    observations = observe(states, Q, tau, rng)

    ### filter: same phi_tau as the particle, but free to move in the whole plane
    W = np.diag([1e-3, 1e-3, 1.0])           # model noise, scaled by tau in the EKF
    x0 = np.array([2.4, 1.3, 0.4])           # wrong initial guess
    Pi0 = np.diag([0.25, 0.25, 0.5])

    ekf = EKF(
        x0=x0, Pi0=Pi0, Q=Q, W=W, tau=tau,
        Phi=particle.phi, dPhi=particle.dphi, h=h, dh=dh,
        observed_trajectory=observations,
        initialize_with_plus=False
    )
    ekf.run_full(len(observations))
    estimates = np.array(ekf.states_plus)

    ### errors
    pos_err = np.hypot(*(estimates[:, :2] - states[:, :2]).T)
    obs_err = np.hypot(*(observations - states[:, :2]).T)
    head_err = np.abs(wrap(estimates[:, 2] - states[:, 2]))
    half = len(states) // 2
    print(f"position RMSE, 2nd half : EKF {np.sqrt(np.mean(pos_err[half:]**2)):.4f}, "
          f"raw observations {np.sqrt(np.mean(obs_err[half:]**2)):.4f}")
    print(f"max position error, 2nd half : {pos_err[half:].max():.4f}")
    print(f"max heading error, 2nd half : {head_err[half:].max():.4f} rad")

    ### plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    grid.plot_grid(ax=ax1)
    ax1.scatter(observations[:, 0], observations[:, 1], s=1, color='gray', alpha=0.2, zorder=2, label='observations')
    ax1.plot(states[:, 0], states[:, 1], color='black', lw=1.2, zorder=3, label='true trajectory')
    ax1.plot(estimates[:, 0], estimates[:, 1], color=ekf.color, lw=1.2, alpha=0.8, zorder=4, label='EKF estimate')
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 7)
    ax1.legend(loc='upper right')

    t = particle.t
    ax2.plot(t, pos_err, color=ekf.color, lw=1, label='position error')
    ax2.plot(t, head_err, color='orange', lw=1, label='heading error (rad)')
    ax2.set_xlabel('time')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
