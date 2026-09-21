import numpy as np
import matplotlib.pyplot as plt

### my own imports
from grid import Grid2D

### classes
class GridParticle:
    """
    Particle moving at constant speed along the lines of a Grid2D and turning at some nodes.
    Regularized ODE (eps > 0) on the state (x, y, theta):

        x'     = v cos(theta) - lam sin^2(theta) dx/(2 pi) sin(2 pi (x - x_min) / dx)
        y'     = v sin(theta) - lam cos^2(theta) dy/(2 pi) sin(2 pi (y - y_min) / dy)
        theta' = sum_n psi_eps(|r - r_n|) Delta_n  -  k sin(4 theta),   k = kappa v / eps

    - psi_eps(d) = v/eps * eta(d/eps) is a bump of width eps around each turning node, normalized
      so that a straight pass through the node centre rotates the heading by exactly Delta_n
    - the -k sin(4 theta) term snaps the heading to the axis directions (multiples of pi/2)
    - the lam terms pull the particle back onto the nearest grid line, transversally to its motion

    Time discretization: explicit Euler, x_{n+1} = phi_tau(x_n) = x_n + tau f(x_n)
    """

    def __init__(self, grid:Grid2D, turns, v=1.0, eps=0.05, kappa=0.4, lam=8.0, tau=None):
        """
        Args:
            - grid : Grid2D  the grid the particle moves on
            - turns : dict {(i, j): Delta}  turning nodes (grid indices) and their relative
                      rotation, e.g. +np.pi/2 (left), -np.pi/2 (right). Other nodes are crossed straight
            - v : float  speed
            - eps : float > 0  width of the turning bumps
            - kappa : float  heading snap rate, in units of v/eps
            - lam : float  rate of the transverse stabilizer (1/time)
            - tau : float  Euler time step, default eps/10. Must resolve the bumps (tau << eps/v)
        """
        assert eps > 0, "only the regularized model (eps > 0) is implemented"
        assert 2 * eps < min(grid.dx, grid.dy), "bumps of neighbouring nodes overlap, reduce eps"
        for node, delta in turns.items():
            assert delta != 0 and np.isclose(delta / (np.pi/2), np.round(delta / (np.pi/2))), \
                f"turn at node {node} must be a nonzero multiple of pi/2"
        # the snap must stay weaker than the peak kick of the bump, or the heading never turns
        assert kappa < self.eta(0.0) * min(abs(d) for d in turns.values()), "kappa too large"
        tau = eps / (10 * v) if tau is None else tau
        assert tau * v < eps / 2, "tau too large, Euler steps would jump over the turning bumps"

        self.grid = grid
        self.turns = turns
        self.v = v
        self.eps = eps
        self.k = kappa * v / eps  # snap rate scales with the bump (see notes)
        self.lam = lam
        self.tau = tau

        _xx, _yy = grid.vals
        idx = list(turns.keys())
        self.nodes_x = np.array([_xx[i, j] for i, j in idx])
        self.nodes_y = np.array([_yy[i, j] for i, j in idx])
        self.deltas = np.array([turns[n] for n in idx])

        # filled by simulate()
        self.t = None
        self.x = None
        self.y = None
        self.theta = None

    @staticmethod
    def eta(s):
        """
        biweight kernel on [-1, 1], integral 1
        """
        return np.where(np.abs(s) < 1, 15/16 * (1 - s**2)**2, 0.0)

    def bump(self, d):
        """
        psi_eps(d) = v/eps * eta(d/eps), d distance to the node
        """
        return self.v / self.eps * self.eta(d / self.eps)

    def f(self, state):
        """
        vector field of the ODE, state = (x, y, theta)
        """
        x, y, theta = state
        g = self.grid

        # transverse stabilizer: periodic restoring force towards the nearest grid line
        pull_x = g.dx / (2*np.pi) * np.sin(2*np.pi * (x - g.x_min) / g.dx)
        pull_y = g.dy / (2*np.pi) * np.sin(2*np.pi * (y - g.y_min) / g.dy)

        x_dot = self.v * np.cos(theta) - self.lam * np.sin(theta)**2 * pull_x
        y_dot = self.v * np.sin(theta) - self.lam * np.cos(theta)**2 * pull_y

        d = np.hypot(x - self.nodes_x, y - self.nodes_y)
        theta_dot = np.sum(self.bump(d) * self.deltas) - self.k * np.sin(4 * theta)

        return np.array([x_dot, y_dot, theta_dot])

    def df(self, state):
        """
        Jacobian of f, (3, 3)
        """
        x, y, theta = state
        g = self.grid

        pull_x = g.dx / (2*np.pi) * np.sin(2*np.pi * (x - g.x_min) / g.dx)
        pull_y = g.dy / (2*np.pi) * np.sin(2*np.pi * (y - g.y_min) / g.dy)
        dpull_x = np.cos(2*np.pi * (x - g.x_min) / g.dx)
        dpull_y = np.cos(2*np.pi * (y - g.y_min) / g.dy)

        # gradient of the bumps: d/dr psi_eps(|r - r_n|) = -15 v / (4 eps^3) (1 - s^2) (r - r_n), s = |r - r_n|/eps < 1
        rx, ry = x - self.nodes_x, y - self.nodes_y
        s = np.hypot(rx, ry) / self.eps
        c = np.where(s < 1, -15 * self.v / (4 * self.eps**3) * (1 - s**2), 0.0) * self.deltas

        J = np.zeros((3, 3))
        J[0, 0] = -self.lam * np.sin(theta)**2 * dpull_x
        J[0, 2] = -self.v * np.sin(theta) - self.lam * np.sin(2*theta) * pull_x
        J[1, 1] = -self.lam * np.cos(theta)**2 * dpull_y
        J[1, 2] = self.v * np.cos(theta) + self.lam * np.sin(2*theta) * pull_y
        J[2, 0] = np.sum(c * rx)
        J[2, 1] = np.sum(c * ry)
        J[2, 2] = -4 * self.k * np.cos(4*theta)
        return J

    def phi(self, state):
        """
        one explicit Euler step: phi_tau(x) = x + tau f(x)
        """
        return state + self.tau * self.f(state)

    def dphi(self, state):
        """
        Jacobian of phi_tau: I + tau Df(x)
        """
        return np.eye(3) + self.tau * self.df(state)

    def simulate(self, x0, y0, theta0, T, stop_node=None):
        """
        iterate phi_tau on [0, T] from (x0, y0, theta0).
        Start on a grid line with an axis heading, and not inside a turning bump.
        Stops early if the particle leaves the grid, or at the step closest to stop_node = (i, j) if given.
        """
        g = self.grid
        margin = 0.5 * min(g.dx, g.dy)
        n_steps = int(round(T / self.tau))
        if stop_node is not None:
            _xx, _yy = g.vals
            stop_r = np.array([_xx[stop_node], _yy[stop_node]])

        states = [np.array([x0, y0, theta0], dtype=float)]
        for n in range(n_steps):
            state = self.phi(states[-1])
            states.append(state)
            x, y, _ = state
            if min(x - g.x_min, g.x_max - x, y - g.y_min, g.y_max - y) < -margin:
                print(f"particle left the grid at t = {(n+1) * self.tau:.3f}")
                break
            if stop_node is not None:
                d_new = np.linalg.norm(states[-1][:2] - stop_r)
                d_old = np.linalg.norm(states[-2][:2] - stop_r)
                if d_old < margin and d_new > d_old: # just passed the node
                    states.pop()
                    break

        states = np.array(states)
        self.t = self.tau * np.arange(len(states))
        self.x, self.y, self.theta = states.T
        return states

    def plot_trajectory(self, ax=None):
        assert self.t is not None, "Run simulate first"
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        self.grid.plot_grid(ax=ax)
        ax.plot(self.x, self.y, color='blue', lw=1.2, zorder=3, label=f'trajectory, eps = {self.eps}')
        ax.plot(self.x[0], self.y[0], 'k>', zorder=5, label='start')
        ax.legend(loc='upper right')
        if show:
            plt.show()
        return ax


if __name__ == "__main__":
    # closed loop of length 20 on a 10x10 grid: 5 left turns and 1 right turn
    grid = Grid2D(x_min=0, x_max=10, y_min=0, y_max=10, Nx=10, Ny=10)
    L, R = np.pi/2, -np.pi/2
    turns = {(6, 1): L, (6, 3): L, (4, 3): R, (4, 6): L, (1, 6): L, (1, 1): L}

    particle = GridParticle(grid, turns, v=1.0, eps=0.05)
    particle.simulate(x0=2, y0=1, theta0=0, T=3*20)
    particle.plot_trajectory()
