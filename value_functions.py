import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

### my own imports 
# from grid import Grid1D

### classes 
class ValueFunction:
    """
    Value function in nD as defined by its covariance and state
    """ 

    def __init__(self, Pi, x_hat, residual=0):
        """
        Args: 
            - Pi : np.array (d,d)   the covariance matrix of the value function
            - x_hat : np.array (d,)  the state associated to the value function
            - res : float,  the residual in the formula, with default value 0 
        """
        self.dimension = x_hat.shape[0]
        assert Pi.shape == (self.dimension, self.dimension) 
        self.Pi = Pi 
        self.Hessian = np.linalg.inv(Pi)
        self.x_hat = x_hat
        self.residual = residual 

    def __call__(self, x):
        return 0.5 * np.dot(x - self.x_hat, self.Hessian @ (x - self.x_hat)) + self.residual 
    
    # def evaluate1D(self, grid:Grid1D):
    #     """
    #     evaluate the value function on a 1D grid 
    #     """
    #     assert self.dimension == 1, "evaluate is only implemented for 1D value functions"
    #     x_vals = grid.vals
    #     V_vals = np.array([self(x) for x in x_vals])
    #     return V_vals

    # def plot1D(self, grid:Grid1D, ax):
    #     assert self.dimension == 1, "plot is only implemented for 1D value functions"
    #     x_vals = grid.vals
    #     V_vals = np.array([self(x) for x in x_vals])
    #     ax.plot(x_vals, V_vals, label=f"Value function at x_hat={self.x_hat[0]:.2f}")
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$V(x)$')
    #     ax.legend()
    
    # def plot_surface3D(self, domain:Grid1D, functions_to_plot='minus', with_estimations=True):
    #     assert self.dimension == 1, "plot_surface3D is only implemented for 1D value functions"
    #     fig = plt.figure(figsize=(12, 6), dpi=150)
    #     ax = fig.add_subplot(111, projection='3d')

    #     functions = self.value_functions_minus if functions_to_plot == 'minus' else self.value_functions_plus

    #     # Build a 2D grid: rows = iterations, cols = spatial points
    #     time_vals = np.array([i * self.tau for i in range(len(functions))])
    #     x_vals = domain.vals

    #     Z = np.array(functions) 

    #     # Smooth interpolation over both axes
    #     spline = RectBivariateSpline(time_vals, x_vals, Z, kx=3, ky=3)

    #     t_fine = np.linspace(time_vals[0], time_vals[-1], 100)
    #     x_fine = np.linspace(x_vals[0], x_vals[-1], 100)
    #     T, X = np.meshgrid(t_fine, x_fine, indexing='ij')
    #     Z_fine = spline(t_fine, x_fine)

    #     surf = ax.plot_surface(
    #         T, X, Z_fine,
    #         cmap='Blues',
    #         linewidth=0,
    #         antialiased=True,
    #         alpha=0.9,
    #     )


    #     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

    #     if with_estimations and self.states_minus is not None:
    #         # states_minus is assumed to be a list/array of (time_index, x_value) pairs,
    #         # or just x_values aligned with time_vals iterations
    #         states = np.array(self.states_minus)

    #         if states.ndim == 1:
    #             # 1D array: one x-value per iteration
    #             t_states = time_vals[:len(states)]
    #             x_states = states
    #         elif states.shape[1] == 1:
    #             # 2D array with single column: still just x-values
    #             t_states = time_vals[:len(states)]
    #             x_states = states[:, 0]
    #         else:
    #             raise ValueError("states_minus should be 1D or 2D with single column")

    #         # Evaluate the surface height (z) at each (t, x) state point
    #         z_states = spline(t_states, x_states, grid=False)

    #         ax.plot(
    #             t_states, x_states, z_states,
    #             color='red', linewidth=1.5, zorder=5
    #         )
    #         ax.scatter(
    #             t_states, x_states, z_states,
    #             color='red', s=30, zorder=6, label='states : $\hat{x}_{n^-}$'
    #         )
    #         ax.legend()

    #     ax.set_xlabel('time')
    #     ax.set_ylabel(r'$x$')
    #     ax.set_zlabel(
    #         r'$V_{n^-}(x)$' if functions_to_plot == 'minus' else r'$V_{n^+}(x)$'
    #     )
    #     ax.set_title(
    #         r'Predicted value function' if functions_to_plot == 'minus' else r'Corrected value function'
    #     )
    #     plt.tight_layout()
    #     plt.show()

if __name__ == '__main__':
    v = ValueFunction(
        Pi = np.eye(2), 
        x_hat = np.array([2.,3.]) 
    )
    print(type(v)) 
    print(type(v) is ValueFunction)



