import numpy as np 
import matplotlib.pyplot as plt 

### classes 


class Grid2D:
    """
    2D uniform grid
    """

    def __init__(self, x_min, x_max, y_min, y_max, Nx, Ny): 
        self.x_min = x_min 
        self.x_max = x_max 
        self.y_min = y_min  
        self.y_max = y_max
        self.xlength = x_max - x_min 
        self.ylength = y_max - y_min
        self.Nx = Nx
        self.Ny = Ny 
        self.dx = self.xlength / Nx
        self.dy = self.ylength / Ny 
        self.vals = np.meshgrid(
            np.linspace(x_min, x_max, Nx+1), # +1 for boundary conditions
            np.linspace(y_min, y_max, Ny+1), 
            indexing='ij' # careful with this, row-major order 
        )
    
    def plot_grid(self, show_lines=True, ax=None, node_size=10, node_color='red', line_color='gray', line_width=0.5):
        """
        plot the grid nodes (and lines). If ax is given, draw on it and don't call plt.show()
        """
        _xx, _yy = self.vals
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        if show_lines:
            # draw mesh lines through the nodes themselves (ax.grid() follows ticks, not nodes)
            ax.plot(_xx, _yy, color=line_color, lw=line_width, zorder=1)     # lines of constant y
            ax.plot(_xx.T, _yy.T, color=line_color, lw=line_width, zorder=1) # lines of constant x
        ax.scatter(_xx, _yy, s=node_size, color=node_color, marker='o', zorder=2)
        ax.set_aspect('equal')
        if show:
            plt.show()
        return ax 