import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import RectBivariateSpline


### my own imports 
from value_functions import ValueFunction

### classes 
class Kalman:
    def __init__(self, x0, Pi0, Q, W, tau, A, H, observed_trajectory):
        """
        Kalman Filter implementation for linear state estimation 
        """
        self.dimension = x0.shape[0] # state dimension

        self.color = 'red' # for plots 

        self.x_minus = x0  # State estimate
        self.x_plus = None 

        self.states_minus = [x0] 
        self.states_plus = [] 


        self.Pi_minus = Pi0  # Estimate Pi
        self.Pi_plus = None 

        self.Pi_minus_list = [Pi0] 
        self.Pi_plus_list = [] 

        self.Q = Q   # observation noise covariance
        self.R = np.linalg.inv(Q) # precision of the observation noise
        self.W = W   # model noise covariance
        self.tau = tau  # Time step 

        ### matrix scalings #############
        self.Q_tau = self.tau**-1 * self.Q   # observation noise over one step
        self.W_tau = self.tau * self.W       # model noise over one step
        ################################# 

        self.A = A # update matrix 
        self.H = H # observation operator 
        self.observed_trajectory = observed_trajectory # Observed trajectory 

        ### value functions
        self.value_function_minus = ValueFunction(Pi0, x0) 
        self.value_functions_minus = [self.value_function_minus]

        self.value_function_plus = None
        self.value_functions_plus = []

        # keep track of the iteration 
        self.n = 0 

    def prediction(self):
        # update the state and covariance
        if self.x_plus is not None:
            self.x_minus = self.A @ self.x_plus 
            self.Pi_minus = self.A @ self.Pi_plus @ self.A.T + self.W_tau

            self.states_minus.append(self.x_minus) 
            self.Pi_minus_list.append(self.Pi_minus)

            self.value_function = ValueFunction(self.Pi_minus, self.x_minus)
            self.value_functions_minus.append(self.value_function)
        else: 
            raise IndexError("Run correction first") 
        
        self.n+= 1 

    def correction(self):
        assert self.n < len(self.observed_trajectory), f"Step {self.n} out of range"        
        y = self.observed_trajectory[self.n]
        z =  y - self.H @ self.x_minus  
        # compute the Kalman Gain
        to_inv = self.Q_tau + self.H @ self.Pi_minus @ self.H.T 
        G = self.Pi_minus @ self.H.T @ np.linalg.inv(to_inv) 

        # update the covariance 
        I = np.eye(self.x_minus.shape[0])   
        M = I - G @ self.H
        self.Pi_plus =  M @ self.Pi_minus @ M.T  + G @ self.Q_tau @ G.T
        # update the state
        z = np.atleast_1d(z) # safety check for matrix multiplication
        self.x_plus = self.x_minus + G @ z

        self.states_plus.append(self.x_plus) 
        self.Pi_plus_list.append(self.Pi_plus) 

        self.value_function_plus = ValueFunction(self.Pi_plus, self.x_plus)
        self.value_functions_plus.append(self.value_function_plus)

    def run_full(self, n_iterations):
        print(f"running {n_iterations} iterations of Kalman")
        print(f"tau = {self.tau}") 
        for _ in range(n_iterations):
            if _ == 0:
                self.correction()
            else:
                self.prediction()
                self.correction()

        self.has_run = True 
    
    def plot_estimations(self, ax, iterations, with_covariances=True):
        assert self.dimension == 1, "Plotting only implemented for d=1"

        states_to_plot = np.array(self.states_minus)
        ax.plot(
            iterations,
            states_to_plot[:,0],
            color='red', 
            marker='o',
            linestyle='dashed',
            label=r'Kalman estimation : $\hat{x}_{n^-}$'
        ) 
        if with_covariances:
            # corresponding to the level set of | V - min V| = 1  
            quantile = 2 ** 0.5 
            states = np.array(self.states_minus).flatten()
            cov_array = np.array(self.Pi_minus_list).flatten()

            lower_bounds = states - quantile*cov_array  
            upper_bounds = states + quantile*cov_array

            ax.fill_between(
                iterations,
                lower_bounds, 
                upper_bounds, 
                color=self.color,
                alpha=.1,
                label=r'84% confidence interval ($\sqrt{2}\sigma$)'
            ) 
        ax.set_xlabel('time')
        ax.legend(loc='upper left', prop={'size': 8})
    
class EKF:
    def __init__(self, x0, Pi0, Q, W, tau, Phi, dPhi, h, dh, observed_trajectory, initialize_with_plus):
        """
        Extended Kalman Filter implementation for nonlinear state estimation 
        """

        self.dimension = x0.shape[0] # state dimension

        self.color = 'blue' # for plots  

        if initialize_with_plus:
            self.x_plus = x0
            self.Pi_plus = Pi0

            self.states_minus = [] 
            self.states_plus = [x0] 
            self.Pi_minus_list = [] 
            self.Pi_plus_list = [Pi0] 

            self.x_minus = None 
            self.Pi_minus = None 

            ### value functions 
            self.value_function_plus = ValueFunction(Pi0, x0) # corresponds to W and not V ! 
            self.value_functions_plus = [self.value_function_plus]
            self.value_function_minus = None
            self.value_functions_minus = []

        else: 
            self.x_minus = x0
            self.Pi_minus = Pi0

            self.states_minus = [x0] 
            self.states_plus = [] 
            self.Pi_minus_list = [Pi0] 
            self.Pi_plus_list = [] 

            Sigma0 = np.linalg.inv(Pi0) # Estimate covariance
            self.covariances_minus = [Sigma0]
            self.covariances_plus = []

            self.x_plus = None 
            self.Pi_plus = None 

            ### value functions 
            self.value_function_minus = ValueFunction(Pi0, x0)
            self.value_functions_minus = [self.value_function_minus]
            self.value_function_plus = None
            self.value_functions_plus = []
        
        self.Q = Q   # observation noise covariance
        self.R = np.linalg.inv(Q) # precision of the observation noise
        self.W = W   # model noise covariance
        self.tau = tau  # Time step 

        ### matrix scalings ###############
        self.Q_tau = self.tau**-1 * self.Q   # observation noise over one step
        self.W_tau = self.tau * self.W       # model noise over one step
        ###################################

        self.Phi = Phi  # State transition function
        self.dPhi = dPhi # Differential of the state transition function
        self.h = h # Observation function
        self.dh = dh # Differential of the Observation function
        self.observed_trajectory = observed_trajectory # Observed trajectory

        # keep track of the iteration 
        self.n = 0 

    def prediction(self):
        # Predict the state and covariance
        if self.x_plus is not None:
            A = self.dPhi(self.x_plus)
            A = np.atleast_2d(A) # safety check for d=1 
            self.x_minus = self.Phi(self.x_plus) 
            self.Pi_minus = A @ self.Pi_plus @ A.T + self.W_tau  # Update estimate covariance

            self.states_minus.append(self.x_minus) 
            self.Pi_minus_list.append(self.Pi_minus)

            self.value_function_minus = ValueFunction(self.Pi_minus, self.x_minus)
            self.value_functions_minus.append(self.value_function_minus)
        else: 
            raise IndexError("Run correction first") 

        self.n+= 1 

    def correction(self):
        # Compute the Kalman Gain
        assert self.n < len(self.observed_trajectory), f"Step {self.n} out of range"        
        y = self.observed_trajectory[self.n]
        z =  y - self.h(self.x_minus)  # Measurement residual
        H = self.dh(self.x_minus)
        to_inv = self.Q_tau + H @ self.Pi_minus @ H.T 
        G = self.Pi_minus @ H.T @ np.linalg.inv(to_inv)  # Kalman Gain

        # update the covariance 
        I = np.eye(self.x_minus.shape[0])   
        M = I - G @ H
        self.Pi_plus =  M @ self.Pi_minus @ M.T + G @ self.Q_tau @ G.T
        # update the state 
        z = np.atleast_1d(z) # safety check for matrix multiplication 
        self.x_plus = self.x_minus + G @ z

        self.states_plus.append(self.x_plus) 
        self.Pi_plus_list.append(self.Pi_plus) 

        self.value_function_plus = ValueFunction(self.Pi_plus, self.x_plus)
        self.value_functions_plus.append(self.value_function_plus)

    def run_full(self, n_iterations):
        print(f"running {n_iterations} iterations of EKF")
        print(f"tau = {self.tau}") 
        for _ in range(n_iterations):
            if _ == 0:
                self.correction()
            else:
                self.prediction()
                self.correction()

        self.has_run = True 
        

if __name__ == "__main__":

    d = 1

    ### EKF 
    x0 = np.zeros(d)  # Initial state estimate
    Pi0 = np.eye(d)  # Initial estimate covariance
    Q = np.eye(d) * 0.1  # Process noise covariance
    R = np.eye(d) * 0.1  # Measurement noise covariance
    tau = 0.1
    Phi = lambda x: np.eye(x.shape[0]) @ x
    dPhi = lambda x: np.eye(x.shape[0]) 
    h = lambda x: x
    dh = lambda x: np.eye(x.shape[0]) 
    
    T = 10.
    n_iterations = round(T/tau) 

    target = np.random.randint(10)
    print(f"target is set to {target}")  
    observed_trajectory = [target*np.ones(d)]*n_iterations

    observer = EKF(
        x0,
        Pi0,
        Q,
        R,
        tau,
        Phi,
        dPhi,
        h,
        dh,
        observed_trajectory, 
        initialize_with_plus=False
    )

    observer.run_full(n_iterations)
    assert observer.has_run 
    print("Final estimation : ", observer.states_minus[-1])

    ### Kalman 
    A = np.eye(d) 
    H = np.eye(d) 

    kalman_filter = Kalman(
        x0, 
        Pi0, 
        Q, 
        R, 
        tau,
        A, 
        H, 
        observed_trajectory
    )

    kalman_filter.run_full(n_iterations) 
    assert kalman_filter.has_run 
    print("Final Kalman estimation : ", kalman_filter.states_minus[-1])

    ### Plotting 
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    iterations = np.linspace(0, observer.tau*observer.n, observer.n) 

    ax.plot(
        iterations,
        observed_trajectory,
        color='green', 
        marker='x',
        label=r'target : $y$'
    ) 
    states_to_plot = np.array(observer.states_minus)
    ax.plot(
        iterations,
        states_to_plot[:,0],
        color='blue', 
        marker='o',
        linestyle='dashed',
        label=r'EKF estimation : $\hat{x}_{n^-}$'
    ) 

    Kalman_states = np.array(kalman_filter.states_minus)
    ax.plot(    
        iterations,
        Kalman_states[:,0],
        color='red', 
        marker='o',
        linestyle='dashed',
        label=r'kalman estimation : $\hat{x}_{n^-}$'
    ) 
    ax.set_xlabel('time')
    ax.legend() 

    plt.show() 
