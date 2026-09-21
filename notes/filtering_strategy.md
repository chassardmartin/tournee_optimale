We want to implement an Extended Kalman Fitler to filter out the nonlinear dynamics described in the note particle_dynamics.md and implemented inside particle_dynamics.py. 

The modeling equations are as follows : write $\mathrm{x} = (x,y,\theta)$ for the full state and 
$\mathrm{r} = (x,y)$ for the position coordinate. 

The nonlinear dynamics write 
$$ 
\dot{\mathrm{x}} = f(\mathrm{x}) + \nu 
$$  
where $\nu$ is a small noise term. A discretization method (that could be for instance Explicit Euler, or more involved explicit methods) writes 
$$ 
    \mathrm{x}_{n+1} = \varphi^\tau(\mathrm{x}_n) + \nu_{n+1}, 
$$
where $\varphi^\tau$ encodes the discretization scheme parametrized by a time discretization parameter $\tau$. 

We have access to noisy observations on the dynamics via 
$$
    y_n = h(\mathrm{x}_n) + \eta_{n}. 
$$

The goal of the filtering strategy is to build an estimator $\hat{\mathrm{x}}_n$ only from observations $y_n$ that converges towards the true dynamics. For linear dynamics and linear $h$ this is the classical Kalman Filter. 

The Extended Kalman Filter works by locally linearizing around the local estimation by casting 
$$ 
    A := \mathrm{d}_x \varphi^\tau(\hat{\mathrm{x}}_n), \quad 
    H := \mathrm{d}_x h(\hat{\mathrm{x}}_n). 
$$

In the source file kalman_filters.py you can find an implementation of Kalman filter and EKF. 



