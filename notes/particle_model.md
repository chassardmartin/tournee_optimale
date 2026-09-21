# Particle on a grid: regularized model

Implemented in `particle_dynamics.py` (`GridParticle`). Full discussion: `particle_dynamics.md`.

## State and equations

State $(x, y, \theta)$, speed $v$, grid spacing $(\Delta x, \Delta y)$, origin $(x_{min}, y_{min})$:

$$
\begin{aligned}
\dot x &= v\cos\theta - \lambda \sin^2\theta \,\frac{\Delta x}{2\pi}\sin\frac{2\pi(x - x_{min})}{\Delta x} \\
\dot y &= v\sin\theta - \lambda \cos^2\theta \,\frac{\Delta y}{2\pi}\sin\frac{2\pi(y - y_{min})}{\Delta y} \\
\dot\theta &= \sum_{n} \psi_\varepsilon(|r - r_n|)\,\Delta_n \;-\; k\sin 4\theta
\end{aligned}
$$

Discretized with explicit Euler: $\varphi^\tau(\mathrm x) = \mathrm x + \tau f(\mathrm x)$, $D\varphi^\tau = I + \tau\, Df$, with $\tau = \varepsilon/(10v)$ by default.

## Terms

- **Turns**: $\Delta_n = \pm\pi/2$ (left/right) at turning node $r_n$.
- **Bump**: $\psi_\varepsilon(d) = \frac{v}{\varepsilon}\,\eta(d/\varepsilon)$, $\eta(s) = \frac{15}{16}(1-s^2)^2$ on $|s|<1$, so crossing a node turns the heading by exactly $\Delta_n$.
- **Heading snap**: $-k\sin 4\theta$, stable at $\theta = m\pi/2$. Needs $k = \kappa v/\varepsilon$, with $\kappa < \eta(0)\,|\Delta_n| \approx 1.47$ (default $\kappa = 0.4$).
- **Transverse stabilizer** ($\lambda$, default 8): pulls the particle onto the nearest grid line, perpendicular to its motion only.

## Constraints

- $0 < \varepsilon < \min(\Delta x, \Delta y)/2$ (bumps must not overlap).
- As $\varepsilon \to 0$: sharp corners, $\theta(t_k^+) = \theta(t_k^-) + \Delta_k$. Distance to the grid is $O(\varepsilon^{0.7})$ and lap-time error is $O(\varepsilon^2)$.
