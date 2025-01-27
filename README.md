### Nikolaevsky Equation Solver with Lyapunov Characteristic Exponent (LCE) Calculation

This Python implementation provides a solver for the Nikolaevsky Equation (NE) using the **Exponential Time Differencing Runge-Kutta (ETDRK4)** method. It also supports the calculation of **Lyapunov Characteristic Exponents (LCEs)** to analyze the chaotic behavior of dynamical systems. The implementation is optimized for performance using **PyTorch**.

---

#### Key Features:
- **Nikolaevsky Equation Solver**: Simulates the evolution of the Nikolaevsky equation on a spatial domain.
- **ETDRK4 Method**: A robust and efficient time-stepping method for stiff partial differential equations.
- **Lyapunov Exponent Calculation**: Computes the Lyapunov Characteristic Exponents to determine the stability and chaotic nature of the system.
- **GPU Support**: Accelerated computation using PyTorch's `cuda` device.

---

#### Class: `NE`
This class encapsulates the dynamics of the Nikolaevsky Equation (NE) with the ability to compute LCEs. 

##### **Initialization**
```python
NE(L, N, h, u_0, r, v, precompute_step=None, device='cpu')
```
**Parameters**:
- `L (float)`: Size of the spatial domain.
- `N (int)`: Number of degrees of freedom (spatial discretization points).
- `h (float)`: Time step size for the simulation.
- `u_0 (numpy.ndarray)`: Initial conditions for the dynamics with shape `(BATCH_SIZE, N)`.
- `r (numpy.ndarray)`: Control parameter with shape `(BATCH_SIZE,)`.
- `v (float)`: Damping constant.
- `precompute_step (int, optional)`: Number of precomputed steps before starting the dynamics.
- `device (str, optional)`: Compute device (`'cpu'` or `'cuda'`).

---

##### **Key Methods**
1. **`forward(n_steps, keep_traj=True)`**  
   Simulates the system forward for `n_steps` using the ETDRK4 method.
   - `n_steps (int)`: Number of time steps to simulate.
   - `keep_traj (bool)`: If `True`, returns the trajectory.
   - **Returns**: `(torch.Tensor)` Trajectory of shape `(BATCH_SIZE, n_steps + 1, N)`.

2. **`__call__()`**  
   Advances the system by one time step using ETDRK4. Automatically updates the state of the system.

3. **`LCE(p, n_forward, n_compute, qr_mode='reduced')`**  
   Computes the Lyapunov Characteristic Exponents (LCEs).
   - `p (int)`: Number of LCEs to compute.
   - `n_forward (int)`: Number of steps to evolve the system before starting LCE calculation.
   - `n_compute (int)`: Number of steps for LCE computation.
   - `qr_mode (str, optional)`: QR decomposition mode (`'reduced'` or `'complete'`).
   - **Returns**:  
     - `LCE (torch.Tensor)`: Computed Lyapunov exponents of shape `(BATCH_SIZE, p)`.  
     - `history (torch.Tensor)`: Evolution of LCE values over the computation.

4. **`next_LTM(W)`**  
   Computes the state of deviation vectors (Linear Tangent Map) after one time step.
   - `W (torch.Tensor)`: Deviation vectors.
   - **Returns**: Updated deviation vectors.

5. **`_one_step(u)`**  
   Evolves the system one time step from the provided condition.
   - `u (torch.Tensor)`: Initial condition of shape `(BATCH_SIZE, N)`.
   - **Returns**: Evolved condition.

---

#### How to Use
1. **Initialize the Solver**:
   ```python
   import numpy as np
   from NikolaevskyDynamics.Nikolaevsky import NE
   
   L = 50*np.pi  # Domain size
   N = 512    # Spatial discretization points
   h = 0.05   # Time step size
   r = np.linspace(0.0, 0.5, 100)    # Control parameter
   u_0 = np.random.randn(N_DOF)
   u_0 = np.tile(u_0, (len(r),1)) # Initial condition for 10 trajectories
   v = 0.15                       # Damping constant

   solver = NE(L=L, N=N_DOF, h=h, u_0=u_0, r=r, v=v, precompute_step=round(1000 / DT), device='cuda')
   ```

2. **Run the Dynamics**:
   ```python
   trajectory = solver.forward(n_steps=round(2000 / DT), keep_traj=True)
   ```

3. **Compute Lyapunov Exponents**:
   ```python
   p = 10  # Number of exponents
   LCE, history = solver.LCE(p=p, n_forward=round(500 / DT), n_compute=round(1500 / DT))
   print("Lyapunov Exponents:", LCE)
   ```

4. **Access Single-Step Evolution**:
   ```python
   evolved_state = solver._one_step(u_0)
   ```

---

#### References
This implementation is inspired by the following works:
- [Nikolaevsky Equation Dynamics](https://arxiv.org/abs/1002.3490)
- [Linear Tangent Map](https://cns.gatech.edu/~predrag/papers/SCD07.pdf)
- [Lyapunov Characteristic Exponents](https://math.iisc.ac.in/~rangaraj/wp-content/uploads/2020/07/jiisc_lyap.pdf)
- [ETDRK4 Method](https://www.sciencedirect.com/science/article/abs/pii/S0021999102969950)
- [Kuramoto-Sivashinsky Solver](https://github.com/ThomasSavary08/Kuramoto-Sivashinsky-ETDRK4)

For further details, explore the repository and the cited works.