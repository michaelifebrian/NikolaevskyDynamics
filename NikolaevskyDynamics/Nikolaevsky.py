import numpy as np
from tqdm import tqdm
import torch
torch.set_default_dtype(torch.float64)

"""
References:
    https://www.sciencedirect.com/science/article/abs/pii/S0021999102969950
    https://cns.gatech.edu/~predrag/papers/SCD07.pdf
    https://arxiv.org/pdf/1902.09651
    https://arxiv.org/pdf/1105.5228
    https://github.com/ThomasSavary08/Lyapynov
    https://github.com/ThomasSavary08/Kuramoto-Sivashinsky-ETDRK4
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fft_and_spectral_methods/ks_solver_etd_and_etdrk2_in_jax.ipynb
    https://math.iisc.ac.in/~rangaraj/wp-content/uploads/2020/07/jiisc_lyap.pdf
"""

class NE:
    def __init__(self, L, N, h, u_0, r, v, precompute_step = None, device = 'cpu', threshold = 1e-3):
        """
        Initialize Nikolaevsky equation dynamics.
            Parameters:
                L (float): Size of the spatial domain.
                N (int): Number of degrees of freedom (spatial discretization points).
                h (float): Time step size for the simulation.
                u_0 (numpy.ndarray): Initial conditions for the dynamics with shape (BATCH_SIZE, N).
                r (numpy.ndarray): Control parameter with shape (BATCH_SIZE,).
                v (float): Damping constant.
                precompute_step (int, optional): Number of precomputed steps before starting the dynamics.
                device (str, optional): Compute device ('cpu' or 'cuda').
                r (float, optional): Threshold limit as c value approaches zero.
        """
        self.L = L
        self.N = N
        self.h = h
        self.t = 0
        self.device = device
        u_0 = torch.tensor(u_0, device=device)
        self.u_hat = torch.fft.rfft(u_0)
        print(f"Number of dynamics calculated: {u_0.shape[0]}")

        # Wavenumbers & derivative operator for FFT
        self.wavenumbers = torch.tensor(np.fft.rfftfreq(n=self.N, d=L / (self.N * 2 * np.pi)), device=self.device)
        self.wavenumbers = self.wavenumbers.repeat(u_0.shape[0],1)
        self.diagonal_wavenumbers = torch.diag_embed(self.wavenumbers)
        self.derivative_operator = 1j * self.wavenumbers

        # Filter for antialiasing
        self.filter = (self.wavenumbers < 2 / 3 * torch.max(self.wavenumbers))

        # Nonlinear function for NE system
        self.f = lambda u: (-0.5 * torch.fft.rfft(u ** 2) * self.derivative_operator) * self.filter

        # Nonlinear function for fundamental matrix of NE system
        self.fn = lambda w: (-1.j * self.diagonal_wavenumbers @ torch.fft.rfft(
            (torch.diag_embed(torch.fft.irfft(self.u_hat, n=self.N)) @ torch.fft.irfft(w, n=self.N, axis=1)), axis=1)) * self.filter[:, :, None]

        # Linear constant for both NE and fundamental matrix
        r = torch.tensor(r, device=self.device).repeat(self.derivative_operator.shape[1],1).permute(1,0)
        v = torch.tensor(v, device=self.device).repeat(self.derivative_operator.shape[1],1).permute(1,0)
        self.c = -(self.derivative_operator**2) * (r-(1+self.derivative_operator**2)**2) - v
        print("using LSA damping")

        # Constant for ETDRK4
        self.exp_term_half = torch.exp((self.c * self.h) / 2)
        self.exp_term = torch.exp(self.c * self.h)
        self.diagonal_exp_term_half = torch.diag_embed(self.exp_term_half)
        self.diagonal_exp_term = torch.diag_embed(self.exp_term)
        self.threshold = threshold
        self.k = torch.where(
            abs(self.c) <= self.threshold,
            h / 2,
            (self.exp_term_half - 1.0) / self.c,
        )
        self.f1 = torch.where(
            abs(self.c) <= self.threshold,
            h / 6,
            (- 4 - self.h * self.c + self.exp_term * (4 - 3 * self.h * self.c + (self.h ** 2 * self.c ** 2))) / (
                        (self.c ** 3) * (self.h ** 2)),
        )
        self.f2 = torch.where(
            abs(self.c) <= self.threshold,
            h / 3,
            (2 * (2 + self.h * self.c + self.exp_term * (-2 + self.h * self.c))) / ((self.c ** 3) * (self.h ** 2)),
        )
        self.f3 = torch.where(
            abs(self.c) <= self.threshold,
            h / 6,
            (- 4 - 3 * self.h * self.c - (self.h ** 2 * self.c ** 2) + self.exp_term * (4 - self.h * self.c)) / (
                        (self.c ** 3) * (self.h ** 2)),
        )
        self.diagonal_k = torch.diag_embed(self.k)
        self.diagonal_f1 = torch.diag_embed(self.f1)
        self.diagonal_f2 = torch.diag_embed(self.f2)
        self.diagonal_f3 = torch.diag_embed(self.f3)

        if precompute_step:
            print(f"Stepping until t={precompute_step*h}")
            self.forward(precompute_step, False)

    def next_LTM(self, W):
        """
        Computes the state of deviation vectors (Linear Tangent Map) after one time step.
            Parameters:
                W (torch.Tensor): Array of deviations vectors.
            Returns:
                res (torch.Tensor): Array of deviations vectors at next time step.
        """
        W1 = torch.fft.rfft(W, axis=1)

        N_W1 = self.fn(W1)
        W2 = self.diagonal_exp_term_half @ W1 + self.diagonal_k @ N_W1

        N_W2 = self.fn(W2)
        W3 = self.diagonal_exp_term_half @ W1 + self.diagonal_k @ N_W2

        N_W3 = self.fn(W3)
        W4 = self.diagonal_exp_term_half @ W2 + self.diagonal_k @ (2 * N_W3 - N_W1)

        N_W4 = self.fn(W4)
        res = self.diagonal_exp_term @ W1 + self.diagonal_f1 @ N_W1 + self.diagonal_f2 @ (
                    N_W2 + N_W3) + self.diagonal_f3 @ N_W4
        res = torch.fft.irfft(res, n=self.N, axis=1)
        return res

    def __call__(self):
        """
        Compute the state of the system after one time step with ETDRK4.
        """
        u = torch.fft.irfft(self.u_hat, n=self.N)

        f_un_hat = self.f(u)
        an_hat = self.exp_term_half * self.u_hat + self.k * f_un_hat
        an = torch.fft.irfft(an_hat, n=self.N)

        f_an_hat = self.f(an)
        bn_hat = self.exp_term_half * self.u_hat + self.k * f_an_hat
        bn = torch.fft.irfft(bn_hat, n=self.N)

        f_bn_hat = self.f(bn)
        cn_hat = self.exp_term_half * an_hat + self.k * (2 * f_bn_hat - f_un_hat)
        cn = torch.fft.irfft(cn_hat, n=self.N)

        f_cn_hat = self.f(cn)
        self.u_hat = self.u_hat * self.exp_term + f_un_hat * self.f1 + (
                    f_an_hat + f_bn_hat) * self.f2 + f_cn_hat * self.f3
        self.t += self.h

    def forward(self, n_steps, keep_traj=True):
        """
        Forward the system for n_steps with ETDRK4 method.
            Parameters:
                n_steps (int): Number of simulation steps to take.
                keep_traj (bool): Return or not the system trajectory.
            Returns:
                traj (torch.Tensor): Trajectory of the system with shape of (BATCH_SIZE, n_steps + 1, N) if keep_traj.
        """
        if keep_traj:
            traj = torch.zeros((self.u_hat.shape[0], n_steps + 1, self.N), device=self.device)
            traj[:, 0, :] = torch.fft.irfft(self.u_hat, n=self.N)
            for i in tqdm(range(1, n_steps + 1)):
                self()
                traj[:, i, :] = torch.fft.irfft(self.u_hat, n=self.N)
            return traj
        else:
            for _ in tqdm(range(n_steps)):
                self()

    def LCE(self, p, n_forward, n_compute, qr_mode='reduced', keep_his=False):
        """
        Compute LCE (Lyapunov Characteristic Exponent).
            Parameters:
                p (int): Number of LCE to compute.
                n_forward (int): Number of time steps before starting the computation of LCE.
                n_compute (int): Number of steps to compute the LCE.
                qr_mode (str, optional): QR decomposition mode ('reduced' or 'complete').
            Returns:
                LCE (torch.Tensor): Computed Lyapunov exponents of shape (BATCH_SIZE, p).
                history (torch.Tensor): Evolution of LCE during the computation.
        """
        # Forward the system before the computation of LCE
        self.forward(n_forward, False)

        # Computation of LCE
        W = torch.eye(self.N, device=self.device).unsqueeze(0).repeat(self.u_hat.shape[0],1,1)[:, :, :p]
        LCE = torch.zeros((self.u_hat.shape[0], p), device=self.device)
        if keep_his:
            history = torch.zeros((self.u_hat.shape[0], n_compute, p), device=self.device)
        for i in tqdm(range(1, n_compute + 1)):
            W = self.next_LTM(W)
            self()
            if keep_his:
                Q, R = torch.linalg.qr(W, mode=qr_mode)
                for j in range(p):
                    history[:, i - 1, j] = torch.log(torch.abs(R[:, j, j])) / (i * self.h)
        Q, R = torch.linalg.qr(W, mode=qr_mode)
        for j in range(p):
            LCE[:, j] = torch.log(torch.abs(R[:, j, j]))
        LCE = LCE / (n_compute * self.h)
        if keep_his:
            return LCE, history
        else:
            return LCE

    def _one_step(self, u):
        """
        Compute one step from given condition.
            Parameters:
                u (torch.Tensor): The initial condition with shape of (BATCH_SIZE, N).
            Returns:
                u (torch.Tensor): The evolved condition after 1 step.
        """
        u_hat = torch.fft.rfft(u)

        f_un_hat = self.f(u)
        an_hat = self.exp_term_half * u_hat + self.k * f_un_hat
        an = torch.fft.irfft(an_hat, n=self.N)

        f_an_hat = self.f(an)
        bn_hat = self.exp_term_half * u_hat + self.k * f_an_hat
        bn = torch.fft.irfft(bn_hat, n=self.N)

        f_bn_hat = self.f(bn)
        cn_hat = self.exp_term_half * an_hat + self.k * (2 * f_bn_hat - f_un_hat)
        cn = torch.fft.irfft(cn_hat, n=self.N)

        f_cn_hat = self.f(cn)
        u_hat = u_hat * self.exp_term + f_un_hat * self.f1 + (f_an_hat + f_bn_hat) * self.f2 + f_cn_hat * self.f3
        u = torch.fft.irfft(u_hat, n=self.N)
        return u
