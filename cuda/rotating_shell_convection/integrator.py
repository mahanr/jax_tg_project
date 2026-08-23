"""IMEX time integrator for W, Z, and temperature spectral coefficients."""
import cupy as cp


class ImexEulerIntegrator:
    """
    First-order IMEX.

    Toroidal Z and temperature: (I - dt κ Δ) with Dirichlet rows.
    Poloidal W: MagIC double-curl viscous operator (Δ - dt Ek Δ²) with
    no-slip W = ∂W/∂r = 0 at r=η and r=1.
    """

    def __init__(self, operators, dt):
        self.operators = operators
        self.dt = float(dt)
        self._lap_cache = {}

    def _laplacian_matrix(self, lm_index):
        if lm_index in self._lap_cache:
            return self._lap_cache[lm_index]
        l = float(self.operators.geometry.l_values[lm_index])
        lap = self.operators.spherical_laplacian_matrix(l)
        self._lap_cache[lm_index] = lap
        return lap

    def _solve_toroidal(self, rhs, coeff):
        n_lm = rhs.shape[1]
        nr = rhs.shape[0]
        out = cp.zeros_like(rhs)
        eye = cp.eye(nr, dtype=cp.float64)
        d_r = self.operators.radial.d_r
        for lm in range(n_lm):
            l = float(self.operators.l_values[lm])
            if l < 1.0:
                continue
            lap = self._laplacian_matrix(lm)
            system = eye - self.dt * coeff * lap
            system[0, :] = 0.0
            system[0, 0] = 1.0
            system[-1, :] = 0.0
            system[-1, -1] = 1.0
            rhs_lm = rhs[:, lm].copy()
            rhs_lm[0] = 0.0
            rhs_lm[-1] = 0.0
            out[:, lm] = cp.linalg.solve(system, rhs_lm)
        return out

    def _solve_poloidal(self, w_old, rhs_force, coeff):
        n_lm = w_old.shape[1]
        nr = w_old.shape[0]
        out = cp.zeros_like(w_old)
        d_r = self.operators.radial.d_r
        for lm in range(n_lm):
            l = float(self.operators.l_values[lm])
            if l < 1.0:
                continue
            lap = self._laplacian_matrix(lm)
            lap2 = lap @ lap
            system = lap - self.dt * coeff * lap2
            rhs_lm = lap @ w_old[:, lm] + self.dt * rhs_force[:, lm]
            system[0, :] = 0.0
            system[0, 0] = 1.0
            system[1, :] = d_r[0]
            system[-1, :] = 0.0
            system[-1, -1] = 1.0
            system[-2, :] = d_r[-1]
            rhs_lm[0] = 0.0
            rhs_lm[1] = 0.0
            rhs_lm[-1] = 0.0
            rhs_lm[-2] = 0.0
            out[:, lm] = cp.linalg.solve(system, rhs_lm)
        return out

    def _solve_temperature(self, rhs, coeff):
        n_lm = rhs.shape[1]
        nr = rhs.shape[0]
        out = cp.zeros_like(rhs)
        eye = cp.eye(nr, dtype=cp.float64)
        for lm in range(n_lm):
            l = float(self.operators.l_values[lm])
            lap = self._laplacian_matrix(lm)
            system = eye - self.dt * coeff * lap
            system[0, :] = 0.0
            system[0, 0] = 1.0
            system[-1, :] = 0.0
            system[-1, -1] = 1.0
            rhs_lm = rhs[:, lm].copy()
            rhs_lm[0] = 0.0
            y00 = 1.0 / (4.0 * 3.141592653589793) ** 0.5
            rhs_lm[-1] = (1.0 / y00) if l < 0.5 else 0.0
            out[:, lm] = cp.linalg.solve(system, rhs_lm)
        return out

    def step(self, w_coeffs, z_coeffs, theta_coeffs):
        ew, ez, etheta = self.operators.rhs_explicit(w_coeffs, z_coeffs, theta_coeffs)
        config = self.operators.config

        z_rhs = z_coeffs + self.dt * ez
        theta_rhs = theta_coeffs + self.dt * etheta

        w_new = self._solve_poloidal(w_coeffs, ew, config.viscous_coeff)
        z_new = self._solve_toroidal(z_rhs, config.viscous_coeff)
        theta_new = self._solve_temperature(
            theta_rhs, config.thermal_diffusion_coeff
        )
        return w_new, z_new, theta_new
