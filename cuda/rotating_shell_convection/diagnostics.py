"""Diagnostics: Nusselt number, kinetic energy, and field monitors."""
import cupy as cp
import numpy as np


def _clenshaw_curtis_weights(n):
    n_int = n - 1
    weights = np.zeros(n, dtype=np.float64)
    theta = np.pi * np.arange(n) / n_int
    for k in range(n):
        s = 0.0
        n_half = n_int // 2
        for j in range(1, n_half + 1):
            b = 1.0 if (j == n_half and n_int % 2 == 0) else 2.0
            s += -b * np.cos(2 * j * theta[k]) / (4 * j * j - 1.0)
        weights[k] = 2.0 / n_int * (1.0 + s)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


class ShellDiagnostics:
    def __init__(self, operators):
        self.operators = operators
        self.times = []
        self.kinetic_energies = []
        self.kinetic_energies = self.kinetic_energies
        self.nusselt_inner = []
        self.nusselt_inner = self.nusselt_inner
        self.nusselt_outer = []
        self.theta_rms = []
        self.max_u = []
        geometry = operators.geometry
        cc = _clenshaw_curtis_weights(geometry.nr)
        self._radial_weights = cp.asarray(cc) * geometry.radial.dr_dxi
        self._theta_weights = geometry.sht.weights
        self._phi_weight = 2.0 * np.pi / geometry.n_phi

    @staticmethod
    def _host(value):
        return float(cp.asnumpy(value).real)

    def _angular_mean(self, field_shell):
        weighted = self._theta_weights[:, None] * field_shell * self._phi_weight
        return cp.sum(weighted) / (4.0 * np.pi)

    def kinetic_energy(self, w_coeffs, z_coeffs):
        u = self.operators.pt.velocity_from_wz_coeffs(w_coeffs, z_coeffs)
        u_sq = cp.real(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        r = self.operators.r
        shells = cp.zeros(u_sq.shape[0], dtype=cp.float64)
        for i in range(u_sq.shape[0]):
            shells[i] = self._angular_mean(u_sq[i])
        volume = 4.0 * np.pi / 3.0 * (1.0 - self.operators.geometry.eta ** 3)
        integral = cp.sum(self._radial_weights * r * r * shells)
        return 0.5 * integral / volume

    def nusselt_numbers(self, theta_coeffs):
        geometry = self.operators.geometry
        eta = geometry.eta
        r = geometry.r
        theta_grid = self.operators.spectral_to_grid(theta_coeffs)
        dr_theta = geometry.radial.d_r @ theta_grid.reshape(geometry.nr, -1)
        dr_theta = dr_theta.reshape(theta_grid.shape)
        dtheta_cond = (eta / (1.0 - eta)) * (1.0 / (r * r))
        flux_outer = self._angular_mean(-dr_theta[0])
        flux_inner = self._angular_mean(-dr_theta[-1])
        nu_inner = float(cp.asnumpy(flux_inner).real) / float(cp.asnumpy(dtheta_cond[-1]).real)
        nu_outer = float(cp.asnumpy(flux_outer).real) / float(cp.asnumpy(dtheta_cond[0]).real)
        return nu_inner, nu_outer

    def record(self, time_value, w_coeffs, z_coeffs, theta_coeffs):
        ke = self.kinetic_energy(w_coeffs, z_coeffs)
        nu_i, nu_o = self.nusselt_numbers(theta_coeffs)
        theta_grid = self.operators.spectral_to_grid(theta_coeffs)
        theta_rms = cp.sqrt(cp.mean(theta_grid * theta_grid))
        u = self.operators.pt.velocity_from_wz_coeffs(w_coeffs, z_coeffs)
        max_u = cp.max(cp.sqrt(cp.sum(cp.real(u) ** 2, axis=0)))

        self.times.append(time_value)
        self.kinetic_energies.append(self._host(ke))
        self.kinetic_energies = self.kinetic_energies
        self.nusselt_inner.append(nu_i)
        self.nusselt_inner = self.nusselt_inner
        self.nusselt_outer.append(nu_o)
        self.theta_rms.append(self._host(theta_rms))
        self.max_u.append(self._host(max_u))
        return ke, nu_i
