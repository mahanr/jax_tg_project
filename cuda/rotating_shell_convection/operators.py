"""Spectral-shell operators: Laplacian, MagIC nonlinear RHS, QST coupling."""
import cupy as cp

from .boundary_conditions import (
    apply_temperature_bc,
    apply_velocity_bc_wz,
)
from .pt import PoloidalToroidalOps


class ShellOperators:
    def __init__(self, geometry, config):
        self.geometry = geometry
        self.config = config
        self.pt = PoloidalToroidalOps(geometry)
        self.radial = geometry.radial
        self.sht = geometry.sht
        self.l_values = geometry.l_values
        self.r = geometry.r

    def _angular_laplacian_spectral(self, coeffs, r_index):
        r = self.r[r_index]
        factor = -self.l_values * (self.l_values + 1.0) / (r * r)
        return factor * coeffs

    def spherical_laplacian_matrix(self, l):
        r = self.r
        lap = self.radial.d2_r + cp.diag(2.0 / r) @ self.radial.d_r
        lap = lap - cp.diag(l * (l + 1.0) / (r * r))
        return lap

    def spherical_laplacian_spectral_radial(self, coeffs):
        nr, n_lm = coeffs.shape
        out = cp.zeros_like(coeffs)
        dr = self.radial.d_r
        d2r = self.radial.d2_r
        for lm in range(n_lm):
            col = coeffs[:, lm]
            dr_col = dr @ col
            radial_part = d2r @ col + 2.0 * dr_col / self.r
            l = float(self.l_values[lm])
            out[:, lm] = radial_part - l * (l + 1.0) * col / (self.r * self.r)
        return out

    def grid_to_spectral(self, grid_field):
        coeffs = self.sht.forward_radial_stack(grid_field)
        return coeffs * self.sht.dealias_mask[None, :]

    def spectral_to_grid(self, coeffs):
        return self.sht.inverse_radial_stack(coeffs)

    def linear_diffusion(self, coeffs, coeff):
        return coeff * self.spherical_laplacian_spectral_radial(coeffs)

    def nonlinear_wz_theta(self, w_coeffs, z_coeffs, theta_coeffs):
        """MagIC explicit RHS: grid force F, spat_to_qst / radial curl projections."""
        theta_grid = self.spectral_to_grid(theta_coeffs)
        u = self.pt.velocity_from_wz_coeffs(w_coeffs, z_coeffs)
        u_r, u_theta, u_phi = u[0], u[1], u[2]

        adv_r, adv_theta, adv_phi = self.pt.spherical_advection(u_r, u_theta, u_phi)
        cor_r, cor_theta, cor_phi = self.pt.coriolis_force(
            u_r, u_theta, u_phi, self.config.coriolis_coeff
        )
        buoy_r = self.config.buoyancy_coeff * theta_grid
        f_r = -adv_r + cor_r + buoy_r
        f_theta = -adv_theta + cor_theta
        f_phi = -adv_phi + cor_phi
        f_r = cp.nan_to_num(f_r)
        f_theta = cp.nan_to_num(f_theta)
        f_phi = cp.nan_to_num(f_phi)

        curl_r, curl_theta, curl_phi = self.pt.curl(f_r, f_theta, f_phi)
        ccurl_r, ccurl_th, ccurl_ph = self.pt.curl(curl_r, curl_theta, curl_phi)

        q_cc, _, _ = self.pt.angular.spat_to_qst(ccurl_r, ccurl_th, ccurl_ph)
        q_c, _, _ = self.pt.angular.spat_to_qst(curl_r, curl_theta, curl_phi)
        rhs_w = q_cc
        rhs_z = q_c
        adv_theta_s = self.pt.advection(theta_grid, u_r, u_theta, u_phi)
        rhs_theta = self.grid_to_spectral(-adv_theta_s)

        rhs_w = rhs_w * (self.l_values[None, :] > 0)
        rhs_z = rhs_z * (self.l_values[None, :] > 0)
        return rhs_w, rhs_z, rhs_theta

    def rhs_explicit(self, w_coeffs, z_coeffs, theta_coeffs):
        return self.nonlinear_wz_theta(w_coeffs, z_coeffs, theta_coeffs)

    def rhs_linear(self, w_coeffs, z_coeffs, theta_coeffs):
        lw = self.linear_diffusion(w_coeffs, self.config.viscous_coeff)
        lz = self.linear_diffusion(z_coeffs, self.config.viscous_coeff)
        ltheta = self.linear_diffusion(
            theta_coeffs, self.config.thermal_diffusion_coeff
        )
        return lw, lz, ltheta

    def apply_bc_grid(self, w_grid, z_grid, theta_grid):
        w_grid, z_grid = apply_velocity_bc_wz(w_grid, z_grid)
        theta_grid = apply_temperature_bc(theta_grid, 1.0, 0.0)
        return w_grid, z_grid, theta_grid

    def max_divergence(self, w_coeffs, z_coeffs):
        return self.pt.spectral_divergence_max(w_coeffs, z_coeffs)

    spectral_to_grid = spectral_to_grid
    max_divergence = max_divergence
