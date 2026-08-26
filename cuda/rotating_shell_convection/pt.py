"""Poloidal-toroidal velocity reconstruction (MagIC QST)."""
import cupy as cp

from .spectral_angular import SpectralAngularOps


class PoloidalToroidalOps:
    """
    u = curl curl (W e_r) + curl (Z e_r) via MagIC QST:

        Q = ℓ(ℓ+1) W / r² ,  S = (∂W/∂r)/r ,  T = Z / r
        u_r = Q Y
        u_θ = S ∂Y/∂θ + T (im/sinθ) Y
        u_φ = S (im/sinθ) Y − T ∂Y/∂θ
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self.angular = SpectralAngularOps(geometry)
        self.r = geometry.r
        self.sin_theta = geometry.sht.sin_theta
        self.cos_theta = geometry.sht.cos_theta
        self.nr = geometry.nr
        self.n_theta = geometry.n_theta
        self.n_phi = geometry.n_phi

    def _dr(self, field):
        return self.angular.dr(field)

    def _dtheta(self, field):
        return self.angular.dtheta(field)

    def _dphi(self, field):
        return self.angular.dphi(field)

    def velocity_from_wz_coeffs(self, w_coeffs, z_coeffs):
        dw = self.angular.coeffs_dr(w_coeffs)
        u_r, u_theta, u_phi = self.angular.torpol_to_spat(w_coeffs, dw, z_coeffs)
        return cp.stack((u_r, u_theta, u_phi))

    def velocity_from_wz(self, w_grid, z_grid):
        w_coeffs = self.angular.coeffs_from_grid(w_grid)
        z_coeffs = self.angular.coeffs_from_grid(z_grid)
        return self.velocity_from_wz_coeffs(w_coeffs, z_coeffs)

    def velocity_from_wz_grid(self, w_grid, z_grid):
        """Legacy grid-curl reconstruction, kept for comparison tests only."""
        r = self.r[:, None, None]
        sin_t = cp.maximum(self.sin_theta[None, :, None], 1e-6)
        dtheta_w = self._dtheta(w_grid)
        dphi_w = self._dphi(w_grid)
        u_pol_r = self.l_horizontal_laplacian(w_grid) / (r * r)
        u_pol_theta = self._dr(dtheta_w) / r
        u_pol_phi = self._dr(dphi_w / sin_t) / r
        u_tor_theta = self._dphi(z_grid) / (r * sin_t)
        u_tor_phi = -self._dtheta(z_grid) / r
        return cp.stack((
            u_pol_r,
            u_pol_theta + u_tor_theta,
            u_pol_phi + u_tor_phi,
        ))

    def l_horizontal_laplacian(self, field):
        coeffs = self.angular.coeffs_from_grid(field)
        ll1 = self.geometry.l_values[None, :]
        return -self.angular.grid_from_coeffs(-ll1 * coeffs)

    def spectral_divergence_max(self, w_coeffs, _z_coeffs):
        angular = self.angular
        r = self.r[:, None]
        dw = angular.coeffs_dr(w_coeffs)
        q = angular.ll1[None, :] * w_coeffs / (r * r)
        s = dw / r
        dq = angular.coeffs_dr(q)
        div = dq + 2.0 * q / r - angular.ll1[None, :] * s / r
        return float(cp.max(cp.abs(div)).get())

    def divergence(self, u_r, u_theta, u_phi):
        r = self.r[:, None, None]
        sin_t = cp.maximum(self.sin_theta[None, :, None], 1e-6)
        return (
            self._dr(u_r)
            + 2.0 * u_r / r
            + self._dtheta(sin_t * u_theta) / (r * sin_t)
            + self._dphi(u_phi) / (r * sin_t)
        )

    def advection(self, scalar, u_r, u_theta, u_phi):
        r = self.r[:, None, None]
        sin_t = cp.maximum(self.sin_theta[None, :, None], 1e-6)
        return (
            u_r * self._dr(scalar)
            + u_theta * self._dtheta(scalar) / r
            + u_phi * self._dphi(scalar) / (r * sin_t)
        )

    def coriolis_force(self, u_r, u_theta, u_phi, coeff=2.0):
        """RHS contribution −coeff ê_z × u in spherical components."""
        sin_t = self.sin_theta[None, :, None]
        cos_t = self.cos_theta[None, :, None]
        f_r = coeff * sin_t * u_phi
        f_theta = coeff * cos_t * u_phi
        f_phi = -coeff * (sin_t * u_r + cos_t * u_theta)
        return f_r, f_theta, f_phi

    def coriolis(self, u_r, u_theta, u_phi, coeff=2.0):
        return self.coriolis_force(u_r, u_theta, u_phi, coeff)

    def curl(self, a_r, a_theta, a_phi):
        r = self.r[:, None, None]
        sin_t = cp.maximum(self.sin_theta[None, :, None], 1e-6)
        cos_t = self.cos_theta[None, :, None]
        dth_ap = self._dtheta(a_phi)
        dph_at = self._dphi(a_theta)
        curl_r = (dth_ap + (cos_t / sin_t) * a_phi - dph_at / sin_t) / r
        dph_ar = self._dphi(a_r)
        dr_ap = self._dr(a_phi)
        curl_theta = (dph_ar / sin_t - a_phi - r * dr_ap) / r
        dr_at = self._dr(a_theta)
        dth_ar = self._dtheta(a_r)
        curl_phi = (a_theta + r * dr_at - dth_ar) / r
        return curl_r, curl_theta, curl_phi

    def spherical_advection(self, u_r, u_theta, u_phi):
        r = self.r[:, None, None]
        sin_t = cp.maximum(self.sin_theta[None, :, None], 1e-6)
        cot_t = self.cos_theta[None, :, None] / sin_t

        def advect(scalar):
            return (
                u_r * self._dr(scalar)
                + u_theta * self._dtheta(scalar) / r
                + u_phi * self._dphi(scalar) / (r * sin_t)
            )

        adv_r = advect(u_r) - (u_theta ** 2 + u_phi ** 2) / r
        adv_theta = (
            advect(u_theta)
            + u_r * u_theta / r
            - u_phi ** 2 * cot_t / r
        )
        adv_phi = (
            advect(u_phi)
            + u_r * u_phi / r
            + u_theta * u_phi * cot_t / r
        )
        return adv_r, adv_theta, adv_phi
