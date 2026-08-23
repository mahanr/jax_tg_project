"""Spectral angular operators and MagIC QST torpol transforms."""
import cupy as cp
import numpy as np

from .sht import _flat_to_lm


class SpectralAngularOps:
    """
    Angular derivatives via Y, ∂Y/∂θ, (im/sinθ)Y on the GL × uniform-φ grid.
    Radial derivatives use Chebyshev collocation.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self.sht = geometry.sht
        self.radial = geometry.radial
        self.nr = geometry.nr
        self.n_theta = geometry.n_theta
        self.n_phi = geometry.n_phi
        self.n_lm = geometry.n_lm
        self.r = geometry.r
        self.forward_mat = self.sht.forward_mat
        self.inverse_mat = self.sht.inverse_mat
        self.dYdtheta_mat = self.sht.dYdtheta_mat
        self.imYsin_mat = self.sht.imYsin_mat
        self.dYdtheta_forward_mat = self.sht.dYdtheta_forward_mat
        self.imYsin_forward_mat = self.sht.imYsin_forward_mat
        self.l_values = self.sht.l_values
        self.ll1 = self.sht.ll1
        self.ll1_inv = self.sht.ll1_inv
        self.dealias_mask = self.sht.dealias_mask
        dphi = np.zeros(self.n_lm, dtype=np.complex128)
        for lm_idx in range(self.n_lm):
            _, m = _flat_to_lm(lm_idx)
            dphi[lm_idx] = 1j * m
        self.dphi_factors = cp.asarray(dphi)

    def _coeffs_from_grid(self, field):
        flat = field.reshape(self.nr, -1)
        return self.sht.dealias((self.forward_mat @ flat.T).T)

    def _grid_from_coeffs(self, coeffs):
        flat = (self.inverse_mat @ self.sht.dealias(coeffs).T).T
        return flat.reshape(self.nr, self.n_theta, self.n_phi).real

    def coeffs_dr(self, coeffs):
        return self.radial.d_r @ coeffs

    def dr(self, field):
        nr, n_theta, n_phi = field.shape
        return (self.radial.d_r @ field.reshape(nr, -1)).reshape(nr, n_theta, n_phi)

    def dtheta(self, field):
        coeffs = self._coeffs_from_grid(field)
        return self.sht.synthesize_dtheta(coeffs)

    def dphi(self, field):
        coeffs = self._coeffs_from_grid(field)
        dphi_coeffs = coeffs * self.dphi_factors
        return self._grid_from_coeffs(dphi_coeffs)

    def dphi_over_sintheta(self, field):
        coeffs = self._coeffs_from_grid(field)
        return self.sht.synthesize_dphi_over_sintheta(coeffs)

    def coeffs_dphi(self, coeffs):
        return coeffs * self.dphi_factors

    def torpol_to_spat(self, w_coeffs, dw_coeffs, z_coeffs):
        """MagIC QST: u from W, dW/dr, Z in coefficient space."""
        r = self.r[:, None]
        ll1 = self.ll1[None, :]
        q_coeffs = ll1 * w_coeffs / (r * r)
        s_coeffs = dw_coeffs / r
        t_coeffs = z_coeffs / r
        u_r = self._grid_from_coeffs(q_coeffs)
        u_theta = (
            self.sht.synthesize_dtheta(s_coeffs)
            + self.sht.synthesize_dphi_over_sintheta(t_coeffs)
        )
        u_phi = (
            self.sht.synthesize_dphi_over_sintheta(s_coeffs)
            - self.sht.synthesize_dtheta(t_coeffs)
        )
        return u_r, u_theta, u_phi

    def spat_to_qst(self, u_r, u_theta, u_phi):
        """Adjoint QST: grid vector -> Q, S, T coefficients (nr, n_lm)."""
        ur_flat = u_r.reshape(self.nr, -1)
        ut_flat = u_theta.reshape(self.nr, -1)
        up_flat = u_phi.reshape(self.nr, -1)
        q_coeffs = (self.forward_mat @ ur_flat.T).T
        s_num = (
            (self.dYdtheta_forward_mat @ ut_flat.T).T
            + (self.imYsin_forward_mat @ up_flat.T).T
        )
        t_num = (
            (self.imYsin_forward_mat @ ut_flat.T).T
            - (self.dYdtheta_forward_mat @ up_flat.T).T
        )
        inv = self.ll1_inv[None, :]
        s_coeffs = s_num * inv
        t_coeffs = t_num * inv
        mask = self.dealias_mask[None, :]
        return q_coeffs * mask, s_coeffs * mask, t_coeffs * mask
