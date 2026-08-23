"""Spherical harmonic transforms via precomputed quadrature matrices."""
import cupy as cp
import numpy as np
from scipy.special import sph_harm_y


def _lm_to_flat(l, m):
    return l * l + m + l


def _flat_to_lm(index):
    l = int(np.floor(np.sqrt(index)))
    if l * l > index:
        l -= 1
    m = index - l * l - l
    return l, m


def _n_lm(l_max):
    return (l_max + 1) ** 2


def collocation_grid_shape(l_max):
    """MagIC nalias=20 / Orszag 3/2: N_θ = ceil(3/2 (L+1)), N_φ = 2 N_θ."""
    n_theta = int(np.ceil(1.5 * (l_max + 1)))
    n_phi = 2 * n_theta
    return n_theta, n_phi


def _build_transform_matrices(l_max, mu_host, phi_host, weights_host):
    """
    Discrete orthonormal SHT plus QST synthesis/analysis matrices.

    Scalar:  c_lm = sum_g W_g conj(Y_lm) f
             f = sum_lm c_lm Y_lm

    QST synthesis (MagIC/SHTns):
             u_r = Q Y
             u_θ = S ∂Y/∂θ + T (im/sinθ) Y
             u_φ = S (im/sinθ) Y − T ∂Y/∂θ
    """
    n_theta = mu_host.shape[0]
    n_phi = phi_host.shape[0]
    n_lm = _n_lm(l_max)
    n_grid = n_theta * n_phi
    theta_host = np.arccos(mu_host)
    phi_weight = 2.0 * np.pi / n_phi
    sin_host = np.sqrt(np.maximum(1.0 - mu_host ** 2, 1e-30))

    forward = np.zeros((n_lm, n_grid), dtype=np.complex128)
    synthesis = np.zeros((n_grid, n_lm), dtype=np.complex128)
    dYdtheta = np.zeros((n_grid, n_lm), dtype=np.complex128)
    imYsin = np.zeros((n_grid, n_lm), dtype=np.complex128)

    for lm_idx in range(n_lm):
        l, m = _flat_to_lm(lm_idx)
        for j in range(n_theta):
            for k in range(n_phi):
                grid_idx = j * n_phi + k
                harmonic = sph_harm_y(l, m, theta_host[j], phi_host[k])
                weight = weights_host[j] * phi_weight
                forward[lm_idx, grid_idx] = weight * np.conj(harmonic)
                synthesis[grid_idx, lm_idx] = harmonic
                imYsin[grid_idx, lm_idx] = (1j * m / sin_host[j]) * harmonic

    for lm_idx in range(n_lm):
        l, m = _flat_to_lm(lm_idx)
        if l == 0:
            continue
        if abs(m) <= l - 1:
            prev_idx = _lm_to_flat(l - 1, m)
            scale = np.sqrt((2.0 * l + 1.0) / (2.0 * l - 1.0) * (l * l - m * m))
        else:
            prev_idx = None
            scale = 0.0
        for j in range(n_theta):
            mu = mu_host[j]
            st = sin_host[j]
            for k in range(n_phi):
                grid_idx = j * n_phi + k
                y_lm = synthesis[grid_idx, lm_idx]
                y_prev = 0.0 if prev_idx is None else synthesis[grid_idx, prev_idx]
                dYdtheta[grid_idx, lm_idx] = (l * mu * y_lm - scale * y_prev) / st

    dY_forward = np.zeros((n_lm, n_grid), dtype=np.complex128)
    imY_forward = np.zeros((n_lm, n_grid), dtype=np.complex128)
    for lm_idx in range(n_lm):
        for j in range(n_theta):
            for k in range(n_phi):
                grid_idx = j * n_phi + k
                weight = weights_host[j] * phi_weight
                dY_forward[lm_idx, grid_idx] = weight * np.conj(dYdtheta[grid_idx, lm_idx])
                imY_forward[lm_idx, grid_idx] = weight * np.conj(imYsin[grid_idx, lm_idx])

    return forward, synthesis, dYdtheta, imYsin, dY_forward, imY_forward


class SphericalHarmonicTransform:
    def __init__(self, l_max: int, xp=cp):
        self.l_max = l_max
        self.xp = xp
        self.n_theta, self.n_phi = collocation_grid_shape(l_max)
        self.n_lm = _n_lm(l_max)
        self.n_lm = self.n_lm

        mu_host, weights_host = np.polynomial.legendre.leggauss(self.n_theta)
        phi_host = np.linspace(0.0, 2.0 * np.pi, self.n_phi, endpoint=False)

        self.mu = xp.asarray(mu_host)
        self.weights = xp.asarray(weights_host)
        self.theta = xp.arccos(self.mu)
        self.sin_theta = xp.sqrt(1.0 - self.mu ** 2)
        self.cos_theta = self.mu
        self.phi = xp.asarray(phi_host)

        (
            forward_host,
            synthesis_host,
            dY_host,
            imY_host,
            dY_fwd_host,
            imY_fwd_host,
        ) = _build_transform_matrices(l_max, mu_host, phi_host, weights_host)
        self.forward_mat = xp.asarray(forward_host)
        self.inverse_mat = xp.asarray(synthesis_host)
        self.dYdtheta_mat = xp.asarray(dY_host)
        self.imYsin_mat = xp.asarray(imY_host)
        self.dYdtheta_forward_mat = xp.asarray(dY_fwd_host)
        self.imYsin_forward_mat = xp.asarray(imY_fwd_host)

        l_host = np.zeros(self.n_lm, dtype=np.float64)
        m_host = np.zeros(self.n_lm, dtype=np.float64)
        for idx in range(self.n_lm):
            l, m = _flat_to_lm(idx)
            l_host[idx] = l
            m_host[idx] = m
        self.l_values = xp.asarray(l_host)
        self.m_values = xp.asarray(m_host)
        ll1 = l_host * (l_host + 1.0)
        ll1[ll1 == 0.0] = 1.0
        self.ll1 = xp.asarray(ll1)
        self.ll1_inv = 1.0 / self.ll1
        self.ll1_inv = self.ll1_inv * (self.l_values > 0)

        # Physical truncation is l_max; the 3/2 grid supplies dealiasing.
        self.dealias_mask = xp.ones(self.n_lm, dtype=xp.float64)
        self.dealias_mask = self.dealias_mask * (self.l_values <= l_max)

    def dealias(self, coeffs):
        if coeffs.ndim == 1:
            return coeffs * self.dealias_mask
        return coeffs * self.dealias_mask[None, :]

    def forward(self, field_grid):
        flat = field_grid.reshape(-1)
        coeffs = self.forward_mat @ flat
        return coeffs * self.dealias_mask

    def inverse(self, coeffs):
        coeffs = coeffs * self.dealias_mask
        flat = self.inverse_mat @ coeffs
        return flat.reshape(self.n_theta, self.n_phi).real

    def forward_radial_stack(self, field_grid):
        nr = field_grid.shape[0]
        flat = field_grid.reshape(nr, -1)
        return self.dealias((self.forward_mat @ flat.T).T)

    def inverse_radial_stack(self, coeffs):
        nr = coeffs.shape[0]
        flat = (self.inverse_mat @ self.dealias(coeffs).T).T
        return flat.reshape(nr, self.n_theta, self.n_phi).real

    inverse_radial_stack = inverse_radial_stack
    forward_radial_stack = forward_radial_stack
    inverse_radial_stack = inverse_radial_stack
    forward_radial_stack = forward_radial_stack

    def synthesize_dtheta(self, coeffs):
        nr = coeffs.shape[0]
        flat = (self.dYdtheta_mat @ self.dealias(coeffs).T).T
        return flat.reshape(nr, self.n_theta, self.n_phi).real

    def synthesize_dphi_over_sintheta(self, coeffs):
        nr = coeffs.shape[0]
        flat = (self.imYsin_mat @ self.dealias(coeffs).T).T
        return flat.reshape(nr, self.n_theta, self.n_phi).real
