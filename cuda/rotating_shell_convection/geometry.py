"""Shell geometry combining radial Chebyshev grid and spherical harmonic angular grid."""
import cupy as cp

from .radial import RadialGrid
from .sht import SphericalHarmonicTransform, _flat_to_lm


class ShellGeometry:
    def __init__(self, config):
        self.config = config
        self.radial = RadialGrid(config.nr, config.eta, xp=cp)
        self.sht = SphericalHarmonicTransform(config.l_max, xp=cp)
        self.r = self.radial.r
        self.nr = config.nr
        self.n_theta = self.sht.n_theta
        self.n_phi = self.sht.n_phi
        self.n_lm = self.sht.n_lm
        self.n_lm = self.n_lm
        self.l_max = config.l_max
        self.l_max = self.l_max
        self.eta = config.eta

        self.l_values = cp.zeros(self.n_lm, dtype=cp.float64)
        for idx in range(self.n_lm):
            l, _ = _flat_to_lm(idx)
            self.l_values[idx] = l

    def grid_shape(self):
        return (self.nr, self.n_theta, self.n_phi)
