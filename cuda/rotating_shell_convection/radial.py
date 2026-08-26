"""Chebyshev collocation on the spherical shell radial interval [eta, 1]."""
import cupy as cp


def chebyshev_nodes(nr: int, xp=cp):
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1]."""
    if nr < 2:
        raise ValueError("nr must be at least 2")
    indices = xp.arange(nr, dtype=xp.float64)
    return xp.cos(xp.pi * indices / (nr - 1))


def map_xi_to_r(xi, eta, xp=cp):  # pylint: disable=unused-argument
    """Map xi in [-1, 1] to physical radius r in [eta, 1]."""
    return 0.5 * ((1.0 - eta) * xi + (1.0 + eta))


def chebyshev_derivative_matrix(nodes, xp=cp):
    """First derivative matrix on Chebyshev-Gauss-Lobatto nodes."""
    nr = int(nodes.shape[0])
    diff = xp.zeros((nr, nr), dtype=xp.float64)
    c = xp.ones(nr, dtype=xp.float64)
    c[0] = 2.0
    c[-1] = 2.0
    for i in range(nr):
        for j in range(nr):
            if i == j:
                continue
            diff[i, j] = c[i] / c[j] * ((-1.0) ** (i + j)) / (nodes[i] - nodes[j])
        diff[i, i] = -xp.sum(diff[i, :])
    return diff


class RadialGrid:
    def __init__(self, nr: int, eta: float, xp=cp):
        self.nr = nr
        self.eta = float(eta)
        self.xp = xp
        self.xi = chebyshev_nodes(nr, xp=xp)
        self.r = map_xi_to_r(self.xi, self.eta, xp=xp)
        self.dxi_dr = 2.0 / (1.0 - self.eta)
        self.dr_dxi = (1.0 - self.eta) / 2.0
        self.d_xi = chebyshev_derivative_matrix(self.xi, xp=xp)
        self.d_r = self.d_xi * self.dxi_dr
        self.d2_r = self.d_r @ self.d_r
        self.dr_dxi = self.dr_dxi

    def apply_dr(self, field):
        """Radial derivative along the last axis."""
        return field @ self.d_r.T

    def apply_d2r(self, field):
        """Second radial derivative along the last axis."""
        return field @ self.d2_r.T
