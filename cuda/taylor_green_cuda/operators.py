import cupy as cp


_project_and_diffuse = cp.ElementwiseKernel(
    "T n0, T n1, T n2, T u0, T u1, T u2, "
    "X kx, X ky, X kz, X k2, X nu, bool keep",
    "T o0, T o1, T o2",
    """
    if (!keep) {
        o0 = T();
        o1 = T();
        o2 = T();
    } else {
        X inv = (k2 == static_cast<X>(0)) ? static_cast<X>(0)
                                          : static_cast<X>(1) / k2;
        T kdot = n0 * kx + n1 * ky + n2 * kz;
        o0 = -n0 + kx * kdot * inv - nu * k2 * u0;
        o1 = -n1 + ky * kdot * inv - nu * k2 * u1;
        o2 = -n2 + kz * kdot * inv - nu * k2 * u2;
    }
    """,
    "tg_project_and_diffuse",
)


class CupySpectralOperator:
    def __init__(self, grid):
        self.grid = grid

    def gradients(self, field_hat):
        """Spectral derivatives ik_d * field_hat[component].

        Returns a single complex array with shape (3, 3, n, n, n), indexed as
        [component, direction].
        """
        return self.grid.dealias(self.grid.ik[None, ...] * field_hat[:, None, ...])

    def spectral_curl_hat(self, field_hat):
        gradients_hat = self.gradients(field_hat)
        return self.grid.dealias(cp.stack((
            gradients_hat[2, 1] - gradients_hat[1, 2],
            gradients_hat[0, 2] - gradients_hat[2, 0],
            gradients_hat[1, 0] - gradients_hat[0, 1],
        )))

    def project_and_diffuse(self, nonlinear_hat, velocity_hat, viscosity):
        dtype = velocity_hat.dtype
        real_dtype = cp.float32 if dtype == cp.complex64 else cp.float64
        o0, o1, o2 = _project_and_diffuse(
            nonlinear_hat[0].astype(dtype, copy=False),
            nonlinear_hat[1].astype(dtype, copy=False),
            nonlinear_hat[2].astype(dtype, copy=False),
            velocity_hat[0],
            velocity_hat[1],
            velocity_hat[2],
            self.grid.kx,
            self.grid.ky,
            self.grid.kz,
            self.grid.k_squared,
            real_dtype(viscosity),
            self.grid.dealias_mask,
        )
        return cp.stack((o0, o1, o2))

    def rhs(self, velocity_hat, viscosity):
        velocity_hat = self.grid.dealias(velocity_hat)
        velocity = self.grid.to_real(velocity_hat)
        gradient_real = self.grid.to_real(self.gradients(velocity_hat))
        nonlinear = (velocity[None, ...] * gradient_real).sum(axis=1)
        nonlinear_hat = cp.fft.fftn(nonlinear, axes=(-3, -2, -1))
        return self.project_and_diffuse(nonlinear_hat, velocity_hat, viscosity)
