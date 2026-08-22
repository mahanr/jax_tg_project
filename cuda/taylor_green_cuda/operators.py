import cupy as cp


class CupySpectralOperator:
    def __init__(self, grid):
        self.grid = grid

    def gradients(self, field_hat):
        """Spectral derivatives: ik_d * field_hat[component] (complex)."""
        waves = (self.grid.kx, self.grid.ky, self.grid.kz)
        return tuple(tuple(
            self.grid.dealias(1j * wave * field_hat[component])
            for wave in waves
        ) for component in range(3))

    def spectral_curl_hat(self, field_hat):
        gradients_hat = self.gradients(field_hat)
        return self.grid.dealias(cp.stack((
            gradients_hat[2][1] - gradients_hat[1][2],
            gradients_hat[0][2] - gradients_hat[2][0],
            gradients_hat[1][0] - gradients_hat[0][1],
        )))

    def rhs(self, velocity_hat, viscosity):
        velocity_hat = self.grid.dealias(velocity_hat)
        velocity = self.grid.to_real(velocity_hat)
        gradients_hat = self.gradients(velocity_hat)
        nonlinear = cp.stack(tuple(
            sum(
                velocity[direction] * self.grid.to_real(gradients_hat[component][direction])
                for direction in range(3)
            )
            for component in range(3)
        ))
        nonlinear_hat = cp.fft.fftn(nonlinear, axes=(-3, -2, -1))
        nonlinear_hat *= self.grid.dealias_mask
        safe_k_squared = cp.where(self.grid.k_squared == 0.0, 1.0, self.grid.k_squared)
        k_dot_nonlinear = (
            self.grid.kx * nonlinear_hat[0]
            + self.grid.ky * nonlinear_hat[1]
            + self.grid.kz * nonlinear_hat[2]
        )
        projected = nonlinear_hat - cp.stack((
            self.grid.kx * k_dot_nonlinear / safe_k_squared,
            self.grid.ky * k_dot_nonlinear / safe_k_squared,
            self.grid.kz * k_dot_nonlinear / safe_k_squared,
        ))
        return -projected - viscosity * self.grid.k_squared * velocity_hat
