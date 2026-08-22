import cupy as cp


class CupySpectralOperator:
    def __init__(self, grid):
        self.grid = grid

    def gradients(self, velocity_hat):
        waves = (self.grid.kx, self.grid.ky, self.grid.kz)
        return tuple(tuple(
            cp.fft.ifftn(1j * wave * velocity_hat[component], axes=(-3, -2, -1)).real
            for wave in waves
        ) for component in range(3))

    def rhs(self, velocity, viscosity):
        velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
        velocity_hat *= self.grid.dealias_mask
        gradients = self.gradients(velocity_hat)
        nonlinear = cp.stack(tuple(
            sum(velocity[direction] * gradients[component][direction] for direction in range(3))
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
        rhs_hat = -projected - viscosity * self.grid.k_squared * velocity_hat
        return cp.fft.ifftn(rhs_hat, axes=(-3, -2, -1)).real
