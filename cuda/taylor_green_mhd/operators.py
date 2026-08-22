import cupy as cp


class MhdSpectralOperator:
    def __init__(self, grid):
        self.grid = grid

    def gradients(self, field_hat):
        waves = (self.grid.kx, self.grid.ky, self.grid.kz)
        return tuple(tuple(
            cp.fft.ifftn(1j * wave * field_hat[component], axes=(-3, -2, -1)).real
            for wave in waves
        ) for component in range(3))

    def spectral_curl(self, field_hat):
        gradients = self.gradients(field_hat)
        return cp.stack((
            gradients[2][1] - gradients[1][2],
            gradients[0][2] - gradients[2][0],
            gradients[1][0] - gradients[0][1],
        ))

    def leray_project(self, vector_hat):
        safe_k_squared = cp.where(self.grid.k_squared == 0.0, 1.0, self.grid.k_squared)
        k_dot = (
            self.grid.kx * vector_hat[0]
            + self.grid.ky * vector_hat[1]
            + self.grid.kz * vector_hat[2]
        )
        return vector_hat - cp.stack((
            self.grid.kx * k_dot / safe_k_squared,
            self.grid.ky * k_dot / safe_k_squared,
            self.grid.kz * k_dot / safe_k_squared,
        ))

    def divergence_hat(self, vector_hat):
        return 1j * (
            self.grid.kx * vector_hat[0]
            + self.grid.ky * vector_hat[1]
            + self.grid.kz * vector_hat[2]
        )

    def divergence_clean(self, vector_hat):
        return self.leray_project(vector_hat)

    def spectral_divergence(self, vector):
        vector_hat = cp.fft.fftn(vector, axes=(-3, -2, -1))
        return cp.fft.ifftn(self.divergence_hat(vector_hat), axes=(-3, -2, -1)).real

    def _advection_rhs_hat(self, velocity):
        velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
        velocity_hat *= self.grid.dealias_mask
        gradients = self.gradients(velocity_hat)
        nonlinear = cp.stack(tuple(
            sum(velocity[direction] * gradients[component][direction] for direction in range(3))
            for component in range(3)
        ))
        nonlinear_hat = cp.fft.fftn(nonlinear, axes=(-3, -2, -1))
        nonlinear_hat *= self.grid.dealias_mask
        return self.leray_project(nonlinear_hat)

    def rhs(self, velocity, magnetic_field, viscosity, resistivity):
        velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
        velocity_hat *= self.grid.dealias_mask
        advection_proj = self._advection_rhs_hat(velocity)

        magnetic_hat = cp.fft.fftn(magnetic_field, axes=(-3, -2, -1))
        magnetic_hat *= self.grid.dealias_mask
        current = self.spectral_curl(magnetic_hat)
        lorentz = cp.stack((
            current[1] * magnetic_field[2] - current[2] * magnetic_field[1],
            current[2] * magnetic_field[0] - current[0] * magnetic_field[2],
            current[0] * magnetic_field[1] - current[1] * magnetic_field[0],
        ))
        lorentz_hat = cp.fft.fftn(lorentz, axes=(-3, -2, -1))
        lorentz_hat *= self.grid.dealias_mask
        lorentz_proj = self.leray_project(lorentz_hat)

        rhs_u_hat = (
            -advection_proj
            - lorentz_proj
            - viscosity * self.grid.k_squared * velocity_hat
        )
        dudt = cp.fft.ifftn(rhs_u_hat, axes=(-3, -2, -1)).real

        cross = cp.stack((
            velocity[1] * magnetic_field[2] - velocity[2] * magnetic_field[1],
            velocity[2] * magnetic_field[0] - velocity[0] * magnetic_field[2],
            velocity[0] * magnetic_field[1] - velocity[1] * magnetic_field[0],
        ))
        cross_hat = cp.fft.fftn(cross, axes=(-3, -2, -1))
        cross_hat *= self.grid.dealias_mask
        induction = self.spectral_curl(cross_hat)
        induction_hat = cp.fft.fftn(induction, axes=(-3, -2, -1))
        induction_hat *= self.grid.dealias_mask
        induction_clean = self.divergence_clean(induction_hat)

        rhs_b_hat = induction_clean - resistivity * self.grid.k_squared * magnetic_hat
        dbdt = cp.fft.ifftn(rhs_b_hat, axes=(-3, -2, -1)).real

        return dudt, dbdt
