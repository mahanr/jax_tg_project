import cupy as cp


class MhdSpectralOperator:
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

    def spectral_curl(self, field_hat):
        return self.grid.to_real(self.spectral_curl_hat(field_hat))

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

    def divergence_from_hat(self, vector_hat):
        return self.grid.to_real(self.divergence_hat(vector_hat))

    def spectral_divergence(self, vector):
        vector_hat = cp.fft.fftn(vector, axes=(-3, -2, -1))
        return self.divergence_from_hat(vector_hat)

    def _advection_rhs_hat(self, velocity_hat, velocity):
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
        return self.leray_project(nonlinear_hat)

    def rhs(self, velocity_hat, magnetic_field_hat, viscosity, resistivity):
        velocity_hat = self.grid.dealias(velocity_hat)
        magnetic_field_hat = self.grid.dealias(magnetic_field_hat)
        velocity = self.grid.to_real(velocity_hat)
        magnetic_field = self.grid.to_real(magnetic_field_hat)

        advection_proj = self._advection_rhs_hat(velocity_hat, velocity)

        current = self.spectral_curl(magnetic_field_hat)
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

        cross = cp.stack((
            velocity[1] * magnetic_field[2] - velocity[2] * magnetic_field[1],
            velocity[2] * magnetic_field[0] - velocity[0] * magnetic_field[2],
            velocity[0] * magnetic_field[1] - velocity[1] * magnetic_field[0],
        ))
        cross_hat = cp.fft.fftn(cross, axes=(-3, -2, -1))
        cross_hat *= self.grid.dealias_mask
        induction_hat = self.spectral_curl_hat(cross_hat)
        induction_hat *= self.grid.dealias_mask
        induction_clean = self.divergence_clean(induction_hat)

        rhs_b_hat = induction_clean - resistivity * self.grid.k_squared * magnetic_field_hat

        return rhs_u_hat, rhs_b_hat
