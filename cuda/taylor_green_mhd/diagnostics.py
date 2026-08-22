import cupy as cp


class MhdDiagnostics:
    def __init__(self, operator):
        self.operator = operator
        self.times = []
        self.kinetic_energies = []
        self.magnetic_energies = []
        self.total_energies = []
        self.cross_helicities = []
        self.enstrophies = []
        self.divergence_u_max = []
        self.divergence_b_max = []

    @staticmethod
    def _host(value):
        return float(value.get())

    def record(self, time, velocity_hat, magnetic_field_hat):
        velocity = self.operator.grid.to_real(velocity_hat)
        magnetic_field = self.operator.grid.to_real(magnetic_field_hat)
        kinetic = 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))
        magnetic = 0.5 * cp.mean(cp.sum(magnetic_field * magnetic_field, axis=0))
        cross = cp.mean(
            velocity[0] * magnetic_field[0]
            + velocity[1] * magnetic_field[1]
            + velocity[2] * magnetic_field[2]
        )
        omega = self.operator.grid.to_real(self.operator.spectral_curl_hat(velocity_hat))
        enstrophy = 0.5 * cp.mean(cp.sum(omega * omega, axis=0))
        div_u_max = self._host(cp.max(cp.abs(self.operator.divergence_from_hat(velocity_hat))))
        div_b_max = self._host(cp.max(cp.abs(self.operator.divergence_from_hat(magnetic_field_hat))))

        kinetic = self._host(kinetic)
        magnetic = self._host(magnetic)
        cross = self._host(cross)
        enstrophy = self._host(enstrophy)

        self.times.append(time)
        self.kinetic_energies.append(kinetic)
        self.magnetic_energies.append(magnetic)
        self.total_energies.append(kinetic + magnetic)
        self.cross_helicities.append(cross)
        self.enstrophies.append(enstrophy)
        self.divergence_u_max.append(div_u_max)
        self.divergence_b_max.append(div_b_max)

        return kinetic, magnetic, cross, enstrophy
