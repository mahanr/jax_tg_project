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

    def record(self, time, velocity, magnetic_field):
        kinetic = 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))
        magnetic = 0.5 * cp.mean(cp.sum(magnetic_field * magnetic_field, axis=0))
        cross = cp.mean(
            velocity[0] * magnetic_field[0]
            + velocity[1] * magnetic_field[1]
            + velocity[2] * magnetic_field[2]
        )
        velocity_hat = cp.fft.fftn(velocity, axes=(-3, -2, -1))
        gradients = self.operator.gradients(velocity_hat)
        omega = cp.stack((
            gradients[2][1] - gradients[1][2],
            gradients[0][2] - gradients[2][0],
            gradients[1][0] - gradients[0][1],
        ))
        enstrophy = 0.5 * cp.mean(cp.sum(omega * omega, axis=0))
        div_u = self.operator.spectral_divergence(velocity)
        div_b = self.operator.spectral_divergence(magnetic_field)

        kinetic = self._host(kinetic)
        magnetic = self._host(magnetic)
        cross = self._host(cross)
        enstrophy = self._host(enstrophy)
        div_u_max = self._host(cp.max(cp.abs(div_u)))
        div_b_max = self._host(cp.max(cp.abs(div_b)))

        self.times.append(time)
        self.kinetic_energies.append(kinetic)
        self.magnetic_energies.append(magnetic)
        self.total_energies.append(kinetic + magnetic)
        self.cross_helicities.append(cross)
        self.enstrophies.append(enstrophy)
        self.divergence_u_max.append(div_u_max)
        self.divergence_b_max.append(div_b_max)

        return kinetic, magnetic, cross, enstrophy
