import cupy as cp


class Diagnostics:
    def __init__(self, operator):
        self.operator = operator
        self.times = []
        self.energies = []
        self.enstrophies = []

    @staticmethod
    def _host(value):
        return float(value.get())

    def record(self, time, state):
        energy = 0.5 * cp.mean(cp.sum(state * state, axis=0))
        velocity_hat = cp.fft.fftn(state, axes=(-3, -2, -1))
        kx, ky, kz = self.operator.grid.kx, self.operator.grid.ky, self.operator.grid.kz
        gradients = tuple(tuple(
            cp.fft.ifftn(1j * wave * velocity_hat[component], axes=(-3, -2, -1)).real
            for wave in (kx, ky, kz)
        ) for component in range(3))
        omega = cp.stack((
            gradients[2][1] - gradients[1][2],
            gradients[0][2] - gradients[2][0],
            gradients[1][0] - gradients[0][1],
        ))
        enstrophy = 0.5 * cp.mean(cp.sum(omega * omega, axis=0))
        energy = self._host(energy)
        enstrophy = self._host(enstrophy)
        self.times.append(time)
        self.energies.append(energy)
        self.enstrophies.append(enstrophy)
        return energy, enstrophy
