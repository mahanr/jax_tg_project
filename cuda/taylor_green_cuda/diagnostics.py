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

    def record(self, time, state_hat):
        velocity = self.operator.grid.to_real(state_hat)
        energy = 0.5 * cp.mean(cp.sum(velocity * velocity, axis=0))
        omega = self.operator.grid.to_real(self.operator.spectral_curl_hat(state_hat))
        enstrophy = 0.5 * cp.mean(cp.sum(omega * omega, axis=0))
        energy = self._host(energy)
        enstrophy = self._host(enstrophy)
        self.times.append(time)
        self.energies.append(energy)
        self.enstrophies.append(enstrophy)
        return energy, enstrophy
