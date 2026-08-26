"""Initial temperature profile and small perturbations for W, Z, theta."""
import cupy as cp
import numpy as np

from .sht import _lm_to_flat


class ShellInitialCondition:
    def __init__(self, geometry, perturbation_amplitude=1e-3):
        self.geometry = geometry
        self.perturbation_amplitude = float(perturbation_amplitude)

    def conductive_temperature(self):
        """Harmonic conductive profile θ = [η/(1-η)](1/r - 1); θ(η)=1, θ(1)=0."""
        r = self.geometry.r
        eta = self.geometry.eta
        return (eta / (1.0 - eta)) * (1.0 / r - 1.0)

    def _no_slip_envelope(self):
        r = self.geometry.r
        eta = self.geometry.eta
        return (r * r) * ((r - eta) ** 2) * ((1.0 - r) ** 2)

    def _bandlimited_velocity_coeffs(self, rng):
        """Random bandlimited W, Z with no-slip envelope W=dW/dr=Z=0 at walls."""
        geometry = self.geometry
        sht = geometry.sht
        l_max = geometry.l_max
        amp = self.perturbation_amplitude
        envelope = self._no_slip_envelope()
        w_coeffs = cp.zeros((geometry.nr, sht.n_lm), dtype=cp.complex128)
        z_coeffs = cp.zeros_like(w_coeffs)

        for l in range(1, l_max + 1):
            lm0 = _lm_to_flat(l, 0)
            amp_w = amp * rng.normal()
            amp_z = amp * rng.normal()
            w_coeffs[:, lm0] = amp_w * envelope
            z_coeffs[:, lm0] = amp_z * envelope
            for m in range(1, l + 1):
                a = rng.normal()
                b = rng.normal()
                w_coeffs[:, _lm_to_flat(l, m)] = amp * envelope * (a + 1j * b) / 2.0
                w_coeffs[:, _lm_to_flat(l, -m)] = (
                    amp * envelope * ((-1) ** m) * (a - 1j * b) / 2.0
                )
                a = rng.normal()
                b = rng.normal()
                z_coeffs[:, _lm_to_flat(l, m)] = amp * envelope * (a + 1j * b) / 2.0
                z_coeffs[:, _lm_to_flat(l, -m)] = (
                    amp * envelope * ((-1) ** m) * (a - 1j * b) / 2.0
                )

        w_coeffs[0] = 0.0
        w_coeffs[-1] = 0.0
        z_coeffs[0] = 0.0
        z_coeffs[-1] = 0.0
        return w_coeffs, z_coeffs

    def create(self):
        geometry = self.geometry
        sht = geometry.sht
        nr = geometry.nr
        n_theta = geometry.n_theta
        n_phi = geometry.n_phi

        theta_grid = cp.zeros((nr, n_theta, n_phi), dtype=cp.float64)
        conductive = self.conductive_temperature()
        for i in range(nr):
            theta_grid[i] = conductive[i]

        rng = np.random.default_rng(42)
        noise = 0.01 * self.perturbation_amplitude * cp.asarray(
            rng.standard_normal((nr, n_theta, n_phi)), dtype=cp.float64
        )
        theta_grid = theta_grid + noise
        theta_grid[0] = 0.0
        theta_grid[-1] = 1.0

        w_coeffs, z_coeffs = self._bandlimited_velocity_coeffs(rng)
        theta_coeffs = sht.forward_radial_stack(theta_grid)
        return w_coeffs, z_coeffs, theta_coeffs
