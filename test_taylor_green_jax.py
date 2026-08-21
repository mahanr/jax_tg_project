import jax.numpy as jnp
import unittest

from taylor_green_jax import (
    advance_one_step,
    initial_taylor_green,
    make_dealias_mask,
    make_wavenumbers,
    reynolds_to_viscosity,
    viscosity_to_reynolds,
    run_simulation,
    spectral_divergence,
    spectral_rhs,
    validate_timestep,
)


LENGTH = 2.0 * jnp.pi


class TaylorGreenTests(unittest.TestCase):
    def test_initial_condition_is_divergence_free(self):
        kx, ky, kz = make_wavenumbers(16, LENGTH)
        velocity = initial_taylor_green(16, LENGTH)
        divergence = spectral_divergence(velocity, kx, ky, kz)
        self.assertLess(float(jnp.max(jnp.abs(divergence))), 1e-12)


    def test_dealias_mask_keeps_two_thirds_of_each_axis(self):
        mask = make_dealias_mask(16, LENGTH)
        self.assertEqual(mask.shape, (16, 16, 16))
        self.assertEqual(int(mask.sum()), 11**3)
        self.assertFalse(bool(mask[8, 0, 0]))
        self.assertTrue(bool(mask[5, 0, 0]))


    def test_timestep_validation_rejects_unstable_values(self):
        kx, ky, kz = make_wavenumbers(16, LENGTH)
        with self.assertRaisesRegex(ValueError, "stability limit"):
            validate_timestep(1.0, 0.01, kx, ky, kz)
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_timestep(0.0, 0.01, kx, ky, kz)


    def test_zero_mode_is_not_viscously_damped(self):
        N = 8
        kx, ky, kz = make_wavenumbers(N, LENGTH)
        mask = make_dealias_mask(N, LENGTH)
        constant_velocity = jnp.ones((3, N, N, N))
        rhs = spectral_rhs(constant_velocity, 0.01, kx, ky, kz, mask)
        self.assertLess(float(jnp.max(jnp.abs(rhs))), 1e-12)


    def test_short_run_is_finite_divergence_free_and_decays(self):
        velocity, energies, diagnostics = run_simulation(
            N=8,
            dt=0.005,
            nu=0.01,
            n_steps=4,
            save_every=2,
            return_diagnostics=True,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(velocity))))
        self.assertEqual(diagnostics["time"], [0.0, 0.01, 0.02])
        self.assertLess(max(diagnostics["divergence_max"]), 1e-10)
        self.assertTrue(all(a >= b for a, b in zip(energies, energies[1:])))


    def test_step_preserves_finite_values(self):
        N = 8
        kx, ky, kz = make_wavenumbers(N, LENGTH)
        mask = make_dealias_mask(N, LENGTH)
        velocity = initial_taylor_green(N, LENGTH)
        updated = advance_one_step(velocity, 0.005, 0.01, kx, ky, kz, mask)
        self.assertTrue(bool(jnp.all(jnp.isfinite(updated))))

    def test_reynolds_number_conversion(self):
        L = 2.0 * jnp.pi
        amp = 1.0
        re = 100.0
        nu = reynolds_to_viscosity(re, amp, L)
        re_back = viscosity_to_reynolds(nu, amp, L)
        self.assertAlmostEqual(re, re_back, places=10)

    def test_run_with_reynolds_number(self):
        u, energies = run_simulation(
            N=8,
            dt=0.005,
            reynolds=100,
            n_steps=4,
            save_every=2,
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(u))))
        self.assertTrue(all(a >= b for a, b in zip(energies, energies[1:])))

    def test_reynolds_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "Reynolds number must be positive"):
            reynolds_to_viscosity(-100)


if __name__ == "__main__":
    unittest.main()
