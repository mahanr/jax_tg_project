import time

import cupy as cp

from taylor_green_cuda.grid import SpectralGrid

from .config import TaylorGreenMhdConfig, validate_timestep
from .diagnostics import MhdDiagnostics
from .initial_conditions import TaylorGreenMhdInitialCondition
from .integrator import MhdRK4Integrator
from .operators import MhdSpectralOperator
from .plotting import save_plots


class TaylorGreenMhdSimulation:
    def __init__(self, config=None, make_plots=True):
        self.config = config or TaylorGreenMhdConfig()
        self.grid = SpectralGrid(self.config.n, self.config.domain_length)
        self.initial_condition = TaylorGreenMhdInitialCondition(
            self.config.domain_length,
            self.config.magnetic_amplitude,
        )
        self.operator = MhdSpectralOperator(self.grid)
        self.integrator = MhdRK4Integrator(
            self.operator,
            self.config.dt,
            self.config.viscosity,
            self.config.resistivity,
        )
        velocity, magnetic_field = self.initial_condition.create(self.config.n)
        self.velocity_hat = self.grid.to_spectral(velocity)
        self.magnetic_field_hat = self.grid.to_spectral(magnetic_field)
        self.diagnostics = MhdDiagnostics(self.operator)
        self.make_plots = make_plots

    def record(self, step):
        kinetic, magnetic, cross, enstrophy = self.diagnostics.record(
            step * self.config.dt,
            self.velocity_hat,
            self.magnetic_field_hat,
        )
        print(
            f"Save time step: {step}, time: {step * self.config.dt:.6f}, "
            f"E_k: {kinetic:.8e}, E_m: {magnetic:.8e}, H_c: {cross:.8e}, "
            f"enstrophy: {enstrophy:.8e}"
        )
        return kinetic, magnetic, cross, enstrophy

    def run(self):
        config = self.config
        validate_timestep(
            config.dt,
            config.viscosity,
            config.resistivity,
            self.grid,
            max_velocity=1.0,
            max_field=config.magnetic_amplitude,
        )
        self.diagnostics.record(0.0, self.velocity_hat, self.magnetic_field_hat)
        start = time.perf_counter()
        state = (self.velocity_hat, self.magnetic_field_hat)
        for step in range(1, config.steps + 1):
            state = self.integrator.step(state)
            self.velocity_hat, self.magnetic_field_hat = state
            self.velocity_hat = self.grid.dealias(self.velocity_hat)
            self.magnetic_field_hat = self.grid.dealias(self.magnetic_field_hat)
            if step % config.save_every == 0 or step == config.steps:
                self.record(step)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        print(f"CUDA device: {device_name}")
        print(f"Reynolds number: Re = {config.reynolds:.1f}")
        print(f"Magnetic Reynolds number: Rm = {config.magnetic_reynolds:.1f}")
        print(f"Simulated time: {config.total_time:.3f} ({config.steps} steps) in {elapsed:.3f} s")
        print(
            f"Kinetic energy trace: {self.diagnostics.kinetic_energies[0]:.6f} -> "
            f"{self.diagnostics.kinetic_energies[-1]:.6f}"
        )
        print(
            f"Magnetic energy trace: {self.diagnostics.magnetic_energies[0]:.6f} -> "
            f"{self.diagnostics.magnetic_energies[-1]:.6f}"
        )
        print(
            f"Max divergence u: {max(self.diagnostics.divergence_u_max):.3e}, "
            f"B: {max(self.diagnostics.divergence_b_max):.3e}"
        )
        if self.make_plots:
            save_plots(
                self.velocity_hat,
                self.magnetic_field_hat,
                self.diagnostics,
                config.n,
                self.grid,
            )
            print("Saved: taylor_green_mhd_slice.png")
            print("Saved: taylor_green_mhd_diagnostics.png")
        return self.velocity_hat, self.magnetic_field_hat, self.diagnostics
