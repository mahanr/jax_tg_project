"""Simulation driver for rotating shell convection."""
import time

import cupy as cp

from .config import ShellConvectionConfig
from .diagnostics import ShellDiagnostics
from .geometry import ShellGeometry
from .initial_conditions import ShellInitialCondition
from .integrator import ImexEulerIntegrator
from .operators import ShellOperators
from .plotting import save_plots


class ShellConvectionSimulation:
    def __init__(self, config=None, make_plots=True):
        self.config = config or ShellConvectionConfig()
        self.geometry = ShellGeometry(self.config)
        self.operators = ShellOperators(self.geometry, self.config)
        self.integrator = ImexEulerIntegrator(self.operators, self.config.dt)
        ic = ShellInitialCondition(self.geometry)
        self.w, self.z, self.theta = ic.create()
        self.diagnostics = ShellDiagnostics(self.operators)
        self.make_plots = make_plots

    def record(self, step):
        time_value = step * self.config.dt
        if step == self.config.steps:
            time_value = self.config.total_time
        ke, nu_i = self.diagnostics.record(
            time_value, self.w, self.z, self.theta
        )
        print(
            f"Save step: {step}, time: {time_value:.6f}, "
            f"KE: {float(cp.asnumpy(ke).real):.8e}, Nu_inner: {nu_i:.4f}"
        )
        return ke, nu_i

    def run(self):
        config = self.config
        self.diagnostics.record(0.0, self.w, self.z, self.theta)
        start = time.perf_counter()
        for step in range(1, config.steps + 1):
            self.w, self.z, self.theta = self.integrator.step(
                self.w, self.z, self.theta
            )
            if step % config.save_every == 0 or step == config.steps:
                self.record(step)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        print(f"CUDA device: {device_name}")
        print(f"Ra={config.ra:g}, Pr={config.pr:g}, Ek={config.ek:g}, eta={config.eta:g}")
        print(
            f"Simulated time: {config.total_time:.3f} ({config.steps} steps) "
            f"in {elapsed:.3f} s"
        )
        if self.diagnostics.kinetic_energies:
            print(
                f"KE trace: {self.diagnostics.kinetic_energies[0]:.6e} -> "
                f"{self.diagnostics.kinetic_energies[-1]:.6e}"
            )
        if self.make_plots:
            save_plots(self.w, self.z, self.theta, self.diagnostics, self.geometry)
        return self.w, self.z, self.theta, self.diagnostics
