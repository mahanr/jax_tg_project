import time

import cupy as cp

from .config import TaylorGreenConfig
from .diagnostics import Diagnostics
from .grid import SpectralGrid
from .initial_conditions import TaylorGreenInitialCondition
from .integrator import RK4Integrator
from .operators import CupySpectralOperator
from .plotting import save_plots


class TaylorGreenSimulation:
    def __init__(self, config=None, make_plots=True):
        self.config = config or TaylorGreenConfig()
        self.grid = SpectralGrid(self.config.n, self.config.domain_length)
        self.initial_condition = TaylorGreenInitialCondition(self.config.domain_length)
        self.operator = CupySpectralOperator(self.grid)
        self.integrator = RK4Integrator(self.operator, self.config.dt, self.config.viscosity)
        self.state = self.grid.to_spectral(self.initial_condition.create(self.config.n))
        self.diagnostics = Diagnostics(self.operator)
        self.make_plots = make_plots

    def record(self, step):
        time_value = step * self.config.dt
        if step == self.config.steps:
            time_value = self.config.total_time
        energy, value = self.diagnostics.record(time_value, self.state)
        now = time.perf_counter()
        interval = now - self._last_record_wall
        self._last_record_wall = now
        print(f"Save time step: {step}, time: {time_value:.6f}, "
              f"enstrophy: {value:.8e}, interval: {interval:.2f} s")
        return energy, value

    def run(self):
        config = self.config
        self._last_record_wall = time.perf_counter()
        self.diagnostics.record(0.0, self.state)
        start = time.perf_counter()
        self._last_record_wall = start
        for step in range(1, config.steps + 1):
            self.state = self.integrator.step(self.state)
            self.state = self.grid.dealias(self.state)
            if step % config.save_every == 0 or step == config.steps:
                self.record(step)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode()
        print(f"CUDA device: {device_name}")
        print(f"Reynolds number: {config.reynolds:.1f}")
        print(f"Simulated time: {config.total_time:.3f} ({config.steps} steps) in {elapsed:.3f} s")
        print(f"Energy trace: {self.diagnostics.energies[0]:.6f} -> "
              f"{self.diagnostics.energies[-1]:.6f}")
        if self.make_plots:
            save_plots(self.state, self.diagnostics, config.n, self.grid, config.total_time)
        return self.state, self.diagnostics
