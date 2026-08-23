"""Non-dimensional incompressible Navier-Stokes configuration.

Governing equation on [0, 2π)³ with U_ref = 1:

    ∂u/∂t + (u·∇)u = -∇p + (1/Re)∇²u,  ∇·u = 0

Reynolds number Re = 1/ν is the sole viscous parameter.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TaylorGreenConfig:
    n: int = 128
    dt: float = 0.005
    reynolds: float = 1000.0
    total_time: float = 1.0
    save_every_time: float = 0.05
    domain_length: float = 2.0 * math.pi

    def __post_init__(self):
        if self.n < 4:
            raise ValueError("n must be at least 4")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.reynolds <= 0.0:
            raise ValueError("reynolds must be positive")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive")
        if self.save_every_time <= 0.0:
            raise ValueError("save_every_time must be positive")
        if self.domain_length <= 0.0:
            raise ValueError("domain_length must be positive")

    @property
    def viscosity(self):
        return 1.0 / self.reynolds

    @property
    def steps(self):
        return max(1, round(self.total_time / self.dt))

    @property
    def save_every(self):
        return max(1, int(self.save_every_time / self.dt))
