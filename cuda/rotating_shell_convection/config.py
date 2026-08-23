"""Non-dimensional rotating Boussinesq convection in a spherical shell.

Reference scales: length d (gap), time 1/Omega, velocity d*Omega, rotation Omega e_z.

Governing equations (nondimensional):

    d u/dt + (u·∇)u + 2 e_z × u = -∇Π + Ek ∇²u + (Ra Ek²/Pr) θ e_r
    ∇·u = 0
    dθ/dt + u·∇θ = (Ek/Pr) ∇²θ
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShellConvectionConfig:
    ra: float = 1e6
    pr: float = 1.0
    ek: float = 1e-5
    eta: float = 0.5
    l_max: int = 31
    nr: int = 64
    dt: float = 1e-3
    total_time: float = 10.0
    save_every_time: float = 0.5

    def __post_init__(self):
        if self.ra <= 0.0:
            raise ValueError("ra must be positive")
        if self.pr <= 0.0:
            raise ValueError("pr must be positive")
        if self.ek <= 0.0:
            raise ValueError("ek must be positive")
        if not 0.0 < self.eta < 1.0:
            raise ValueError("eta must be between 0 and 1")
        if self.l_max < 1:
            raise ValueError("l_max must be at least 1")
        if self.nr < 4:
            raise ValueError("nr must be at least 4")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.total_time <= 0.0:
            raise ValueError("total_time must be positive")
        if self.save_every_time <= 0.0:
            raise ValueError("save_every_time must be positive")

    @property
    def viscous_coeff(self) -> float:
        return self.ek

    @property
    def thermal_diffusion_coeff(self) -> float:
        return self.ek / self.pr

    @property
    def buoyancy_coeff(self) -> float:
        return self.ra * self.ek**2 / self.pr

    @property
    def coriolis_coeff(self) -> float:
        return 2.0

    @property
    def n_phi(self) -> int:
        return 2 * (self.l_max + 1)

    @property
    def n_theta(self) -> int:
        return self.l_max + 1

    @property
    def n_lm(self) -> int:
        return (self.l_max + 1) ** 2

    @property
    def steps(self) -> int:
        return max(1, round(self.total_time / self.dt))

    @property
    def save_every(self) -> int:
        return max(1, int(self.save_every_time / self.dt))

    @property
    def r_inner(self) -> float:
        return self.eta

    @property
    def r_outer(self) -> float:
        return 1.0
