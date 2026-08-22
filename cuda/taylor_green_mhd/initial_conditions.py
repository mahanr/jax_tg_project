import cupy as cp


class TaylorGreenMhdInitialCondition:
    def __init__(self, domain_length=2.0 * cp.pi, magnetic_amplitude=0.5):
        self.domain_length = domain_length
        self.magnetic_amplitude = magnetic_amplitude

    def create(self, n):
        x = cp.linspace(0.0, self.domain_length, n, endpoint=False, dtype=cp.float32)
        X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")
        u = cp.sin(X) * cp.cos(Y) * cp.cos(Z)
        v = -cp.cos(X) * cp.sin(Y) * cp.cos(Z)
        velocity = cp.stack((u, v, cp.zeros_like(u)))
        b_amp = cp.float32(self.magnetic_amplitude)
        magnetic_field = cp.stack((
            b_amp * u,
            b_amp * v,
            cp.zeros_like(u),
        ))
        return velocity, magnetic_field
