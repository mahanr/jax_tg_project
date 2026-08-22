import cupy as cp


class TaylorGreenInitialCondition:
    def __init__(self, domain_length=2.0 * cp.pi):
        self.domain_length = domain_length

    def create(self, n):
        x = cp.linspace(0.0, self.domain_length, n, endpoint=False, dtype=cp.float32)
        X, Y, Z = cp.meshgrid(x, x, x, indexing="ij")
        u = cp.sin(X) * cp.cos(Y) * cp.cos(Z)
        v = -cp.cos(X) * cp.sin(Y) * cp.cos(Z)
        return cp.stack((u, v, cp.zeros_like(u)))
