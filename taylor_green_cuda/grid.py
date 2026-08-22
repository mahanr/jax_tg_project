import cupy as cp


class SpectralGrid:
    def __init__(self, n, domain_length):
        self.n = n
        self.domain_length = domain_length
        self.k = 2.0 * cp.pi * cp.fft.fftfreq(n, d=domain_length / n)
        self.kx, self.ky, self.kz = cp.meshgrid(self.k, self.k, self.k, indexing="ij")
        self.k_squared = self.kx * self.kx + self.ky * self.ky + self.kz * self.kz
        cutoff = (n // 3) * (2.0 * cp.pi / domain_length)
        keep = cp.abs(self.k) <= cutoff
        self.dealias_mask = keep[:, None, None] & keep[None, :, None] & keep[None, None, :]
