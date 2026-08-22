import cupy as cp


class SpectralGrid:
    def __init__(self, n, domain_length):
        self.n = n
        self.domain_length = domain_length
        k = 2.0 * cp.pi * cp.fft.fftfreq(n, d=domain_length / n)
        kx, ky, kz = cp.meshgrid(k, k, k, indexing="ij")
        cutoff = (n // 3) * (2.0 * cp.pi / domain_length)
        keep = cp.abs(k) <= cutoff
        dealias_mask = keep[:, None, None] & keep[None, :, None] & keep[None, None, :]
        self._set_wavenumbers(kx, ky, kz, dealias_mask)

    @classmethod
    def from_arrays(cls, kx, ky, kz, dealias_mask):
        """Build a grid from precomputed wavenumber arrays (for shim APIs)."""
        grid = cls.__new__(cls)
        grid.n = int(kx.shape[0])
        grid.domain_length = None
        grid._set_wavenumbers(kx, ky, kz, dealias_mask)
        return grid

    def _set_wavenumbers(self, kx, ky, kz, dealias_mask):
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k_squared = kx * kx + ky * ky + kz * kz
        self.dealias_mask = dealias_mask

    def to_spectral(self, field):
        field_hat = cp.fft.fftn(field, axes=(-3, -2, -1))
        field_hat *= self.dealias_mask
        return field_hat

    def to_real(self, field_hat):
        return cp.fft.ifftn(field_hat, axes=(-3, -2, -1)).real

    def dealias(self, field_hat):
        return field_hat * self.dealias_mask
