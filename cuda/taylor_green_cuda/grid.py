import cupy as cp


class SpectralGrid:
    def __init__(self, n, domain_length):
        self.n = n
        self.domain_length = domain_length
        k = (2.0 * cp.pi * cp.fft.fftfreq(n, d=domain_length / n)).astype(cp.float32)
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
        self.waves = cp.stack((kx, ky, kz))
        # Python 1j is complex128 and would upcast every spectral multiply.
        ik_dtype = cp.complex64 if kx.dtype == cp.float32 else cp.complex128
        self.ik = cp.array(1j, dtype=ik_dtype) * self.waves

    def to_spectral(self, field):
        field_hat = cp.fft.fftn(field, axes=(-3, -2, -1))
        field_hat *= self.dealias_mask
        return field_hat

    def to_real(self, field_hat):
        spatial = field_hat.shape[-3:]
        if field_hat.ndim > 3:
            batch_shape = field_hat.shape[:-3]
            flat = field_hat.reshape(-1, *spatial)
            return cp.fft.ifftn(flat, axes=(-3, -2, -1)).real.reshape(*batch_shape, *spatial)
        return cp.fft.ifftn(field_hat, axes=(-3, -2, -1)).real

    def dealias(self, field_hat):
        return field_hat * self.dealias_mask
