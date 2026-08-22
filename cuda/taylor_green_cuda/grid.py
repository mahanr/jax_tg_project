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

    def to_spectral(self, field):
        field_hat = cp.fft.fftn(field, axes=(-3, -2, -1))
        field_hat *= self.dealias_mask
        return field_hat

    def to_real(self, field_hat):
        return cp.fft.ifftn(field_hat, axes=(-3, -2, -1)).real

    def dealias(self, field_hat):
        return field_hat * self.dealias_mask


class SpectralGridView:
    """Grid view built from precomputed wavenumber arrays (for shim APIs)."""

    def __init__(self, kx, ky, kz, dealias_mask):
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
