# Taylor-Green vortex DNS

3D decaying Taylor-Green vortex solvers in two implementations:

- **`jax/`** — JAX pseudo-spectral solver (GPU via XLA)
- **`cuda/`** — Native CUDA/cuFFT solver (`taylor_green_cuda.cu`), CuPy HD package, and CuPy MHD package

## Quick start

**JAX** (from `jax/`):

```bash
cd jax
python taylor_green_jax.py
python -m unittest test_taylor_green_jax.py
```

**CUDA HD** — see [cuda/README_CUDA.md](cuda/README_CUDA.md) for CuPy and native `nvcc` builds.

**CUDA MHD** — see [cuda/README_MHD.md](cuda/README_MHD.md) for the Taylor-Green MHD CuPy solver.

**Rotating shell convection** — see [cuda/README_SHELL.md](cuda/README_SHELL.md) for Boussinesq convection in a spherical shell.
