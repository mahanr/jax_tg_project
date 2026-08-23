# Rotating Shell Convection (CuPy)

Non-dimensional Boussinesq rotating convection in a spherical shell.

## Equations

Reference scales: length `d` (gap), time `1/Ω`, velocity `dΩ`, rotation `Ω ê_z`.

```
∂u/∂t + (u·∇)u + 2 ê_z × u = −∇Π + Ek ∇²u + (Ra Ek²/Pr) θ ê_r
∇·u = 0
∂θ/∂t + u·∇θ = (Ek/Pr) ∇²θ
```

- `ν = Ek`, thermal diffusion `Ek/Pr`, buoyancy `Ra Ek²/Pr`
- Velocity is MagIC QST: `torpol_to_spat(W, ∂W/∂r, Z)` with
  `Q = ℓ(ℓ+1) W / r²`, `S = (∂W/∂r)/r`, `T = Z / r`
- Nonlinear momentum: grid force `F = −(u·∇)u − 2 ê_z×u + buoyancy ê_r`, then
  `spat_to_qst` of `curl curl F` (poloidal) and `curl F` (toroidal)
- Angular derivatives use `∂Y/∂θ` and `(im/sinθ)Y`, not finite differences
- BCs: no-slip `W = ∂W/∂r = Z = 0`; `θ = 1` at inner `r = η` (last Chebyshev
  node) and `θ = 0` at outer `r = 1` (first node)

## Install

```bash
python -m pip install cupy-cuda12x matplotlib scipy
```

## Run

```bash
cd cuda
python -m rotating_shell_convection --l-max 31 --nr 32 --ra 1e5 --pr 1 --ek 1e-4 --total-time 1.0 --no-plots
```

## Tests

```bash
cd cuda
python -m unittest test_rotating_shell_convection.py -v
```
