# Taylor-Green MHD (CuPy)

Resistive incompressible MHD Taylor-Green vortex solver using CuPy on NVIDIA GPUs.

## Equations

Incompressible MHD with kinematic viscosity `ν` and magnetic resistivity `η`:

- Momentum: `∂u/∂t + (u·∇)u = -∇p + (∇×B)×B + ν∇²u`, `∇·u = 0`
- Induction: `∂B/∂t = ∇×(u×B) + η∇²B`, `∇·B = 0`

Parameters follow the HD convention: `ν = L/Re`, `η = L/Rm` with `L = 2π`.

## Install

```powershell
python -m pip install cupy-cuda12x matplotlib
```

Use `cupy-cuda11x` for CUDA 11.

## Run

From the `cuda` directory:

```powershell
cd cuda
python -m taylor_green_mhd --n 128 --dt 0.005 --reynolds 1000 --magnetic-reynolds 1000 --total-time 1.0 --save-every-time 0.05
```

Or via the compatibility shim:

```powershell
python taylor_green_mhd.py --n 128 --dt 0.005 --reynolds 1000 --magnetic-reynolds 1000 --total-time 1.0
```

CLI flags:

| Flag | Default | Description |
|---|---|---|
| `--n` | 128 | Grid resolution |
| `--dt` | 0.005 | Timestep |
| `--reynolds` | 1000 | Fluid Reynolds number |
| `--magnetic-reynolds` | 1000 | Magnetic Reynolds number |
| `--magnetic-amplitude` | 0.5 | Initial `B` amplitude relative to `u_amp = 1` |
| `--total-time` | 1.0 | Simulation duration |
| `--save-every-time` | 0.05 | Diagnostic sampling interval |
| `--no-plots` | off | Skip PNG output |

## Output

- `taylor_green_mhd_slice.png` — mid-plane velocity and magnetic field slices
- `taylor_green_mhd_diagnostics.png` — kinetic, magnetic, total energy, and cross helicity

## Tests

```powershell
cd cuda
python -m unittest test_taylor_green_mhd.py -v
```

## Package layout

```
taylor_green_mhd/
  config.py           # TaylorGreenMhdConfig, timestep validation
  operators.py        # MhdSpectralOperator (Lorentz, induction, projections)
  integrator.py       # MhdRK4Integrator
  initial_conditions.py
  diagnostics.py
  simulation.py
  plotting.py
  cli.py
```

Reuses `SpectralGrid` from `taylor_green_cuda.grid`.
