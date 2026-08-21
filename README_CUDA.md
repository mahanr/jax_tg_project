# CUDA Taylor-Green solvers

## Python/CuPy version

The recommended Python implementation is `taylor_green_cuda.py`. It uses CuPy,
which calls NVIDIA CUDA and cuFFT underneath. Install the CuPy wheel matching
the CUDA major version installed on Windows:

```powershell
python -m pip install cupy-cuda12x matplotlib
```

For CUDA 11, use `cupy-cuda11x` instead. Verify the GPU and run a T4-friendly
case with:

```powershell
python taylor_green_cuda.py --n 128 --dt 0.005 --reynolds 1000 --total-time 1.0 --save-every-time 0.05
```

The Python solver prints saved time steps and enstrophy, and writes
`taylor_green_cuda_slice.png` and `taylor_green_cuda_diagnostics.png`.

`taylor_green_cuda.cu` is a standalone CUDA/cuFFT implementation of the 3D decaying Taylor-Green vortex. It uses `float32`, RK4 time integration, Fourier 2/3 dealiasing, Leray projection, and prints saved-step enstrophy and kinetic energy.

## Windows build

1. Install the NVIDIA CUDA Toolkit, a supported Visual Studio version, and a CUDA-capable driver.
2. Open **x64 Native Tools Command Prompt for VS** or **Developer PowerShell for VS**.
3. Change to this repository directory and compile:

```powershell
nvcc -O3 -arch=sm_75 -o taylor_green_cuda.exe taylor_green_cuda.cu -lcufft
```

`sm_75` targets the T4. For another GPU, replace it with that GPU's compute capability.

Run the default T4-friendly case:

```powershell
.\taylor_green_cuda.exe 128 0.005 1000 1.0 0.05
```

Arguments are:

```text
N dt Reynolds total_time save_time
```

For the requested diffusion time, pass its numeric value or use PowerShell arithmetic:

```powershell
.\taylor_green_cuda.exe 256 0.005 1000 (10 * 2 * 3.141592653589793 * 1000) 0.05
```

The long run requires approximately 12,566,370 steps at `dt=0.005`, so test a short duration first.
