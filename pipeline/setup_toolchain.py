"""Sets up everything the training pipeline needs, inside the repository and nowhere else:

  .toolchain/cuda-12.8/    NVIDIA's CUDA 12.8.1 components, from the official redistributable
                           archives, checksum-verified (compiler, runtime, headers, math libs)
  .toolchain/cuda_env.bat  sets up a cmd shell with that toolkit and MSVC 14.38
  .toolchain/run.bat       runs one command inside that shell
  .venv/                   Python with PyTorch (CUDA 12.8 build), gsplat, numpy, Pillow

Both folders are ignored by git. Nothing is installed system-wide. Run once from the repository
root with the system Python:

    python pipeline/setup_toolchain.py

Why not the prebuilt gsplat wheels: they stop at CUDA 12.4, and the RTX 5090 (sm_120) needs
CUDA 12.8 or newer. Why not NVIDIA's pip compiler package: on Windows it ships ptxas but not
nvcc. So gsplat compiles its CUDA code on first use, against this local toolkit.

One local patch is applied to gsplat 1.5.3: it passes the GCC-only flag -Wno-attributes to
MSVC, which rejects it; the patch skips that flag on Windows.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, ".toolchain")
CUDA = os.path.join(TC, "cuda-12.8")
VENV = os.path.join(ROOT, ".venv")
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvtx",
              "libcublas", "libcusparse", "libcusolver", "libcurand"]

CUDA_ENV = r"""@echo off
rem Sets up a shell for building CUDA code with the project-local CUDA 12.8 toolkit and
rem Visual Studio 2022's MSVC 14.38 toolset (a host compiler CUDA 12.8 supports).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul
set "CUDA_HOME=%~dp0cuda-12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set DISTUTILS_USE_SDK=1
rem Keep windows.h from defining "small" (rpcndr.h), which breaks a PyTorch CUDA header.
set "NVCC_APPEND_FLAGS=-DWIN32_LEAN_AND_MEAN -DNOMINMAX"
rem Build caches for PyTorch extensions stay inside the project.
set "TORCH_EXTENSIONS_DIR=%~dp0torch_extensions"
"""

RUN = "@echo off\r\ncall \"%~dp0cuda_env.bat\"\r\n%*\r\n"


def cuda_toolkit():
    if os.path.exists(os.path.join(CUDA, "bin", "nvcc.exe")):
        print("CUDA toolkit already present")
        return
    os.makedirs(CUDA, exist_ok=True)
    index = json.load(urllib.request.urlopen(REDIST + "redistrib_12.8.1.json"))
    for name in COMPONENTS:
        entry = index[name]["windows-x86_64"]
        archive = os.path.join(TC, os.path.basename(entry["relative_path"]))
        print("downloading", name)
        urllib.request.urlretrieve(REDIST + entry["relative_path"], archive)
        if hashlib.sha256(open(archive, "rb").read()).hexdigest() != entry["sha256"]:
            sys.exit(name + ": checksum mismatch")
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                rel = m.filename.split("/", 1)[1] if "/" in m.filename else ""
                if not rel or m.is_dir():
                    continue
                out = os.path.join(CUDA, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with z.open(m) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        os.remove(archive)


def scripts():
    with open(os.path.join(TC, "cuda_env.bat"), "w", newline="") as f:
        f.write(CUDA_ENV.replace("\n", "\r\n"))
    with open(os.path.join(TC, "run.bat"), "w", newline="") as f:
        f.write(RUN)


def python_env():
    py = os.path.join(VENV, "Scripts", "python.exe")
    if not os.path.exists(py):
        subprocess.check_call([sys.executable, "-m", "venv", VENV])
    pip = [py, "-m", "pip", "install", "-q"]
    subprocess.check_call(pip + ["--upgrade", "pip"])
    subprocess.check_call(pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu128"])
    subprocess.check_call(pip + ["numpy", "pillow", "ninja", "packaging", "setuptools", "wheel", "gsplat==1.5.3"])

    backend = os.path.join(VENV, "Lib", "site-packages", "gsplat", "cuda", "_backend.py")
    text = open(backend, encoding="utf-8").read()
    old = 'extra_cflags = [opt_level, "-Wno-attributes"]'
    new = 'extra_cflags = [opt_level] if os.name == "nt" else [opt_level, "-Wno-attributes"]'
    if old in text:
        open(backend, "w", encoding="utf-8").write(text.replace(old, new))
        print("patched gsplat for MSVC")


if __name__ == "__main__":
    os.makedirs(TC, exist_ok=True)
    cuda_toolkit()
    scripts()
    python_env()
    print("done. First gsplat use compiles its CUDA code (about a minute).")
