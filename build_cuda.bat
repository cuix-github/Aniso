@echo off
rem Builds Aniso with the CUDA renderer into build-cuda\, using the project-local CUDA 12.8
rem toolkit (run python pipeline\setup_toolchain.py once first) and Ninja, then runs the tests.

setlocal
cd /d "%~dp0"
if not exist .toolchain\cuda_env.bat (
    echo Run "python pipeline\setup_toolchain.py" first.
    exit /b 1
)
call .toolchain\cuda_env.bat
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%~dp0.venv\Scripts;%PATH%"

if not exist build-cuda\CMakeCache.txt (
    cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DANISO_CUDA=ON ^
        -DCMAKE_CUDA_COMPILER="%CUDA_HOME%\bin\nvcc.exe" -DCUDAToolkit_ROOT="%CUDA_HOME%" || exit /b 1
)
cmake --build build-cuda || exit /b 1
ctest --test-dir build-cuda --output-on-failure || exit /b 1
echo.
echo Built: build-cuda\aniso.exe and build-cuda\aniso_viewer.exe, with the CUDA renderer
