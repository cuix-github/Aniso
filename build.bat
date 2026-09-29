@echo off
rem Configures and builds Aniso in Release with Visual Studio 2022's bundled CMake, then runs
rem the tests. Run from anywhere; it works in the folder this script lives in.

setlocal
cd /d "%~dp0"
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"

if not exist build\CMakeCache.txt (
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 || exit /b 1
)
cmake --build build --config Release || exit /b 1
ctest --test-dir build -C Release --output-on-failure || exit /b 1
echo.
echo Built: build\Release\aniso.exe
