@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul
set "BIFROST_LOCATION=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Users\5d149\source\Aniso\.venv\Scripts;%PATH%"
cd /d "%~dp0"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build build || exit /b 1
cmake --install build >nul || exit /b 1
echo BUILD OK
