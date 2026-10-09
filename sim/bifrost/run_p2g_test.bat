@echo off
rem Runs the fused-P2G wrapper graph (p2g_test.json) headlessly.
rem Driven by validate_p2g.py.

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
if not exist "%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json" (
    call "%~dp0PFFlipSpike\build.bat" || exit /b 1
)
set "BIFROST_LIB_CONFIG_FILES=%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0p2g_test.json" %*
