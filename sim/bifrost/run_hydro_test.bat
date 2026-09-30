@echo off
rem Builds the PFFlipSpike Bifrost pack if needed and runs the two-phase hydrostatic
rem projection test headlessly through bifcmd. A correct run reports a max velocity
rem after projection near zero (the column returns to rest).

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
if not exist "%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json" (
    call "%~dp0PFFlipSpike\build.bat" || exit /b 1
)
set "BIFROST_LIB_CONFIG_FILES=%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0hydro_test.json" --set-port cells 256 --set-port density_ratio 1000
