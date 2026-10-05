@echo off
rem Runs the milestone-5 simulation-loop graph (sim_2d.json) headlessly: one bifcmd
rem invocation executes the whole 2D PF-FLIP run via an iterate around the custom
rem step node, dumping per-substep .npy frames. Driven by validate_loop.py.

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
if not exist "%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json" (
    call "%~dp0PFFlipSpike\build.bat" || exit /b 1
)
set "BIFROST_LIB_CONFIG_FILES=%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0sim_2d.json" %*
