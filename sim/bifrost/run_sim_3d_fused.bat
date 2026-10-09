@echo off
rem Runs the milestone-6 3D simulation-loop graph (sim_3d_fused.json) headlessly: one
rem bifcmd invocation executes the whole 3D PF-FLIP run (fused-P2G node (one custom scatter pass) + the
rem multigrid step node inside an iterate). Driven by demo3d.py.

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
if not exist "%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json" (
    call "%~dp0PFFlipSpike\build.bat" || exit /b 1
)
set "BIFROST_LIB_CONFIG_FILES=%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0sim_3d_fused.json" %*
