@echo off
rem Runs the milestone-5 step-node wrapper graph (solve_step_test.json) headlessly,
rem with the PFFlipSpike pack (which now also carries PFFlip::Solve::step_2d) on the
rem library path. Driven by validate_step.py.

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
if not exist "%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json" (
    call "%~dp0PFFlipSpike\build.bat" || exit /b 1
)
set "BIFROST_LIB_CONFIG_FILES=%~dp0PFFlipSpike\build\PFFlipSpike-1.0.0\PFFlipSpikePackConfig.json"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0solve_step_test.json" %*
