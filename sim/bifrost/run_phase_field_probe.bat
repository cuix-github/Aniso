@echo off
rem Runs the milestone-4 phase-field probe graph headlessly through bifcmd.
rem The graph reads particle positions, per-particle phase and probe positions
rem from .npy files, splats phase into a volume with the STOCK
rem splat_points_into_volume node, samples the result at the probes, and writes
rem phi back as .npy. Driven by compare_phase_field.py, which also makes the
rem comparison figure against the 2D reference implementation.
rem
rem Usage: run_phase_field_probe.bat --set-port particles_npy <path> ... (any bifcmd args)

setlocal
set "BIFROST=C:\Program Files\Autodesk\Bifrost\Maya2027\3.1.0.8\bifrost"
set "PATH=%BIFROST%\bin;%BIFROST%\thirdparty\bin;%PATH%"
bifcmd "%~dp0phase_field_probe.json" %*
