@echo off
rem One click from repo to Maya: exports the 2D dam-break sequence if missing,
rem builds the Bifrost play scene if missing, then opens Maya on it with the
rem PF-FLIP compound library on BIFROST_LIB_CONFIG_FILES. Press play in Maya:
rem the viewport fog is the phase field of the two-phase dam break, frame by
rem frame the same simulation validated in sim/reference2d.

setlocal
set "HERE=%~dp0"
set "BIFROST_LIB_CONFIG_FILES=%HERE%pfflip_graphs_config.json"

if not exist "%HERE%out\seq\particles.0060.npy" (
    echo Exporting the dam-break sequence...
    "%HERE%..\..\.venv\Scripts\python.exe" "%HERE%export_play_sequence.py" || exit /b 1
)
if not exist "%HERE%out\play_dam_break.ma" (
    echo Building the Maya scene...
    "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" "%HERE%build_maya_play_scene.py" || exit /b 1
)
echo Opening Maya...
start "" "C:\Program Files\Autodesk\Maya2027\bin\maya.exe" "%HERE%out\play_dam_break.ma"
