"""Builds a ready-to-open Maya scene around the phase_field_play compound: a Bifrost
graph wired so Maya's timeline drives the frame port and the viewport shows the
two-phase dam break's phase field as fog. Saves out/play_dam_break.ma.

Run headlessly (or from Maya's script editor, where it skips standalone init):

  "C:\\Program Files\\Autodesk\\Maya2027\\bin\\mayapy.exe" build_maya_play_scene.py

Needs out/seq/ from export_play_sequence.py first. The scene references the compound
by name, so Maya must be started with BIFROST_LIB_CONFIG_FILES pointing at
pfflip_graphs_config.json (play_in_maya.bat does all of this in order).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "pfflip_graphs_config.json")
SEQ = os.path.join(HERE, "out", "seq")
SCENE = os.path.join(HERE, "out", "play_dam_break.ma")

os.environ["BIFROST_LIB_CONFIG_FILES"] = CONFIG
os.environ.setdefault("MAYA_DISABLE_CER", "1")
os.environ.setdefault("MAYA_DISABLE_CIP", "1")
os.environ.setdefault("MAYA_DISABLE_CLIC_IPM", "1")

STANDALONE = "maya.standalone" not in sys.modules and not os.environ.get("MAYA_APP_DIR_SET_BY_MAYA")


def main():
    import maya.standalone
    try:
        maya.standalone.initialize(name="python")
        standalone = True
    except Exception:
        standalone = False  # already inside Maya
    import maya.cmds as cmds

    if not cmds.pluginInfo("bifrostGraph", q=True, loaded=True):
        cmds.loadPlugin("bifrostGraph")

    cmds.file(new=True, force=True)
    shape = cmds.createNode("bifrostGraphShape", name="phaseFieldPlayShape")

    node = cmds.vnnCompound(
        shape, "/", addNode="PFFlipGraphs,User::PFFlip,phase_field_play")[0]
    cmds.vnnNode(shape, "/input", createOutputPort=("frame", "float"))
    cmds.vnnNode(shape, "/output", createInputPort=("view_volume", "Object"))
    cmds.vnnConnect(shape, "/input.frame", "/" + node + ".frame")
    cmds.vnnConnect(shape, "/" + node + ".view_volume", "/output.view_volume")
    for port, name in (("particles_pattern", "particles.####"),
                       ("phase_pattern", "phase.####")):
        cmds.vnnNode(shape, "/" + node, setPortDefaultValues=(
            port, os.path.join(SEQ, name).replace("\\", "/")))

    cmds.connectAttr("time1.outTime", shape + ".frame")
    cmds.playbackOptions(min=1, max=60, ast=1, aet=60)
    cmds.currentTime(1)

    cmds.file(rename=SCENE)
    cmds.file(save=True, type="mayaAscii")
    print("saved", SCENE)
    print("compound node in graph:", node)

    if standalone:
        sys.stdout.flush()
        import ctypes
        ctypes.windll.kernel32.TerminateProcess(
            ctypes.windll.kernel32.GetCurrentProcess(), 0)


if __name__ == "__main__":
    main()
