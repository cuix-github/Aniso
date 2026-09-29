"""Builds a synthetic multi-view dataset of the Pixar Kitchen Set in Maya, for training
Gaussian splats: import the USD, shade and light it, place cameras, save a Maya scene that
renders every camera, and export the cameras and starting points in COLMAP's binary format.

Run with Maya's Python, headless:

    mayapy pipeline/maya/kitchen.py <Kitchen_set.usd> <out-dir> [--views 320] [--seed 7]

Then render every camera with Maya Hardware 2.0 (pipeline/maya/render_kitchen.py does both).

Output layout, which trainers such as gsplat read directly:
    <out-dir>/kitchen.mb           the scene, one renderable camera per view
    <out-dir>/sparse/0/cameras.bin one shared pinhole camera
    <out-dir>/sparse/0/images.bin  one pose per view, named view_000.png ...
    <out-dir>/sparse/0/points3D.bin  starting points sampled from the mesh vertices, coloured
    <out-dir>/images/              filled by the render step

Coordinates are Maya's world (Y up) converted from centimetres to metres. Cameras follow the
COLMAP convention: x right, y down, z forward.

Rendering uses Hardware 2.0 rather than Arnold: Arnold batch rendering needs its own
authorization, which the Maya login used here does not provide.
"""
import argparse
import math
import os
import random
import struct
import sys

import maya.standalone

maya.standalone.initialize(name="python")

import maya.api.OpenMaya as om  # noqa: E402
import maya.cmds as cmds  # noqa: E402

WIDTH, HEIGHT = 960, 540
FOCAL_MM, APERTURE_MM = 24.0, 36.0  # horizontal fit: fx = WIDTH * FOCAL / APERTURE
CM_TO_M = 0.01


def hard_exit(code=0):
    """Maya crashes during its own shutdown; end the process directly instead."""
    import ctypes

    sys.stdout.flush()
    sys.stderr.flush()
    ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), code)


def import_kitchen(usd_path):
    cmds.loadPlugin("mayaUsdPlugin", quiet=True)
    cmds.mayaUSDImport(file=usd_path, shadingMode=[["displayColor", "none"]], readAnimData=False)
    return cmds.ls(type="mesh", noIntermediate=True, long=True)


def reshade(meshes):
    """Replace each imported lambert (display colour only) with a blinn of the same colour and a
    soft highlight, so the scene has some view-dependent appearance for the splats to learn."""
    done = {}
    for sg in cmds.ls(type="shadingEngine"):
        if sg == "initialShadingGroup":
            continue
        src = cmds.listConnections(sg + ".surfaceShader") or []
        if not src or cmds.nodeType(src[0]) != "lambert":
            continue
        colour = cmds.getAttr(src[0] + ".color")[0]
        key = tuple(round(c, 4) for c in colour)
        if key not in done:
            b = cmds.shadingNode("blinn", asShader=True, name="kitchenBlinn#")
            cmds.setAttr(b + ".color", *colour, type="double3")
            cmds.setAttr(b + ".diffuse", 0.9)
            cmds.setAttr(b + ".eccentricity", 0.35)
            cmds.setAttr(b + ".specularRollOff", 0.5)
            cmds.setAttr(b + ".specularColor", 0.22, 0.22, 0.22, type="double3")
            done[key] = b
        cmds.connectAttr(done[key] + ".outColor", sg + ".surfaceShader", force=True)
    return len(done)


def bounds(nodes):
    b = cmds.exactWorldBoundingBox(nodes)
    return (b[0], b[1], b[2]), (b[3], b[4], b[5])


def light(room_lo, room_hi):
    """One fixed setup: soft ambient, a sun-like key through the room at an angle with shadow
    maps, and point lights at the ceiling fixtures."""
    amb = cmds.ambientLight(intensity=0.36, ambientShade=0.0)
    key = cmds.directionalLight(intensity=0.85, rgb=(1.0, 0.95, 0.88))
    key_t = cmds.listRelatives(key, parent=True)[0]
    cmds.xform(key_t, ws=True, ro=(-55, 30, 0))
    cmds.setAttr(key + ".useDepthMapShadows", 1)
    cmds.setAttr(key + ".dmapResolution", 4096)
    fill = cmds.directionalLight(intensity=0.25, rgb=(0.85, 0.9, 1.0))
    cmds.xform(cmds.listRelatives(fill, parent=True)[0], ws=True, ro=(-30, -150, 0))

    fixtures = [t for t in cmds.ls(type="transform", long=True) if t.split("|")[-1].startswith("CeilingLight_")]
    for f in fixtures:
        (lo, hi) = bounds(f)
        p = cmds.pointLight(intensity=0.35, rgb=(1.0, 0.9, 0.75))
        cmds.xform(cmds.listRelatives(p, parent=True)[0], ws=True,
                   t=((lo[0] + hi[0]) / 2, lo[1] - 8, (lo[2] + hi[2]) / 2))
        cmds.setAttr(p + ".decayRate", 0)

    g = "hardwareRenderingGlobals"
    cmds.setAttr(g + ".lightingMode", 1)  # all scene lights
    cmds.setAttr(g + ".ssaoEnable", 1)
    cmds.setAttr(g + ".ssaoAmount", 1.2)
    cmds.setAttr(g + ".ssaoRadius", 24)
    cmds.setAttr(g + ".ssaoSamples", 32)
    cmds.setAttr(g + ".multiSampleEnable", 1)
    cmds.setAttr(g + ".multiSampleCount", 16)
    cmds.setAttr(g + ".motionBlurEnable", 0)
    return len(fixtures)


def occupied_boxes():
    """Bounding boxes of every prop, used to keep cameras out of solid objects."""
    boxes = []
    for m in cmds.ls(type="mesh", noIntermediate=True, long=True):
        t = cmds.listRelatives(m, parent=True, fullPath=True)[0]
        if "|Arch_grp|" in t:
            continue
        lo, hi = bounds(t)
        boxes.append((lo, hi))
    return boxes


def inside_any(p, boxes, margin):
    for lo, hi in boxes:
        if all(lo[k] - margin <= p[k] <= hi[k] + margin for k in range(3)):
            return True
    return False


def ray_mesh(meshes):
    """One merged copy of the whole kitchen, for fast ray casts when judging camera views."""
    transforms = list({cmds.listRelatives(m, parent=True, fullPath=True)[0] for m in meshes})
    dup = cmds.duplicate(transforms, returnRootsOnly=True)
    merged = cmds.polyUnite(dup, constructionHistory=False, name="rayMesh")[0]
    fn = om.MFnMesh(om.MSelectionList().add(merged).getDagPath(0))
    return merged, fn, fn.autoUniformGridParams()


def view_is_clear(fn, accel, pos, tgt):
    """Casts nine rays across the central part of the view. Rejects views blocked by something
    closer than 60 cm, and views that look out through the open sides of the set."""
    f = om.MVector(*[tgt[k] - pos[k] for k in range(3)]).normal()
    right = (f ^ om.MVector(0, 1, 0)).normal()
    up = right ^ f
    src = om.MFloatPoint(*pos)
    misses = 0
    for a in (-0.35, 0.0, 0.35):
        for b in (-0.2, 0.0, 0.2):
            d = (f + right * a + up * b).normal()
            hit = fn.closestIntersection(src, om.MFloatVector(d), om.MSpace.kWorld, 2000.0, False,
                                         accelParams=accel)
            if not hit or not hit[3]:
                misses += 1
                continue
            if hit[1] < 60.0:
                return False
    return misses <= 2


def place_cameras(n, room_lo, room_hi, boxes, seed, fn, accel):
    """Cameras stand inside the room at human-ish heights and look at points across the room,
    never from inside a prop, never at a target closer than 1.2 m, never with a nearby object
    filling the view or with the view looking out of the set."""
    rng = random.Random(seed)
    m = 45.0
    lo = (room_lo[0] + m, room_lo[1], room_lo[2] + m)
    hi = (room_hi[0] - m, room_hi[1], room_hi[2] - m)
    cams = []
    tries = 0
    while len(cams) < n and tries < n * 200:
        tries += 1
        pos = (rng.uniform(lo[0], hi[0]), room_lo[1] + rng.uniform(80, 200), rng.uniform(lo[2], hi[2]))
        if inside_any(pos, boxes, 15):
            continue
        tgt = (rng.uniform(lo[0], hi[0]), room_lo[1] + rng.uniform(20, 190), rng.uniform(lo[2], hi[2]))
        if math.dist(pos, tgt) < 120:
            continue
        if not view_is_clear(fn, accel, pos, tgt):
            continue
        cams.append((pos, tgt))
    return cams


def make_camera(i, pos, tgt):
    t, s = cmds.camera(focalLength=FOCAL_MM, horizontalFilmAperture=APERTURE_MM / 25.4,
                       verticalFilmAperture=APERTURE_MM / 25.4 * HEIGHT / WIDTH, filmFit="horizontal",
                       nearClipPlane=1.0, farClipPlane=5000.0)
    t = cmds.rename(t, "view_%03d" % i)
    s = cmds.listRelatives(t, shapes=True)[0]
    cmds.xform(t, ws=True, t=pos)
    cmds.viewPlace(s, eye=pos, lookAt=tgt, up=(0, 1, 0))
    cmds.setAttr(s + ".renderable", 1)
    return t, s


def colmap_pose(transform):
    """World-to-camera rotation (as a quaternion w, x, y, z) and translation in COLMAP's
    convention, in metres."""
    m = om.MMatrix(cmds.xform(transform, q=True, ws=True, matrix=True))
    x = [m[0], m[1], m[2]]
    y = [m[4], m[5], m[6]]
    z = [m[8], m[9], m[10]]
    c = [m[12] * CM_TO_M, m[13] * CM_TO_M, m[14] * CM_TO_M]
    # Maya cameras look down -z with +y up; COLMAP looks down +z with +y down.
    rows = [x, [-v for v in y], [-v for v in z]]
    tvec = [-sum(rows[r][k] * c[k] for k in range(3)) for r in range(3)]
    q = om.MQuaternion()
    q.setValue(om.MMatrix([rows[0][0], rows[0][1], rows[0][2], 0,
                           rows[1][0], rows[1][1], rows[1][2], 0,
                           rows[2][0], rows[2][1], rows[2][2], 0,
                           0, 0, 0, 1]).transpose())
    return (q.w, q.x, q.y, q.z), tvec


def write_colmap(out, views, points):
    d = os.path.join(out, "sparse", "0")
    os.makedirs(d, exist_ok=True)
    fx = WIDTH * FOCAL_MM / APERTURE_MM
    with open(os.path.join(d, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, WIDTH, HEIGHT))  # id 1, model 1 = PINHOLE
        f.write(struct.pack("<4d", fx, fx, WIDTH / 2.0, HEIGHT / 2.0))
    with open(os.path.join(d, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(views)))
        for i, (name, q, t) in enumerate(views):
            f.write(struct.pack("<I4d3dI", i + 1, *q, *t, 1))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", 0))  # no 2D observations
    with open(os.path.join(d, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, (p, rgb) in enumerate(points):
            f.write(struct.pack("<Q3d3Bd", i + 1, *p, *rgb, 0.0))
            f.write(struct.pack("<Q", 0))  # empty track


def sample_points(meshes, limit, seed):
    """World-space mesh vertices with their shader colour, randomly thinned to `limit`."""
    pts = []
    for m in meshes:
        sg = (cmds.listConnections(m, type="shadingEngine") or ["initialShadingGroup"])[0]
        sh = (cmds.listConnections(sg + ".surfaceShader") or [None])[0]
        col = cmds.getAttr(sh + ".color")[0] if sh and cmds.attributeQuery("color", node=sh, exists=True) else (0.5, 0.5, 0.5)
        rgb = tuple(max(0, min(255, int(round(c ** (1 / 2.2) * 255)))) for c in col)
        fn = om.MFnMesh(om.MSelectionList().add(m).getDagPath(0))
        for v in fn.getPoints(om.MSpace.kWorld):
            pts.append(((v.x * CM_TO_M, v.y * CM_TO_M, v.z * CM_TO_M), rgb))
    random.Random(seed).shuffle(pts)
    return pts[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("out")
    ap.add_argument("--views", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--points", type=int, default=150000)
    a = ap.parse_args(sys.argv[1:])
    a.out = os.path.abspath(a.out).replace("\\", "/")
    a.usd = os.path.abspath(a.usd).replace("\\", "/")
    os.makedirs(a.out, exist_ok=True)

    meshes = import_kitchen(a.usd)
    shaders = reshade(meshes)
    arch = [t for t in cmds.ls("Kitchen_set|Arch_grp", long=True)]
    room_lo, room_hi = bounds(arch)
    fixtures = light(room_lo, room_hi)
    boxes = occupied_boxes()
    merged, fn, accel = ray_mesh(meshes)
    placements = place_cameras(a.views, room_lo, room_hi, boxes, a.seed, fn, accel)
    cmds.delete(merged)

    for cam in cmds.ls(type="camera"):
        cmds.setAttr(cam + ".renderable", 0)
    views = []
    for i, (pos, tgt) in enumerate(placements):
        t, _ = make_camera(i, pos, tgt)
        q, tv = colmap_pose(t)
        views.append(("view_%03d.png" % i, q, tv))

    rg = "defaultRenderGlobals"
    cmds.setAttr(rg + ".currentRenderer", "mayaHardware2", type="string")
    cmds.setAttr(rg + ".imageFormat", 32)  # png
    cmds.setAttr(rg + ".imageFilePrefix", "<Camera>", type="string")
    cmds.setAttr("defaultResolution.width", WIDTH)
    cmds.setAttr("defaultResolution.height", HEIGHT)
    cmds.setAttr("defaultResolution.deviceAspectRatio", WIDTH / HEIGHT)

    cmds.file(rename=os.path.join(a.out, "kitchen.mb"))
    cmds.file(save=True, type="mayaBinary")

    points = sample_points(meshes, a.points, a.seed)
    write_colmap(a.out, views, points)
    print("meshes %d, shaders %d, ceiling fixtures %d, views %d, points %d" %
          (len(meshes), shaders, fixtures, len(views), len(points)))
    print("room bounds (cm)", [round(v, 1) for v in room_lo], [round(v, 1) for v in room_hi])
    hard_exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # make failures visible, then exit without Maya's crash dialog
        import traceback

        traceback.print_exc()
        hard_exit(1)
