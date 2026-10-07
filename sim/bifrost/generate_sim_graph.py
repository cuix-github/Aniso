"""Generates sim_2d.json: the milestone-5 simulation loop as a Bifrost graph.

One bifcmd run executes the whole 2D PF-FLIP simulation: an iterate drives N substeps;
each substep rebuilds the particles (stock), splats mass / momentum / phase onto the two
MAC face grids via the shifted-points trick (stock splat_points_into_volume, one property
per node), samples the channels at face centres (stock), and hands them to the custom
PFFlip::Solve::step_2d node, whose outputs are the iterate's state for the next substep.
Per-substep particle positions and diagnostics are dumped as frame-numbered .npy files.

Regenerate after editing:  python generate_sim_graph.py
"""
import json
import os


def P(n, d, t=None, default=None, **extra):
    p = {"portName": n, "portDirection": d}
    if t:
        p["portType"] = t
    if default is not None:
        p["portDefault"] = default
    p.update(extra)
    return p


def build():
    body_nodes = [
        {"nodeName": "cp", "nodeType": "Geometry::Points::construct_points"},
        {"nodeName": "set_m", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "set_mx", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "set_my", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "set_ph", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "zero", "valueType": "float"},
        {"nodeName": "step", "nodeType": "PFFlip::Solve::step_2d"},
        {"nodeName": "w_pos", "nodeType": "File::NumPy::write_NumPy"},
        {"nodeName": "w_diag", "nodeType": "File::NumPy::write_NumPy"},
        {"nodeName": "diag_arr", "nodeType": "Core::Array::build_array",
         "multiInPortNames": ["a", "b", "c", "d"]},
        {"nodeName": "it_f", "nodeType": "Core::Type_Conversion::to_float"},
        {"nodeName": "ok_fp", "nodeType": "Core::Type_Conversion::to_float"},
        {"nodeName": "ok_fd", "nodeType": "Core::Type_Conversion::to_float"},
        {"nodeName": "ok_acc", "nodeType": "Core::Math::add",
         "multiInPortNames": ["s", "p", "q"]},
    ]
    body_conns = [
        {"source": ".positions_in", "target": "cp.point_position"},
        {"source": "cp.points", "target": "set_m.geometry"},
        {"source": "set_m.out_geometry", "target": "set_mx.geometry"},
        {"source": "set_mx.out_geometry", "target": "set_my.geometry"},
        {"source": "set_my.out_geometry", "target": "set_ph.geometry"},
        {"source": ".mass_in", "target": "set_m.data"},
        {"source": ".momx_in", "target": "set_mx.data"},
        {"source": ".momy_in", "target": "set_my.data"},
        {"source": ".particle_phase", "target": "set_ph.data"},
    ]
    for s in ("set_m", "set_mx", "set_my", "set_ph"):
        body_conns.append({"source": "zero.output", "target": s + ".default"})
    body_values = [
        {"valueName": "zero.value", "valueType": "float", "value": "0f"},
        {"valueName": "set_m.property", "valueType": "string", "value": "voxel_m"},
        {"valueName": "set_mx.property", "valueType": "string", "value": "voxel_mx"},
        {"valueName": "set_my.property", "valueType": "string", "value": "voxel_my"},
        {"valueName": "set_ph.property", "valueType": "string", "value": "voxel_ph"},
    ]
    grids = (("u", {"x": "0.5f", "y": "0f", "z": "0f"}, "voxel_mx"),
             ("v", {"x": "0f", "y": "0.5f", "z": "0f"}, "voxel_my"))
    for g, off, mom in grids:
        body_nodes += [
            {"nodeName": "off_" + g, "valueType": "Math::float3"},
            {"nodeName": "add_" + g, "nodeType": "Core::Math::add",
             "multiInPortNames": ["pts", "off"]},
            {"nodeName": "spos_" + g, "nodeType": "Geometry::Properties::set_geo_property_data"},
            {"nodeName": "p2v_" + g, "nodeType": "Geometry::Converters::points_to_volume"},
            {"nodeName": "sp1_" + g, "nodeType": "Geometry::Volume::splat_points_into_volume"},
            {"nodeName": "sp2_" + g, "nodeType": "Geometry::Volume::splat_points_into_volume"},
            {"nodeName": "sp3_" + g, "nodeType": "Geometry::Volume::splat_points_into_volume"},
            {"nodeName": "sm_" + g, "nodeType": "Geometry::Query::sample_volume"},
            {"nodeName": "so_" + g, "nodeType": "Geometry::Query::sample_volume"},
            {"nodeName": "sq_" + g, "nodeType": "Geometry::Query::sample_volume"},
        ]
        body_conns += [
            {"source": ".positions_in", "target": "add_" + g + ".first.pts"},
            {"source": "off_" + g + ".output", "target": "add_" + g + ".first.off"},
            {"source": "set_ph.out_geometry", "target": "spos_" + g + ".geometry"},
            {"source": "add_" + g + ".output", "target": "spos_" + g + ".data"},
            {"source": "spos_" + g + ".out_geometry", "target": "p2v_" + g + ".points"},
            {"source": "spos_" + g + ".out_geometry", "target": "sp1_" + g + ".points"},
            {"source": "spos_" + g + ".out_geometry", "target": "sp2_" + g + ".points"},
            {"source": "spos_" + g + ".out_geometry", "target": "sp3_" + g + ".points"},
            {"source": "p2v_" + g + ".volume", "target": "sp1_" + g + ".volume"},
            {"source": "sp1_" + g + ".out_volume", "target": "sp2_" + g + ".volume"},
            {"source": "sp2_" + g + ".out_volume", "target": "sp3_" + g + ".volume"},
            {"source": "sp3_" + g + ".out_volume", "target": "sm_" + g + ".volume"},
            {"source": "sp3_" + g + ".out_volume", "target": "so_" + g + ".volume"},
            {"source": "sp3_" + g + ".out_volume", "target": "sq_" + g + ".volume"},
            {"source": ".probes_" + g, "target": "sm_" + g + ".positions"},
            {"source": ".probes_" + g, "target": "so_" + g + ".positions"},
            {"source": ".probes_" + g, "target": "sq_" + g + ".positions"},
        ]
        for sp in ("sp1_", "sp2_", "sp3_"):
            body_conns += [
                {"source": ".radius", "target": sp + g + ".radius"},
                {"source": ".add_to_denominator", "target": sp + g + ".add_to_denominator"},
            ]
        for sm in ("sm_", "so_", "sq_"):
            body_conns.append({"source": ".sample_default", "target": sm + g + ".default"})
        body_values.append({"valueName": "off_" + g + ".value",
                            "valueType": "Math::float3", "value": off})
        for sp, prop in (("sp1_" + g, "voxel_m"), ("sp2_" + g, mom), ("sp3_" + g, "voxel_ph")):
            body_values += [
                {"valueName": sp + ".create_properties", "valueType": "bool", "value": "true"},
                {"valueName": sp + ".properties", "valueType": "string", "value": prop},
                {"valueName": sp + ".kernel",
                 "valueType": "Geometry::Volume::SplatKernelType", "value": "kLinearKernel"},
                {"valueName": sp + ".add_to_weights", "valueType": "float", "value": "0f"},
                {"valueName": sp + ".smoothing", "valueType": "float", "value": "0f"},
                {"valueName": sp + ".coarsest_depth", "valueType": "int", "value": "0"},
            ]
        body_values += [
            {"valueName": "p2v_" + g + ".detail_size", "valueType": "float", "value": "1f"},
            {"valueName": "p2v_" + g + ".properties", "valueType": "string", "value": ""},
            {"valueName": "p2v_" + g + ".resolution_mode",
             "valueType": "Geometry::Volume::ResolutionType", "value": "Absolute"},
            {"valueName": "sm_" + g + ".property", "valueType": "string", "value": "voxel_m"},
            {"valueName": "so_" + g + ".property", "valueType": "string", "value": mom},
            {"valueName": "sq_" + g + ".property", "valueType": "string", "value": "voxel_ph"},
            {"valueName": "sm_" + g + ".sampler",
             "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
            {"valueName": "so_" + g + ".sampler",
             "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
            {"valueName": "sq_" + g + ".sampler",
             "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
        ]
    for n in ("nx", "ny", "dt", "gravity", "rho_liquid", "rho_air",
              "alpha_liquid", "alpha_air", "max_iterations", "tolerance"):
        body_conns.append({"source": "." + n, "target": "step." + n})
    body_conns += [
        {"source": "sm_u.sampled_data", "target": "step.u_mass"},
        {"source": "so_u.sampled_data", "target": "step.u_mom"},
        {"source": "sq_u.sampled_data", "target": "step.u_phase"},
        {"source": "sm_v.sampled_data", "target": "step.v_mass"},
        {"source": "so_v.sampled_data", "target": "step.v_mom"},
        {"source": "sq_v.sampled_data", "target": "step.v_phase"},
        {"source": ".positions_in", "target": "step.positions"},
        {"source": ".velocities_in", "target": "step.velocities"},
        {"source": ".particle_phase", "target": "step.particle_phase"},
        {"source": "step.out_positions", "target": ".positions_out"},
        {"source": "step.out_velocities", "target": ".velocities_out"},
        {"source": "step.out_mom_x", "target": ".momx_out"},
        {"source": "step.out_mom_y", "target": ".momy_out"},
        {"source": "step.out_positions", "target": "w_pos.data"},
        {"source": ".pos_pattern", "target": "w_pos.file_path"},
        {"source": ".current_index", "target": "w_pos.frame"},
        {"source": "step.iterations_used", "target": "it_f.from"},
        {"source": "it_f.float", "target": "diag_arr.first.a"},
        {"source": "step.final_residual", "target": "diag_arr.first.b"},
        {"source": "step.max_divergence_after", "target": "diag_arr.first.c"},
        {"source": "step.max_speed", "target": "diag_arr.first.d"},
        {"source": "diag_arr.array", "target": "w_diag.data"},
        {"source": ".diag_pattern", "target": "w_diag.file_path"},
        {"source": ".current_index", "target": "w_diag.frame"},
        # the writes must feed a live output or lazy evaluation prunes them:
        # accumulate their success flags through the loop state.
        {"source": "w_pos.success", "target": "ok_fp.from"},
        {"source": "w_diag.success", "target": "ok_fd.from"},
        {"source": ".ok_in", "target": "ok_acc.first.s"},
        {"source": "ok_fp.float", "target": "ok_acc.first.p"},
        {"source": "ok_fd.float", "target": "ok_acc.first.q"},
        {"source": "ok_acc.output", "target": ".ok_out"},
    ]
    body_values += [
        {"valueName": "w_pos.overwrite", "valueType": "bool", "value": "true"},
        {"valueName": "w_pos.create_directories", "valueType": "bool", "value": "true"},
        {"valueName": "w_diag.overwrite", "valueType": "bool", "value": "true"},
        {"valueName": "w_diag.create_directories", "valueType": "bool", "value": "true"},
    ]
    body_ports = [
        P("iteration_limit", "input", "long", portIterationLimit="true"),
        P("current_index", "input", "long", portIterationCounter="true"),
        P("positions_in", "input", "array<Math::float3>"),
        P("positions_out", "output", "array<Math::float3>"),
        P("velocities_in", "input", "array<Math::float3>"),
        P("velocities_out", "output", "array<Math::float3>"),
        P("momx_in", "input", "array<float>"), P("momx_out", "output", "array<float>"),
        P("momy_in", "input", "array<float>"), P("momy_out", "output", "array<float>"),
        P("ok_in", "input", "float"), P("ok_out", "output", "float"),
        P("mass_in", "input", "array<float>"), P("particle_phase", "input", "array<float>"),
        P("probes_u", "input", "array<Math::float3>"),
        P("probes_v", "input", "array<Math::float3>"),
        P("pos_pattern", "input", "string"), P("diag_pattern", "input", "string"),
        P("radius", "input", "float"), P("add_to_denominator", "input", "float"),
        P("sample_default", "input", "float"),
        P("nx", "input", "int"), P("ny", "input", "int"),
        P("dt", "input", "float"), P("gravity", "input", "float"),
        P("rho_liquid", "input", "float"), P("rho_air", "input", "float"),
        P("alpha_liquid", "input", "float"), P("alpha_air", "input", "float"),
        P("max_iterations", "input", "int"), P("tolerance", "input", "float"),
    ]
    body = {"name": "loop", "ports": body_ports, "compounds": [],
            "compoundNodes": body_nodes, "connections": body_conns, "values": body_values,
            "iterateCompound": {"ports": [
                {"portKind": "state", "inputPortName": "positions_in",
                 "outputPortName": "positions_out"},
                {"portKind": "state", "inputPortName": "velocities_in",
                 "outputPortName": "velocities_out"},
                {"portKind": "state", "inputPortName": "momx_in", "outputPortName": "momx_out"},
                {"portKind": "state", "inputPortName": "momy_in", "outputPortName": "momy_out"},
                {"portKind": "state", "inputPortName": "ok_in", "outputPortName": "ok_out"},
            ]}}

    scal = [("nx", "int", "160"), ("ny", "int", "80"), ("dt", "float", "0.015f"),
            ("gravity", "float", "-9.8f"), ("rho_liquid", "float", "1000f"),
            ("rho_air", "float", "1f"), ("alpha_liquid", "float", "0.97f"),
            ("alpha_air", "float", "0.9f"), ("max_iterations", "int", "4000"),
            ("tolerance", "float", "0.000001f"), ("radius", "float", "1.5f"),
            ("add_to_denominator", "float", "0.000001f"), ("sample_default", "float", "0f"),
            ("substeps", "long", "4")]
    in_arr = [("positions", "t_pos"), ("velocities", "t_pos"), ("mass", "t_flt"),
              ("momx", "t_flt"), ("momy", "t_flt"), ("particle_phase", "t_flt"),
              ("probes_u", "t_pos"), ("probes_v", "t_pos")]
    nodes = [{"nodeName": "t_pos", "valueType": "array<Math::float3>"},
             {"nodeName": "t_flt", "valueType": "array<float>"},
             {"nodeName": "zerof", "valueType": "float"},
             {"nodeName": "loop", "nodeType": "loop"},
             {"nodeName": "w_final", "nodeType": "File::NumPy::write_NumPy"}]
    ports, conns = [], []
    values = [
        {"valueName": "t_pos.value", "valueType": "array<Math::float3>",
         "value": [{"x": "0f", "y": "0f", "z": "0f"}]},
        {"valueName": "t_flt.value", "valueType": "array<float>", "value": ["0f"]},
        {"valueName": "w_final.overwrite", "valueType": "bool", "value": "true"},
        {"valueName": "w_final.create_directories", "valueType": "bool", "value": "true"},
        {"valueName": "zerof.value", "valueType": "float", "value": "0f"},
    ]
    for n, t, dv in scal:
        ports.append(P(n, "input", t, dv))
        tgt = "iteration_limit" if n == "substeps" else n
        conns.append({"source": "." + n, "target": "loop." + tgt})
    state_map = {"positions": "positions_in", "velocities": "velocities_in",
                 "mass": "mass_in", "momx": "momx_in", "momy": "momy_in"}
    for n, rt in in_arr:
        ports.append(P("path_" + n, "input", "string", ""))
        nodes.append({"nodeName": "r_" + n, "nodeType": "File::NumPy::read_NumPy"})
        conns += [
            {"source": ".path_" + n, "target": "r_" + n + ".file_path"},
            {"source": rt + ".output", "target": "r_" + n + ".type"},
            {"source": "r_" + n + ".data", "target": "loop." + state_map.get(n, n)},
        ]
    for n in ("pos_pattern", "diag_pattern"):
        ports.append(P(n, "input", "string", ""))
        conns.append({"source": "." + n, "target": "loop." + n})
    ports += [P("path_final_positions", "input", "string", ""), P("ok_final", "output", "bool"),
              P("writes_total", "output", "float")]
    conns += [
        {"source": "zerof.output", "target": "loop.ok_in"},
        {"source": "loop.ok_out", "target": ".writes_total"},
        {"source": "loop.positions_out", "target": "w_final.data"},
        {"source": ".path_final_positions", "target": "w_final.file_path"},
        {"source": "w_final.success", "target": ".ok_final"},
    ]
    return {"header": {"metadata": [{"metaName": "adskFileFormatVersion",
                                     "metaValue": "100L"}]},
            "namespaces": [], "types": [],
            "compounds": [{"name": "User::PFFlip::sim_2d", "ports": ports,
                           "compounds": [body], "compoundNodes": nodes,
                           "connections": conns, "values": values}]}


def build3d():
    """The 3D loop: same architecture as build(), with a third staggered splat chain
    for w, the step_3d node, momz as extra state, and a preconditioner port."""
    g = build()
    top = g["compounds"][0]
    body = top["compounds"][0]

    for n in body["compoundNodes"]:
        if n["nodeName"] == "step":
            n["nodeType"] = "PFFlip::Solve::step_3d"
    body["compoundNodes"] += [
        {"nodeName": "set_mz", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "off_w", "valueType": "Math::float3"},
        {"nodeName": "add_w", "nodeType": "Core::Math::add", "multiInPortNames": ["pts", "off"]},
        {"nodeName": "spos_w", "nodeType": "Geometry::Properties::set_geo_property_data"},
        {"nodeName": "p2v_w", "nodeType": "Geometry::Converters::points_to_volume"},
        {"nodeName": "sp1_w", "nodeType": "Geometry::Volume::splat_points_into_volume"},
        {"nodeName": "sp2_w", "nodeType": "Geometry::Volume::splat_points_into_volume"},
        {"nodeName": "sp3_w", "nodeType": "Geometry::Volume::splat_points_into_volume"},
        {"nodeName": "sm_w", "nodeType": "Geometry::Query::sample_volume"},
        {"nodeName": "so_w", "nodeType": "Geometry::Query::sample_volume"},
        {"nodeName": "sq_w", "nodeType": "Geometry::Query::sample_volume"},
    ]
    conns = body["connections"]
    for k in conns:
        if k["source"] == "set_my.out_geometry" and k["target"] == "set_ph.geometry":
            k["source"] = "set_mz.out_geometry"
    for n in ("escape", "esc_phi", "drag_droplet", "drag_bubble", "buoyancy", "rho0_face"):
        body["ports"].append(P(n, "input", "float" if n != "escape" else "int"))
        conns.append({"source": "." + n, "target": "step." + n})
        top["ports"].append(P(n, "input", "float" if n != "escape" else "int",
                              "0" if n == "escape" else "0f"))
        top["connections"].append({"source": "." + n, "target": "loop." + n})
    body["compoundNodes"] += [
        {"nodeName": "w_esc", "nodeType": "File::NumPy::write_NumPy"},
        {"nodeName": "ok_fe", "nodeType": "Core::Type_Conversion::to_float"},
    ]
    for n2 in body["compoundNodes"]:
        if n2["nodeName"] == "ok_acc":
            n2["multiInPortNames"] = ["s", "p", "q", "e"]
    conns += [
        {"source": "step.out_escaped", "target": "w_esc.data"},
        {"source": ".esc_pattern", "target": "w_esc.file_path"},
        {"source": ".current_index", "target": "w_esc.frame"},
        {"source": "w_esc.success", "target": "ok_fe.from"},
        {"source": "ok_fe.float", "target": "ok_acc.first.e"},
    ]
    body["values"] += [
        {"valueName": "w_esc.overwrite", "valueType": "bool", "value": "true"},
        {"valueName": "w_esc.create_directories", "valueType": "bool", "value": "true"},
    ]
    body["ports"].append(P("esc_pattern", "input", "string"))
    top["ports"].append(P("esc_pattern", "input", "string", ""))
    top["connections"].append({"source": ".esc_pattern", "target": "loop.esc_pattern"})

    conns += [
        {"source": "set_my.out_geometry", "target": "set_mz.geometry"},
        {"source": ".momz_in", "target": "set_mz.data"},
        {"source": "zero.output", "target": "set_mz.default"},
        {"source": ".positions_in", "target": "add_w.first.pts"},
        {"source": "off_w.output", "target": "add_w.first.off"},
        {"source": "set_ph.out_geometry", "target": "spos_w.geometry"},
        {"source": "add_w.output", "target": "spos_w.data"},
        {"source": "spos_w.out_geometry", "target": "p2v_w.points"},
        {"source": "spos_w.out_geometry", "target": "sp1_w.points"},
        {"source": "spos_w.out_geometry", "target": "sp2_w.points"},
        {"source": "spos_w.out_geometry", "target": "sp3_w.points"},
        {"source": "p2v_w.volume", "target": "sp1_w.volume"},
        {"source": "sp1_w.out_volume", "target": "sp2_w.volume"},
        {"source": "sp2_w.out_volume", "target": "sp3_w.volume"},
        {"source": "sp3_w.out_volume", "target": "sm_w.volume"},
        {"source": "sp3_w.out_volume", "target": "so_w.volume"},
        {"source": "sp3_w.out_volume", "target": "sq_w.volume"},
        {"source": ".probes_w", "target": "sm_w.positions"},
        {"source": ".probes_w", "target": "so_w.positions"},
        {"source": ".probes_w", "target": "sq_w.positions"},
        {"source": ".radius", "target": "sp1_w.radius"},
        {"source": ".radius", "target": "sp2_w.radius"},
        {"source": ".radius", "target": "sp3_w.radius"},
        {"source": ".add_to_denominator", "target": "sp1_w.add_to_denominator"},
        {"source": ".add_to_denominator", "target": "sp2_w.add_to_denominator"},
        {"source": ".add_to_denominator", "target": "sp3_w.add_to_denominator"},
        {"source": ".sample_default", "target": "sm_w.default"},
        {"source": ".sample_default", "target": "so_w.default"},
        {"source": ".sample_default", "target": "sq_w.default"},
        {"source": "sm_w.sampled_data", "target": "step.w_mass"},
        {"source": "so_w.sampled_data", "target": "step.w_mom"},
        {"source": "sq_w.sampled_data", "target": "step.w_phase"},
        {"source": ".nz", "target": "step.nz"},
        {"source": ".preconditioner", "target": "step.preconditioner"},
        {"source": "step.out_mom_z", "target": ".momz_out"},
    ]
    body["values"] += [
        {"valueName": "set_mz.property", "valueType": "string", "value": "voxel_mz"},
        {"valueName": "off_w.value", "valueType": "Math::float3",
         "value": {"x": "0f", "y": "0f", "z": "0.5f"}},
        {"valueName": "p2v_w.detail_size", "valueType": "float", "value": "1f"},
        {"valueName": "p2v_w.properties", "valueType": "string", "value": ""},
        {"valueName": "p2v_w.resolution_mode",
         "valueType": "Geometry::Volume::ResolutionType", "value": "Absolute"},
        {"valueName": "sm_w.property", "valueType": "string", "value": "voxel_m"},
        {"valueName": "so_w.property", "valueType": "string", "value": "voxel_mz"},
        {"valueName": "sq_w.property", "valueType": "string", "value": "voxel_ph"},
        {"valueName": "sm_w.sampler", "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
        {"valueName": "so_w.sampler", "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
        {"valueName": "sq_w.sampler", "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
    ]
    for sp, prop in (("sp1_w", "voxel_m"), ("sp2_w", "voxel_mz"), ("sp3_w", "voxel_ph")):
        body["values"] += [
            {"valueName": sp + ".create_properties", "valueType": "bool", "value": "true"},
            {"valueName": sp + ".properties", "valueType": "string", "value": prop},
            {"valueName": sp + ".kernel",
             "valueType": "Geometry::Volume::SplatKernelType", "value": "kLinearKernel"},
            {"valueName": sp + ".add_to_weights", "valueType": "float", "value": "0f"},
            {"valueName": sp + ".smoothing", "valueType": "float", "value": "0f"},
            {"valueName": sp + ".coarsest_depth", "valueType": "int", "value": "0"},
        ]
    body["ports"] += [
        P("momz_in", "input", "array<float>"), P("momz_out", "output", "array<float>"),
        P("probes_w", "input", "array<Math::float3>"),
        P("nz", "input", "int"), P("preconditioner", "input", "int"),
    ]
    body["iterateCompound"]["ports"].insert(
        3, {"portKind": "state", "inputPortName": "momz_in", "outputPortName": "momz_out"})

    top["ports"] += [P("nz", "input", "int", "32"), P("preconditioner", "input", "int", "1"),
                     P("path_momz", "input", "string", ""),
                     P("path_probes_w", "input", "string", "")]
    top["compoundNodes"] += [
        {"nodeName": "r_momz", "nodeType": "File::NumPy::read_NumPy"},
        {"nodeName": "r_probes_w", "nodeType": "File::NumPy::read_NumPy"},
    ]
    top["connections"] += [
        {"source": ".nz", "target": "loop.nz"},
        {"source": ".preconditioner", "target": "loop.preconditioner"},
        {"source": ".path_momz", "target": "r_momz.file_path"},
        {"source": "t_flt.output", "target": "r_momz.type"},
        {"source": "r_momz.data", "target": "loop.momz_in"},
        {"source": ".path_probes_w", "target": "r_probes_w.file_path"},
        {"source": "t_pos.output", "target": "r_probes_w.type"},
        {"source": "r_probes_w.data", "target": "loop.probes_w"},
    ]
    # ---- ST wiring: tau/wt/wtph as loop state, the temporal-weight channel as a
    # fourth splat+sample per staggered grid, scalars, and synced-position dumps.
    body["compoundNodes"] += [
        {"nodeName": "set_wt", "nodeType": "Geometry::Properties::set_geo_property"},
        {"nodeName": "idx_i", "nodeType": "Core::Type_Conversion::to_int"},
    ]
    for g2 in ("u", "v", "w"):
        body["compoundNodes"] += [
            {"nodeName": "sp4_" + g2, "nodeType": "Geometry::Volume::splat_points_into_volume"},
            {"nodeName": "sw_" + g2, "nodeType": "Geometry::Query::sample_volume"},
        ]
    conns = body["connections"]
    # property chain: insert voxel_wt between set_mz and set_ph
    for k in conns:
        if k["source"] == "set_mz.out_geometry" and k["target"] == "set_ph.geometry":
            k["source"] = "set_wt.out_geometry"
    conns += [
        {"source": "set_mz.out_geometry", "target": "set_wt.geometry"},
        {"source": ".wt_in", "target": "set_wt.data"},
        {"source": "zero.output", "target": "set_wt.default"},
    ]
    # voxel_ph now carries the premultiplied phase state
    for k in conns:
        if k["source"] == ".particle_phase" and k["target"] == "set_ph.data":
            k["source"] = ".wtph_in"
    for g2 in ("u", "v", "w"):
        conns += [
            {"source": "sp3_" + g2 + ".out_volume", "target": "sp4_" + g2 + ".volume"},
            {"source": "spos_" + g2 + ".out_geometry", "target": "sp4_" + g2 + ".points"},
            {"source": ".radius", "target": "sp4_" + g2 + ".radius"},
            {"source": ".add_to_denominator", "target": "sp4_" + g2 + ".add_to_denominator"},
            {"source": "sp4_" + g2 + ".out_volume", "target": "sw_" + g2 + ".volume"},
            {"source": ".probes_" + g2, "target": "sw_" + g2 + ".positions"},
            {"source": ".sample_default", "target": "sw_" + g2 + ".default"},
            {"source": "sw_" + g2 + ".sampled_data", "target": "step." + g2 + "_wt"},
        ]
        body["values"] += [
            {"valueName": "sp4_" + g2 + ".create_properties", "valueType": "bool", "value": "true"},
            {"valueName": "sp4_" + g2 + ".properties", "valueType": "string", "value": "voxel_wt"},
            {"valueName": "sp4_" + g2 + ".kernel",
             "valueType": "Geometry::Volume::SplatKernelType", "value": "kLinearKernel"},
            {"valueName": "sp4_" + g2 + ".add_to_weights", "valueType": "float", "value": "0f"},
            {"valueName": "sp4_" + g2 + ".smoothing", "valueType": "float", "value": "0f"},
            {"valueName": "sp4_" + g2 + ".coarsest_depth", "valueType": "int", "value": "0"},
            {"valueName": "sw_" + g2 + ".property", "valueType": "string", "value": "voxel_wt"},
            {"valueName": "sw_" + g2 + ".sampler",
             "valueType": "Geometry::Query::SamplerType", "value": "kLinear"},
        ]
    # sample_volume on sp3 was the phase channel chain end; sp4 extends it, so the
    # existing sm/so/sq nodes still read sp3's volume which lacks voxel_wt - repoint
    # their volume source to sp4 so every sampler sees the full volume.
    for g2 in ("u", "v", "w"):
        for k in conns:
            if k["source"] == "sp3_" + g2 + ".out_volume" and k["target"].startswith(
                    ("sm_" + g2, "so_" + g2, "sq_" + g2)):
                k["source"] = "sp4_" + g2 + ".out_volume"
    body["values"] += [
        {"valueName": "set_wt.property", "valueType": "string", "value": "voxel_wt"},
    ]
    # scalars and state
    for n, t in (("st", "int"), ("st_seed", "int")):
        body["ports"].append(P(n, "input", t))
        conns.append({"source": "." + n, "target": "step." + n})
        top["ports"].append(P(n, "input", t, "0" if n == "st" else "7"))
        top["connections"].append({"source": "." + n, "target": "loop." + n})
    conns += [
        {"source": ".current_index", "target": "idx_i.from"},
        {"source": "idx_i.int", "target": "step.step_index"},
    ]
    body["ports"] += [
        P("tau_in", "input", "array<float>"), P("tau_out2", "output", "array<float>"),
        P("wt_in", "input", "array<float>"), P("wt_out2", "output", "array<float>"),
        P("wtph_in", "input", "array<float>"), P("wtph_out2", "output", "array<float>"),
    ]
    conns += [
        {"source": ".tau_in", "target": "step.tau_in"},
        {"source": "step.tau_out", "target": ".tau_out2"},
        {"source": "step.out_wt", "target": ".wt_out2"},
        {"source": "step.out_wtph", "target": ".wtph_out2"},
    ]
    body["iterateCompound"]["ports"] += [
        {"portKind": "state", "inputPortName": "tau_in", "outputPortName": "tau_out2"},
        {"portKind": "state", "inputPortName": "wt_in", "outputPortName": "wt_out2"},
        {"portKind": "state", "inputPortName": "wtph_in", "outputPortName": "wtph_out2"},
    ]
    # dumps use time-resynchronized positions
    for k in conns:
        if k["source"] == "step.out_positions" and k["target"] == "w_pos.data":
            k["source"] = "step.out_pos_synced"
    # outer reads for the three new initial states
    for n in ("tau", "wt", "wtph"):
        top["ports"].append(P("path_" + n, "input", "string", ""))
        top["compoundNodes"].append({"nodeName": "r_" + n, "nodeType": "File::NumPy::read_NumPy"})
        top["connections"] += [
            {"source": ".path_" + n, "target": "r_" + n + ".file_path"},
            {"source": "t_flt.output", "target": "r_" + n + ".type"},
            {"source": "r_" + n + ".data", "target": "loop." + n + "_in"},
        ]

    g["compounds"][0]["name"] = "User::PFFlip::sim_3d"
    return g


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sim_2d.json"), "w") as f:
        json.dump(build(), f, indent=1)
    with open(os.path.join(here, "sim_3d.json"), "w") as f:
        json.dump(build3d(), f, indent=1)
    print("wrote sim_2d.json and sim_3d.json")
