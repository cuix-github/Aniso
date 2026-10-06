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


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_2d.json")
    with open(out, "w") as f:
        json.dump(build(), f, indent=1)
    print("wrote", out)
