import bpy
from pathlib import Path


# Run this script while the retargeted pose is visibly twisted.
# All armatures are included, so object names do not need to be configured.
OUTPUT_NAME = "shoulder_retarget_report.txt"

INTERESTING_BONE_SUFFIXES = (
    "Spine",
    "Spine1",
    "Spine2",
    "Chest",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
)


def vec(value):
    return "(" + ", ".join(f"{item:.9f}" for item in value) + ")"


def matrix_rows(matrix, indent="    "):
    return [
        indent + " ".join(f"{value: .9f}" for value in row)
        for row in matrix
    ]


def bone_suffix(name):
    return name.rsplit(":", 1)[-1]


def interesting_pose_bones(obj):
    wanted = set(INTERESTING_BONE_SUFFIXES)
    return [bone for bone in obj.pose.bones if bone_suffix(bone.name) in wanted]


def transform_lines(label, matrix, indent="  "):
    location, rotation, scale = matrix.decompose()
    euler = rotation.to_euler("XYZ")
    axis, angle = rotation.to_axis_angle()
    return [
        f"{indent}{label}:",
        f"{indent}  location: {vec(location)}",
        f"{indent}  quaternion_wxyz: {vec(rotation)}",
        f"{indent}  euler_xyz_degrees: {vec(tuple(v * 57.29577951308232 for v in euler))}",
        f"{indent}  axis_angle_degrees: axis={vec(axis)} angle={angle * 57.29577951308232:.9f}",
        f"{indent}  scale: {vec(scale)}",
        f"{indent}  matrix:",
        *matrix_rows(matrix, indent + "    "),
    ]


def constraint_lines(constraint):
    target = getattr(constraint, "target", None)
    lines = [
        f"    CONSTRAINT: {constraint.name}",
        f"      type: {constraint.type}",
        f"      enabled: {not constraint.mute}",
        f"      influence: {constraint.influence:.9f}",
        f"      target: {target.name if target else '(none)'}",
        f"      subtarget: {getattr(constraint, 'subtarget', '') or '(none)'}",
    ]
    for attr in (
        "owner_space",
        "target_space",
        "mix_mode",
        "euler_order",
        "use_x",
        "use_y",
        "use_z",
        "invert_x",
        "invert_y",
        "invert_z",
        "use_offset",
    ):
        if hasattr(constraint, attr):
            lines.append(f"      {attr}: {getattr(constraint, attr)}")
    return lines


def bone_report(obj, pose_bone):
    rest_bone = pose_bone.bone
    parent_name = pose_bone.parent.name if pose_bone.parent else "(none)"
    # matrix is armature-space evaluated pose. matrix_basis is the animation/
    # constraint input relative to the rest pose and parent.
    lines = [
        f"  BONE: {pose_bone.name}",
        f"    suffix: {bone_suffix(pose_bone.name)}",
        f"    parent: {parent_name}",
        f"    rotation_mode: {pose_bone.rotation_mode}",
        f"    inherit_rotation: {rest_bone.use_inherit_rotation}",
        f"    inherit_scale: {rest_bone.inherit_scale}",
        f"    rest_head_local: {vec(rest_bone.head_local)}",
        f"    rest_tail_local: {vec(rest_bone.tail_local)}",
        f"    rest_length: {rest_bone.length:.9f}",
    ]
    lines.extend(transform_lines("rest_matrix_local", rest_bone.matrix_local, "    "))
    lines.extend(transform_lines("matrix_basis", pose_bone.matrix_basis, "    "))
    lines.extend(transform_lines("evaluated_pose_matrix_armature_space", pose_bone.matrix, "    "))
    lines.extend(transform_lines(
        "evaluated_pose_matrix_world_space",
        obj.matrix_world @ pose_bone.matrix,
        "    ",
    ))

    if pose_bone.parent:
        parent_relative = pose_bone.parent.matrix.inverted_safe() @ pose_bone.matrix
        lines.extend(transform_lines(
            "evaluated_matrix_relative_to_parent_pose",
            parent_relative,
            "    ",
        ))

    if pose_bone.constraints:
        for constraint in pose_bone.constraints:
            lines.extend(constraint_lines(constraint))
    else:
        lines.append("    constraints: (none)")
    return lines


def fcurve_report(obj):
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return ["  ACTION: (none)"]

    interesting_names = {bone.name for bone in interesting_pose_bones(obj)}
    lines = [f"  ACTION: {action.name}", f"  action_frame_range: {vec(action.frame_range)}"]
    matching = []
    for curve in action.fcurves:
        if any(f'pose.bones["{name}"]' in curve.data_path for name in interesting_names):
            matching.append(curve)
    lines.append(f"  matching_arm_fcurves: {len(matching)}")
    for curve in matching:
        values = [point.co.y for point in curve.keyframe_points]
        value_range = (
            f"{min(values):.9f} .. {max(values):.9f}" if values else "(no keys)"
        )
        lines.append(
            f"    {curve.data_path}[{curve.array_index}] "
            f"keys={len(curve.keyframe_points)} range={value_range}"
        )
    return lines


def armature_report(obj):
    lines = [
        "=" * 100,
        f"ARMATURE: {obj.name}",
        f"  data: {obj.data.name}",
        f"  parent_object: {obj.parent.name if obj.parent else '(none)'}",
        f"  display_type: {obj.display_type}",
        f"  pose_position: {obj.data.pose_position}",
        f"  object_rotation_mode: {obj.rotation_mode}",
    ]
    lines.extend(transform_lines("object_matrix_world", obj.matrix_world, "  "))
    lines.extend(fcurve_report(obj))

    bones = interesting_pose_bones(obj)
    lines.append(f"  reported_bones: {', '.join(b.name for b in bones) or '(none)'}")
    for pose_bone in bones:
        lines.append("")
        lines.extend(bone_report(obj, pose_bone))
    return lines


def report_path():
    project_path = Path(
        r"H:\SummerCampOP2026\trackingprogram\CSVtoBVH\BlenderScript"
    ) / OUTPUT_NAME
    if project_path.parent.exists():
        return project_path

    try:
        source = Path(__file__).resolve()
        if source.exists():
            return source.with_name(OUTPUT_NAME)
    except NameError:
        pass
    return Path(bpy.path.abspath("//")) / OUTPUT_NAME


def main():
    scene = bpy.context.scene
    # Force dependency-graph evaluation at the currently displayed frame.
    scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    bpy.context.view_layer.update()

    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature objects were found in the current scene.")

    active = bpy.context.view_layer.objects.active
    lines = [
        "SHOULDER RETARGET DIAGNOSTIC REPORT",
        f"Blender: {bpy.app.version_string}",
        f"Blend file: {bpy.data.filepath or '(not saved)'}",
        f"Scene: {scene.name}",
        f"Current frame: {scene.frame_current} + {scene.frame_subframe:.9f}",
        f"FPS: {scene.render.fps} / {scene.render.fps_base}",
        f"Active object: {active.name if active else '(none)'}",
        f"Armatures in scene: {', '.join(obj.name for obj in armatures)}",
        "",
        "Quaternion order in this report is (W, X, Y, Z).",
        "Matrices are evaluated after animation, drivers and constraints.",
    ]
    for obj in armatures:
        lines.extend(["", *armature_report(obj)])

    path = report_path()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Shoulder retarget report saved: {path}")


main()
