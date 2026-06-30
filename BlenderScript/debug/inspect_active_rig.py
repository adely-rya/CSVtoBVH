import bpy
from pathlib import Path


def vec(value):
    return f"({value.x:.6f}, {value.y:.6f}, {value.z:.6f})"


def matrix_rows(matrix):
    return [
        "    " + " ".join(f"{value: .6f}" for value in row)
        for row in matrix
    ]


def armature_report(obj):
    lines = [
        f"ARMATURE: {obj.name}",
        f"data: {obj.data.name}",
        f"location: {vec(obj.location)}",
        f"rotation_mode: {obj.rotation_mode}",
        f"rotation_euler: {vec(obj.rotation_euler)}",
        f"scale: {vec(obj.scale)}",
        f"dimensions: {vec(obj.dimensions)}",
        f"matrix_world:",
        *matrix_rows(obj.matrix_world),
        f"action: {obj.animation_data.action.name if obj.animation_data and obj.animation_data.action else '(none)'}",
        f"bones: {len(obj.data.bones)}",
        "",
        "BONE HIERARCHY AND REST DATA",
    ]

    for bone in obj.data.bones:
        pose_bone = obj.pose.bones.get(bone.name)
        parent = bone.parent.name if bone.parent else "(root)"
        children = ", ".join(child.name for child in bone.children) or "(none)"
        lines.extend([
            f"BONE: {bone.name}",
            f"  parent: {parent}",
            f"  children: {children}",
            f"  use_connect: {bone.use_connect}",
            f"  head_local: {vec(bone.head_local)}",
            f"  tail_local: {vec(bone.tail_local)}",
            f"  length: {bone.length:.6f}",
            f"  matrix_local:",
            *matrix_rows(bone.matrix_local),
        ])
        if pose_bone:
            lines.extend([
                f"  pose_rotation_mode: {pose_bone.rotation_mode}",
                f"  constraints: {', '.join(c.type + ':' + c.name for c in pose_bone.constraints) or '(none)'}",
            ])
        lines.append("")

    return lines


def report_path():
    project_path = Path(
        r"H:\SummerCampOP2026\trackingprogram\CSVtoBVH\BlenderScript\rig_report.txt"
    )
    if project_path.parent.exists():
        return project_path

    try:
        source = Path(__file__).resolve()
        if source.exists():
            return source.with_name("rig_report.txt")
    except NameError:
        pass
    return Path(bpy.path.abspath("//")) / "rig_report.txt"


def main():
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature objects were found in the current Blender file.")

    active = bpy.context.view_layer.objects.active
    lines = [
        f"Blender: {bpy.app.version_string}",
        f"Scene: {bpy.context.scene.name}",
        f"Active object: {active.name if active else '(none)'}",
        f"Armature objects: {', '.join(obj.name for obj in armatures)}",
        "",
    ]

    for index, obj in enumerate(armatures):
        if index:
            lines.extend(["", "=" * 80, ""])
        lines.extend(armature_report(obj))

    path = report_path()
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rig report saved: {path}")


main()
