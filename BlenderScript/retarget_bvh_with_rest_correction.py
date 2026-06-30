import math
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


# Blender's Scripting workspace에서実行する独立スクリプトです。
# オブジェクト名だけ現在の .blend に合わせて変更してください。
SOURCE_ARMATURE_NAME = "output_plus_start60_mixamo_v2"
TARGET_ARMATURE_NAME = "Armature.001"

FRAME_STEP = 1
ACTION_NAME = "FM_Retarget_RestCorrected"
GENERATED_CONSTRAINT_PREFIX = "FM_RETARGET_"
LOG_FILE_NAME = "retarget_bvh_result.txt"

# True: ターゲットの既存Actionを外し、新しいActionへベイクします。
# False: 現在のActionへキーを追加します。
CREATE_NEW_ACTION = True

# 腰など、階層の最上位にある対応ボーンの移動も転送します。
COPY_ROOT_LOCATION = True

# Positive values lower both shoulders. Try roughly 4 to 12 degrees.
# Set to 0.0 to disable the adjustment.
SHOULDER_SLOPE_DEGREES = 3.0

# Positive values open the upper arms away from the torso.
# Set to 0.0 to disable the adjustment.
ARM_OUTWARD_DEGREES = 0

DIAGNOSTIC_BONE_SUFFIXES = {
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
}

# These bones may use an arms-down rest pose in the BVH while the character
# uses a T-pose. Preserve their direction at the first animation frame instead
# of forcing that frame onto the character's T-pose.
CALIBRATE_INITIAL_DIRECTION_SUFFIXES = {
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
}

SHOULDER_SUFFIXES = {
    "Left": "LeftShoulder",
    "Right": "RightShoulder",
}

ARM_CHAIN_SUFFIXES = {
    "Left": {"LeftArm", "LeftForeArm", "LeftHand"},
    "Right": {"RightArm", "RightForeArm", "RightHand"},
}


def get_armature(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Armature object not found: {name}")
    if obj.type != "ARMATURE":
        raise RuntimeError(f"Object is not an armature: {name}")
    return obj


def bone_suffix(name):
    return name.rsplit(":", 1)[-1]


def log_path():
    project_path = Path(
        r"H:\SummerCampOP2026\trackingprogram\CSVtoBVH\BlenderScript"
    ) / LOG_FILE_NAME
    if project_path.parent.exists():
        return project_path

    try:
        script_path = Path(__file__).resolve()
        if script_path.exists():
            return script_path.with_name(LOG_FILE_NAME)
    except NameError:
        pass

    return Path(bpy.path.abspath("//")) / LOG_FILE_NAME


def vec(value):
    return "(" + ", ".join(f"{item:.9f}" for item in value) + ")"


def transform_log(label, matrix, indent="    "):
    location, rotation, scale = matrix.decompose()
    euler = rotation.to_euler("XYZ")
    degrees = tuple(math.degrees(value) for value in euler)
    return [
        f"{indent}{label}:",
        f"{indent}  location: {vec(location)}",
        f"{indent}  quaternion_wxyz: {vec(rotation)}",
        f"{indent}  euler_xyz_degrees: {vec(degrees)}",
        f"{indent}  scale: {vec(scale)}",
    ]


def build_bone_mapping(source, target):
    """Match exact names first, then unique names without namespace prefixes."""
    target_by_suffix = {}
    for bone in target.pose.bones:
        target_by_suffix.setdefault(bone_suffix(bone.name), []).append(bone.name)

    mapping = {}
    for source_bone in source.pose.bones:
        if target.pose.bones.get(source_bone.name):
            mapping[source_bone.name] = source_bone.name
            continue

        candidates = target_by_suffix.get(bone_suffix(source_bone.name), [])
        if len(candidates) == 1:
            mapping[source_bone.name] = candidates[0]

    if not mapping:
        raise RuntimeError("No corresponding bones were found.")
    return mapping


def hierarchy_depth(pose_bone):
    depth = 0
    while pose_bone.parent is not None:
        depth += 1
        pose_bone = pose_bone.parent
    return depth


def source_frame_range(source):
    if source.animation_data is None or source.animation_data.action is None:
        raise RuntimeError(f"Source armature has no active action: {source.name}")
    start, end = source.animation_data.action.frame_range
    return int(math.ceil(start)), int(math.floor(end))


def remove_generated_constraints(target):
    for pose_bone in target.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.name.startswith(GENERATED_CONSTRAINT_PREFIX):
                pose_bone.constraints.remove(constraint)


def mute_existing_constraints(target, mapped_target_names):
    muted = []
    for bone_name in mapped_target_names:
        pose_bone = target.pose.bones[bone_name]
        for constraint in pose_bone.constraints:
            if not constraint.mute:
                constraint.mute = True
                muted.append(f"{bone_name} / {constraint.name}")
    return muted


def prepare_target_action(target):
    if target.animation_data is None:
        target.animation_data_create()

    old_action = target.animation_data.action
    if CREATE_NEW_ACTION:
        target.animation_data.action = bpy.data.actions.new(ACTION_NAME)
    elif old_action is None:
        target.animation_data.action = bpy.data.actions.new(ACTION_NAME)

    target.animation_data.use_nla = False
    return old_action.name if old_action else "(none)"


def reset_mapped_pose(target, mapped_target_names):
    for bone_name in mapped_target_names:
        target.pose.bones[bone_name].matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def rest_matrix_world(armature, bone_name):
    return armature.matrix_world @ armature.data.bones[bone_name].matrix_local


def pose_matrix_world(armature, bone_name):
    return armature.matrix_world @ armature.pose.bones[bone_name].matrix


def rotation_only_matrix(quaternion):
    return quaternion.normalized().to_matrix().to_4x4()


def root_source_names(source, mapping):
    mapped = set(mapping)
    roots = []
    for source_name in mapping:
        parent = source.pose.bones[source_name].parent
        if parent is None or parent.name not in mapped:
            roots.append(source_name)
    return roots


def set_rotation_from_world_matrix(target, target_bone_name, desired_world):
    pose_bone = target.pose.bones[target_bone_name]
    desired_armature = target.matrix_world.inverted_safe() @ desired_world

    # Keep the target rest length and current translation. Only the evaluated
    # orientation is taken from the retarget matrix.
    current_location = pose_bone.matrix.translation.copy()
    desired_rotation = desired_armature.to_quaternion().to_matrix().to_4x4()
    desired_rotation.translation = current_location
    pose_bone.matrix = desired_rotation


def key_rotation(pose_bone, frame):
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.keyframe_insert(
        data_path="rotation_quaternion",
        frame=frame,
        group=pose_bone.name,
    )


def set_linear_interpolation(action):
    for curve in action.fcurves:
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def write_log(lines):
    path = log_path()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Retarget log saved: {path}")
    return path


def main(log):
    source = get_armature(SOURCE_ARMATURE_NAME)
    target = get_armature(TARGET_ARMATURE_NAME)
    mapping = build_bone_mapping(source, target)

    # Parents must be evaluated before their children.
    ordered_mapping = sorted(
        mapping.items(),
        key=lambda pair: hierarchy_depth(target.pose.bones[pair[1]]),
    )
    mapped_target_names = {target_name for _, target_name in ordered_mapping}
    root_names = root_source_names(source, mapping)

    frame_start, frame_end = source_frame_range(source)
    scene = bpy.context.scene
    original_frame = scene.frame_current

    remove_generated_constraints(target)
    muted_constraints = mute_existing_constraints(target, mapped_target_names)
    old_action_name = prepare_target_action(target)
    reset_mapped_pose(target, mapped_target_names)

    # Constant change-of-coordinate matrices between the source and target
    # rest frames. Multiplying this on the LEFT converts the direction of the
    # motion as well as its orientation. Applying a correction on the right
    # makes e.g. a source "raise arm" rotation act around the target's backward
    # axis when the two armatures use different object/rest coordinate systems.
    rest_conversions = {}
    for source_name, target_name in ordered_mapping:
        source_rest = rest_matrix_world(source, source_name)
        target_rest = rest_matrix_world(target, target_name)
        rest_conversions[source_name] = target_rest @ source_rest.inverted_safe()

    log.extend([
        "RETARGET BVH RESULT",
        f"Blender: {bpy.app.version_string}",
        f"Blend file: {bpy.data.filepath or '(not saved)'}",
        f"Source armature: {source.name}",
        f"Target armature: {target.name}",
        f"Frame range: {frame_start} .. {frame_end}",
        f"Frame step: {FRAME_STEP}",
        f"Mapped bones: {len(mapping)}",
        f"Previous target action: {old_action_name}",
        f"Output action: {target.animation_data.action.name}",
        f"Muted constraints: {len(muted_constraints)}",
        "",
        "BONE MAPPING AND REST CORRECTION",
    ])
    for source_name, target_name in ordered_mapping:
        correction = rest_conversions[source_name].to_quaternion()
        axis, angle = correction.to_axis_angle()
        log.append(
            f"  {source_name} -> {target_name}: "
            f"angle_degrees={math.degrees(angle):.9f}, axis={vec(axis)}"
        )

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    # Per-bone rest conversion correctly maps rotation axes, but it also maps
    # an arms-down BVH rest pose to a target T-pose. Build a constant
    # calibration which preserves the source bone direction at frame_start.
    # A common root-space conversion maps anatomical directions; the swing
    # from the target rest Y axis to that direction retains the target roll.
    source_root_name = root_names[0]
    target_root_name = mapping[source_root_name]
    # pose_matrix_world() has already applied each armature object's transform.
    # Both values are therefore in the same Blender world coordinate system;
    # applying the target object's 90-degree X rotation again would turn the
    # standing arm's downward direction toward the character's back.
    root_conversion_rotation = Quaternion()
    initial_direction_calibrations = {}
    for source_name, target_name in ordered_mapping:
        if bone_suffix(target_name) not in CALIBRATE_INITIAL_DIRECTION_SUFFIXES:
            continue

        source_first_rotation = pose_matrix_world(
            source, source_name
        ).to_quaternion()
        mapped_source_y = (
            root_conversion_rotation
            @ (source_first_rotation @ Vector((0.0, 1.0, 0.0)))
        ).normalized()

        target_rest_rotation = rest_matrix_world(
            target, target_name
        ).to_quaternion()
        target_rest_y = (
            target_rest_rotation @ Vector((0.0, 1.0, 0.0))
        ).normalized()
        direction_swing = target_rest_y.rotation_difference(mapped_source_y)
        desired_first_rotation = direction_swing @ target_rest_rotation

        raw_first_rotation = (
            root_conversion_rotation
            @ pose_matrix_world(source, source_name).to_quaternion()
        )
        # Apply the bone-specific rest/roll correction on the right. The
        # common object-space conversion remains on the left, so source world
        # motion axes (including arm raising) cannot be rotated by arbitrary
        # target/source bone roll differences.
        calibration = raw_first_rotation.inverted() @ desired_first_rotation
        initial_direction_calibrations[source_name] = rotation_only_matrix(
            calibration
        )
        log.append(
            f"  Initial direction calibration {source_name}: "
            f"angle_degrees={math.degrees(calibration.angle):.9f}"
        )

    # Rotate only the clavicle/shoulder bone toward the character's down
    # direction. Rotating the complete arm chain here also pulls the hands
    # inward. Upper-arm and lower-arm orientations remain controlled by the
    # retargeted animation.
    target_up = (
        rest_matrix_world(target, target_root_name).to_quaternion()
        @ Vector((0.0, 1.0, 0.0))
    ).normalized()
    shoulder_slope_rotations = {}
    if abs(SHOULDER_SLOPE_DEGREES) > 1.0e-8:
        for side, shoulder_suffix in SHOULDER_SUFFIXES.items():
            shoulder_pair = next(
                (
                    (source_name, target_name)
                    for source_name, target_name in ordered_mapping
                    if bone_suffix(target_name) == shoulder_suffix
                ),
                None,
            )
            if shoulder_pair is None:
                continue

            _, target_shoulder_name = shoulder_pair
            shoulder_rest = rest_matrix_world(target, target_shoulder_name)
            outward = (
                shoulder_rest.to_quaternion() @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            slope_axis = outward.cross(-target_up)
            if slope_axis.length < 1.0e-8:
                continue
            slope_rotation = Quaternion(
                slope_axis.normalized(),
                math.radians(SHOULDER_SLOPE_DEGREES),
            )
            source_shoulder_name, _ = shoulder_pair
            shoulder_slope_rotations[source_shoulder_name] = (
                rotation_only_matrix(slope_rotation)
            )
            log.append(
                f"  Shoulder slope {side}: "
                f"degrees={SHOULDER_SLOPE_DEGREES:.9f}, "
                f"axis={vec(slope_axis.normalized())}"
            )

    # The source arms-down reference can lean slightly toward the torso.
    # Open the complete arm chain around an axis which moves the upper-arm
    # direction toward the character's anatomical outside.
    arm_outward_rotations = {}
    if abs(ARM_OUTWARD_DEGREES) > 1.0e-8:
        for side, suffixes in ARM_CHAIN_SUFFIXES.items():
            arm_suffix = f"{side}Arm"
            arm_pair = next(
                (
                    (source_name, target_name)
                    for source_name, target_name in ordered_mapping
                    if bone_suffix(target_name) == arm_suffix
                ),
                None,
            )
            shoulder_pair = next(
                (
                    (source_name, target_name)
                    for source_name, target_name in ordered_mapping
                    if bone_suffix(target_name) == f"{side}Shoulder"
                ),
                None,
            )
            if arm_pair is None or shoulder_pair is None:
                continue

            source_arm_name, target_arm_name = arm_pair
            _, target_shoulder_name = shoulder_pair
            raw_first = (
                rotation_only_matrix(root_conversion_rotation)
                @ pose_matrix_world(source, source_arm_name)
            )
            calibrated_first_rotation = raw_first.to_quaternion()
            if source_arm_name in initial_direction_calibrations:
                calibrated_first_rotation = (
                    calibrated_first_rotation
                    @ initial_direction_calibrations[
                        source_arm_name
                    ].to_quaternion()
                )
            arm_direction = (
                calibrated_first_rotation @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            outward = (
                rest_matrix_world(
                    target, target_shoulder_name
                ).to_quaternion()
                @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            outward_axis = arm_direction.cross(outward)
            if outward_axis.length < 1.0e-8:
                continue
            outward_rotation = Quaternion(
                outward_axis.normalized(),
                math.radians(ARM_OUTWARD_DEGREES),
            )
            for source_name, target_name in ordered_mapping:
                if bone_suffix(target_name) in suffixes:
                    arm_outward_rotations[source_name] = (
                        rotation_only_matrix(outward_rotation)
                    )
            log.append(
                f"  Arm outward {side}: "
                f"degrees={ARM_OUTWARD_DEGREES:.9f}, "
                f"axis={vec(outward_axis.normalized())}"
            )

    first_root_positions = {
        name: pose_matrix_world(source, name).translation.copy()
        for name in root_names
    }
    target_root_rest_positions = {
        name: rest_matrix_world(target, mapping[name]).translation.copy()
        for name in root_names
    }
    sample_frames = {
        frame_start,
        frame_start + (frame_end - frame_start) // 2,
        frame_end,
    }
    angular_errors = {
        source_name: []
        for source_name, _ in ordered_mapping
    }
    sample_lines = []

    for frame in range(frame_start, frame_end + 1, FRAME_STEP):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        for source_name, target_name in ordered_mapping:
            source_pose = pose_matrix_world(source, source_name)
            if source_name in initial_direction_calibrations:
                desired_world = (
                    rotation_only_matrix(root_conversion_rotation)
                    @ source_pose
                    @ initial_direction_calibrations[source_name]
                )
            else:
                desired_world = rest_conversions[source_name] @ source_pose
            if source_name in shoulder_slope_rotations:
                desired_world = (
                    shoulder_slope_rotations[source_name] @ desired_world
                )
            if source_name in arm_outward_rotations:
                desired_world = (
                    arm_outward_rotations[source_name] @ desired_world
                )
            set_rotation_from_world_matrix(target, target_name, desired_world)
            bpy.context.view_layer.update()

            target_bone = target.pose.bones[target_name]
            actual_world = pose_matrix_world(target, target_name)
            angular_error = actual_world.to_quaternion().rotation_difference(
                desired_world.to_quaternion()
            ).angle
            angular_errors[source_name].append(math.degrees(angular_error))

            if (
                frame in sample_frames
                and bone_suffix(target_name) in DIAGNOSTIC_BONE_SUFFIXES
            ):
                sample_lines.extend([
                    f"  FRAME {frame} / {source_name} -> {target_name}",
                    *transform_log("source_pose_world", source_pose, "    "),
                    *transform_log("desired_target_world", desired_world, "    "),
                    *transform_log("actual_target_world", actual_world, "    "),
                    f"    angular_error_degrees: "
                    f"{math.degrees(angular_error):.9f}",
                ])

            key_rotation(target_bone, frame)

            if COPY_ROOT_LOCATION and source_name in root_names:
                source_delta = (
                    source_pose.translation - first_root_positions[source_name]
                )
                desired_root_world = (
                    target_root_rest_positions[source_name] + source_delta
                )
                desired_root_armature = (
                    target.matrix_world.inverted_safe() @ desired_root_world
                )
                current_matrix = target_bone.matrix.copy()
                current_matrix.translation = desired_root_armature
                target_bone.matrix = current_matrix
                target_bone.keyframe_insert(
                    data_path="location",
                    frame=frame,
                    group=target_bone.name,
                )

    action = target.animation_data.action
    set_linear_interpolation(action)
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(original_frame)
    bpy.context.view_layer.update()

    log.extend(["", "ROTATION ERROR SUMMARY"])
    for source_name, target_name in ordered_mapping:
        errors = angular_errors[source_name]
        log.append(
            f"  {source_name} -> {target_name}: "
            f"mean_degrees={sum(errors) / len(errors):.9f}, "
            f"max_degrees={max(errors):.9f}"
        )
    log.extend(["", "ARM SAMPLE POSES", *sample_lines])
    if muted_constraints:
        log.extend(["", "MUTED CONSTRAINTS"])
        log.extend(f"  {item}" for item in muted_constraints)
    log.extend(["", "STATUS: SUCCESS"])

    print("Retarget complete")
    print(f"  source: {source.name}")
    print(f"  target: {target.name}")
    print(f"  frames: {frame_start} .. {frame_end}")
    print(f"  mapped bones: {len(mapping)}")
    print(f"  previous target action: {old_action_name}")
    print(f"  muted constraints: {len(muted_constraints)}")
    if muted_constraints:
        print("  Muted constraints:")
        for item in muted_constraints:
            print(f"    {item}")


log_lines = []
try:
    main(log_lines)
except Exception:
    log_lines.extend([
        "",
        "STATUS: FAILED",
        traceback.format_exc(),
    ])
    write_log(log_lines)
    raise
else:
    write_log(log_lines)
