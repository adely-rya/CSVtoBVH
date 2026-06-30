import math

import bpy
from mathutils import Matrix, Vector


# Input BVH armature and the character rig to animate.
SOURCE_ARMATURE_NAME = "output_plus_start60_blender"
TARGET_ARMATURE_NAME = "Armature"

PREFIX = "mixamorig:"
FRAME_STEP = 1

LEG_BONES = {
    "Left": {
        "upper": f"{PREFIX}LeftUpLeg",
        "lower": f"{PREFIX}LeftLeg",
        "foot": f"{PREFIX}LeftFoot",
    },
    "Right": {
        "upper": f"{PREFIX}RightUpLeg",
        "lower": f"{PREFIX}RightLeg",
        "foot": f"{PREFIX}RightFoot",
    },
}

GENERATED_PREFIX = "FM_LEG_"


def get_armature(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Armature object not found: {name}")
    if obj.type != "ARMATURE":
        raise RuntimeError(f"Object is not an armature: {name}")
    return obj


def require_bones(armature):
    for side_bones in LEG_BONES.values():
        for name in side_bones.values():
            if armature.pose.bones.get(name) is None:
                raise RuntimeError(f"Bone not found: {armature.name} / {name}")


def normalized(vector, label):
    vector = Vector(vector)
    if vector.length < 1.0e-8:
        raise RuntimeError(f"Cannot calculate body basis: {label} is zero length")
    return vector.normalized()


def body_basis(left_hip, right_hip, left_ankle, right_ankle):
    hip_center = (left_hip + right_hip) * 0.5
    ankle_center = (left_ankle + right_ankle) * 0.5

    # X points from the anatomical right hip toward the left hip.
    x_axis = normalized(left_hip - right_hip, "hip width")
    up_hint = normalized(hip_center - ankle_center, "body height")
    forward = normalized(x_axis.cross(up_hint), "body forward")
    up_axis = normalized(forward.cross(x_axis), "body up")

    # Matrix constructor takes rows, so transpose to store the axes as columns.
    return Matrix((x_axis, up_axis, forward)).transposed()


def pose_head(armature, bone_name):
    return armature.pose.bones[bone_name].head.copy()


def rest_head(armature, bone_name):
    return armature.data.bones[bone_name].head_local.copy()


def source_points(source):
    points = {}
    for side, names in LEG_BONES.items():
        points[side] = {
            "hip": pose_head(source, names["upper"]),
            "knee": pose_head(source, names["lower"]),
            "ankle": pose_head(source, names["foot"]),
        }
    return points


def target_rest_points(target):
    points = {}
    for side, names in LEG_BONES.items():
        points[side] = {
            "hip": rest_head(target, names["upper"]),
            "knee": rest_head(target, names["lower"]),
            "ankle": rest_head(target, names["foot"]),
        }
    return points


def average_leg_length(points):
    lengths = []
    for side in ("Left", "Right"):
        hip = points[side]["hip"]
        knee = points[side]["knee"]
        ankle = points[side]["ankle"]
        lengths.append((knee - hip).length + (ankle - knee).length)
    return sum(lengths) / len(lengths)


def ensure_empty(name, display_type, size):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(obj)
    obj.parent = None
    obj.empty_display_type = display_type
    obj.empty_display_size = size
    if obj.animation_data:
        obj.animation_data_clear()
    return obj


def mute_conflicting_leg_constraints(target):
    muted = []
    for names in LEG_BONES.values():
        for bone_name in names.values():
            pose_bone = target.pose.bones[bone_name]
            for constraint in pose_bone.constraints:
                if constraint.name.startswith(GENERATED_PREFIX):
                    pose_bone.constraints.remove(constraint)
                elif not constraint.mute:
                    constraint.mute = True
                    muted.append(f"{bone_name} / {constraint.name}")
    return muted


def target_world_position(target, armature_local_position):
    return target.matrix_world @ armature_local_position


def set_empty_world_position(empty, world_position):
    empty.location = world_position


def pole_position(hip, knee, ankle, distance):
    chain = ankle - hip
    if chain.length < 1.0e-8:
        return knee.copy()

    closest = hip + chain * ((knee - hip).dot(chain) / chain.length_squared)
    direction = knee - closest
    if direction.length < 1.0e-8:
        direction = Vector((0.0, 0.0, 1.0))
    return knee + direction.normalized() * distance


def source_frame_range(source):
    if (
        source.animation_data is None
        or source.animation_data.action is None
    ):
        raise RuntimeError(f"Source armature has no active action: {source.name}")
    start, end = source.animation_data.action.frame_range
    return int(math.ceil(start)), int(math.floor(end))


def detach_target_animation(target):
    old_action_name = "(none)"
    if target.animation_data is not None:
        if target.animation_data.action is not None:
            old_action_name = target.animation_data.action.name
            target.animation_data.action = None
        target.animation_data.use_nla = False

    for names in LEG_BONES.values():
        for bone_name in names.values():
            target.pose.bones[bone_name].matrix_basis = Matrix.Identity(4)

    bpy.context.view_layer.update()
    return old_action_name


def find_best_pole_angle(target, lower_bone_name, desired_knee_local):
    pose_bone = target.pose.bones[lower_bone_name]
    constraint = pose_bone.constraints[f"{GENERATED_PREFIX}IK"]
    candidates = (
        0.0,
        math.radians(90.0),
        math.radians(-90.0),
        math.radians(180.0),
    )

    best_angle = 0.0
    best_error = float("inf")
    for angle in candidates:
        constraint.pole_angle = angle
        bpy.context.view_layer.update()
        error = (pose_bone.head - desired_knee_local).length
        if error < best_error:
            best_error = error
            best_angle = angle

    constraint.pole_angle = best_angle
    bpy.context.view_layer.update()
    return math.degrees(best_angle), best_error


def set_linear_interpolation(obj):
    if obj.animation_data is None or obj.animation_data.action is None:
        return
    for curve in obj.animation_data.action.fcurves:
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def main():
    source = get_armature(SOURCE_ARMATURE_NAME)
    target = get_armature(TARGET_ARMATURE_NAME)
    require_bones(source)
    require_bones(target)

    old_target_action = detach_target_animation(target)
    frame_start, frame_end = source_frame_range(source)
    scene = bpy.context.scene
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    source_reference = source_points(source)
    target_reference = target_rest_points(target)

    source_basis = body_basis(
        source_reference["Left"]["hip"],
        source_reference["Right"]["hip"],
        source_reference["Left"]["ankle"],
        source_reference["Right"]["ankle"],
    )
    target_basis = body_basis(
        target_reference["Left"]["hip"],
        target_reference["Right"]["hip"],
        target_reference["Left"]["ankle"],
        target_reference["Right"]["ankle"],
    )
    source_to_target_rotation = target_basis @ source_basis.transposed()

    source_leg_length = average_leg_length(source_reference)
    target_leg_length = average_leg_length(target_reference)
    if source_leg_length < 1.0e-8:
        raise RuntimeError("Source leg length is zero")
    scale = target_leg_length / source_leg_length

    target_hip_center = (
        target_reference["Left"]["hip"] + target_reference["Right"]["hip"]
    ) * 0.5
    pole_distance = target_leg_length * 0.75

    muted = mute_conflicting_leg_constraints(target)

    controls = {}
    for side, names in LEG_BONES.items():
        ankle_target = ensure_empty(
            f"{GENERATED_PREFIX}{side}_ANKLE",
            "SPHERE",
            target_leg_length * 0.035,
        )
        knee_pole = ensure_empty(
            f"{GENERATED_PREFIX}{side}_KNEE_POLE",
            "PLAIN_AXES",
            target_leg_length * 0.06,
        )
        controls[side] = {"ankle": ankle_target, "pole": knee_pole}

        lower_bone = target.pose.bones[names["lower"]]
        ik = lower_bone.constraints.new(type="IK")
        ik.name = f"{GENERATED_PREFIX}IK"
        ik.target = ankle_target
        ik.pole_target = knee_pole
        ik.chain_count = 2
        ik.use_rotation = False
        ik.influence = 1.0

    first_mapped = {}
    for frame in range(frame_start, frame_end + 1, FRAME_STEP):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        points = source_points(source)
        source_hip_center = (
            points["Left"]["hip"] + points["Right"]["hip"]
        ) * 0.5

        mapped = {}
        for side in ("Left", "Right"):
            mapped[side] = {}
            for joint in ("hip", "knee", "ankle"):
                relative = points[side][joint] - source_hip_center
                mapped[side][joint] = (
                    target_hip_center
                    + (source_to_target_rotation @ relative) * scale
                )

            ankle_local = mapped[side]["ankle"]
            pole_local = pole_position(
                mapped[side]["hip"],
                mapped[side]["knee"],
                ankle_local,
                pole_distance,
            )
            set_empty_world_position(
                controls[side]["ankle"],
                target_world_position(target, ankle_local),
            )
            set_empty_world_position(
                controls[side]["pole"],
                target_world_position(target, pole_local),
            )
            controls[side]["ankle"].keyframe_insert("location", frame=frame)
            controls[side]["pole"].keyframe_insert("location", frame=frame)

        if frame == frame_start:
            first_mapped = mapped

    for side_controls in controls.values():
        set_linear_interpolation(side_controls["ankle"])
        set_linear_interpolation(side_controls["pole"])

    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    pole_results = {}
    for side, names in LEG_BONES.items():
        pole_results[side] = find_best_pole_angle(
            target,
            names["lower"],
            first_mapped[side]["knee"],
        )

    print("Leg-only retarget setup completed.")
    print(f"Source: {source.name}")
    print(f"Target: {target.name}")
    print(f"Frames: {frame_start} - {frame_end}")
    print(f"Scale: {scale:.6f}")
    print(f"Detached target action: {old_target_action}")
    print(f"Muted old leg constraints: {len(muted)}")
    for side, (angle, error) in pole_results.items():
        print(f"{side} pole angle: {angle:.1f} deg, reference error: {error:.6f}")
    print("Only upper/lower legs are driven. Hips, spine, arms and foot rotation are unchanged.")


main()
