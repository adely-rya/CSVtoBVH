import bpy
from mathutils import Vector
import math


# ============================================================
# 設定
# ============================================================

SOURCE_ARMATURE_NAME = "freemocap_mixamo"      # BVH側アーマチュア名に変更
TARGET_ARMATURE_NAME = "Rigged Character Armature.002"             # アバター側Armature名に変更

SRC_LEFT_ARM = "mixamorig:LeftArm"
SRC_LEFT_FOREARM = "mixamorig:LeftForeArm"
SRC_LEFT_HAND = "mixamorig:LeftHand"

TGT_LEFT_ARM = "mixamorig:LeftArm"
TGT_LEFT_FOREARM = "mixamorig:LeftForeArm"
TGT_LEFT_HAND = "mixamorig:LeftHand"

# Empty名
HAND_TARGET_NAME = "FM_Target_LeftHand"
ELBOW_POLE_NAME = "FM_Pole_LeftElbow"

# フレーム範囲
FRAME_START = bpy.context.scene.frame_start
FRAME_END = bpy.context.scene.frame_end

# 手動スケール。Noneなら腕長から自動推定
RETARGET_SCALE = None

# ソースの動きの反転が必要な場合に変更
FLIP_X = False
FLIP_Y = False
FLIP_Z = False

# 左手IKのpole angle。肘が逆に曲がる場合はここを変える
# よく試す値: 0, math.radians(90), math.radians(-90), math.radians(180)
LEFT_ARM_POLE_ANGLE = 0.0

# IKをBakeするか
BAKE_AFTER_SETUP = False

# Bake後にIK制約を消すか
CLEAR_CONSTRAINTS_AFTER_BAKE = False


# ============================================================
# Utility
# ============================================================

def get_obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError(f"Object not found: {name}")
    return obj


def get_pose_bone(arm_obj, bone_name):
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        raise RuntimeError(f"Pose bone not found: {arm_obj.name} / {bone_name}")
    return pb


def pose_bone_head_world(arm_obj, bone_name):
    pb = get_pose_bone(arm_obj, bone_name)
    return arm_obj.matrix_world @ pb.head


def pose_bone_tail_world(arm_obj, bone_name):
    pb = get_pose_bone(arm_obj, bone_name)
    return arm_obj.matrix_world @ pb.tail


def pose_bone_matrix_world(arm_obj, bone_name):
    pb = get_pose_bone(arm_obj, bone_name)
    return arm_obj.matrix_world @ pb.matrix


def bone_length_world(arm_obj, bone_name):
    return (pose_bone_tail_world(arm_obj, bone_name) - pose_bone_head_world(arm_obj, bone_name)).length


def ensure_empty(name, display_type="SPHERE", size=0.08):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        bpy.context.collection.objects.link(obj)
    obj.empty_display_type = display_type
    obj.empty_display_size = size
    return obj


def clear_object_animation(obj):
    if obj.animation_data:
        obj.animation_data_clear()


def set_linear_interpolation(obj):
    if obj.animation_data is None or obj.animation_data.action is None:
        return

    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def apply_axis_flips(v):
    v = Vector(v)
    if FLIP_X:
        v.x *= -1.0
    if FLIP_Y:
        v.y *= -1.0
    if FLIP_Z:
        v.z *= -1.0
    return v


def remove_existing_constraint(pb, name):
    for c in list(pb.constraints):
        if c.name == name:
            pb.constraints.remove(c)


# ============================================================
# Main
# ============================================================

def main():
    source = get_obj(SOURCE_ARMATURE_NAME)
    target = get_obj(TARGET_ARMATURE_NAME)

    print("Source:", source.name)
    print("Target:", target.name)

    # 必要ボーン確認
    for b in [SRC_LEFT_ARM, SRC_LEFT_FOREARM, SRC_LEFT_HAND]:
        get_pose_bone(source, b)

    for b in [TGT_LEFT_ARM, TGT_LEFT_FOREARM, TGT_LEFT_HAND]:
        get_pose_bone(target, b)

    # 初期フレーム
    bpy.context.scene.frame_set(FRAME_START)
    bpy.context.view_layer.update()

    # ソース左手初期位置
    # LeftHandのheadはだいたい手首位置。tailは手先寄り。
    # まずはhead基準が安全。
    src_hand_initial = pose_bone_head_world(source, SRC_LEFT_HAND)

    # ターゲット左手初期位置
    tgt_hand_initial = pose_bone_head_world(target, TGT_LEFT_HAND)

    # スケール推定
    if RETARGET_SCALE is None:
        src_arm_len = bone_length_world(source, SRC_LEFT_ARM) + bone_length_world(source, SRC_LEFT_FOREARM)
        tgt_arm_len = bone_length_world(target, TGT_LEFT_ARM) + bone_length_world(target, TGT_LEFT_FOREARM)

        if src_arm_len < 1e-8:
            scale = 1.0
        else:
            scale = tgt_arm_len / src_arm_len
    else:
        scale = RETARGET_SCALE

    print("Retarget scale:", scale)

    # Empty作成
    hand_target = ensure_empty(HAND_TARGET_NAME, display_type="SPHERE", size=0.10)
    elbow_pole = ensure_empty(ELBOW_POLE_NAME, display_type="PLAIN_AXES", size=0.15)

    clear_object_animation(hand_target)
    clear_object_animation(elbow_pole)

    # 各フレームで左手ターゲットと肘Poleを打つ
    for frame in range(FRAME_START, FRAME_END + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()

        src_hand = pose_bone_head_world(source, SRC_LEFT_HAND)
        src_elbow = pose_bone_head_world(source, SRC_LEFT_FOREARM)

        # ソース初期位置からの相対移動だけをターゲットに乗せる
        delta_hand = apply_axis_flips(src_hand - src_hand_initial) * scale
        delta_elbow = apply_axis_flips(src_elbow - src_hand_initial) * scale

        hand_target.location = tgt_hand_initial + delta_hand

        # Poleは肘位置をそのまま使うより、少し外側に出す方が安定することがある
        # まずはソース肘位置ベースで置く
        elbow_pole.location = tgt_hand_initial + delta_elbow

        hand_target.keyframe_insert(data_path="location", frame=frame)
        elbow_pole.keyframe_insert(data_path="location", frame=frame)

    set_linear_interpolation(hand_target)
    set_linear_interpolation(elbow_pole)

    # ターゲット左手ボーンにIKを追加
    tgt_hand_pb = get_pose_bone(target, TGT_LEFT_HAND)

    remove_existing_constraint(tgt_hand_pb, "FM_LeftHand_IK")

    ik = tgt_hand_pb.constraints.new(type="IK")
    ik.name = "FM_LeftHand_IK"
    ik.target = hand_target
    ik.pole_target = elbow_pole
    ik.pole_angle = LEFT_ARM_POLE_ANGLE
    ik.chain_count = 2
    ik.use_rotation = False

    print("Added IK constraint to:", TGT_LEFT_HAND)
    print("IK target:", HAND_TARGET_NAME)
    print("Pole target:", ELBOW_POLE_NAME)

    # Bakeする場合
    if BAKE_AFTER_SETUP:
        bpy.context.view_layer.objects.active = target
        target.select_set(True)

        bpy.ops.nla.bake(
            frame_start=FRAME_START,
            frame_end=FRAME_END,
            only_selected=False,
            visual_keying=True,
            clear_constraints=CLEAR_CONSTRAINTS_AFTER_BAKE,
            clear_parents=False,
            use_current_action=True,
            bake_types={"POSE"},
        )

        print("Baked target armature animation.")

    bpy.context.scene.frame_set(FRAME_START)
    bpy.context.view_layer.update()

    print("Done. Play timeline and check left hand motion.")
    print("If elbow bends wrong, change LEFT_ARM_POLE_ANGLE to 90, -90, or 180 degrees.")


main()