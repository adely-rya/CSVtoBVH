import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# =========================
# MediaPipe / FreeMoCap names
# =========================

POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

FACE_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "mouth_left", "mouth_right",
]


def canonical_key(name):
    text = str(name).strip().lower()
    text = re.sub(r"^(body|pose|face|landmark|mediapipe)[_\-\.\s]+", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


ALIASES = {}
for name in POSE_NAMES + HAND_NAMES + FACE_NAMES:
    ALIASES[canonical_key(name)] = name

ALIASES.update({
    "lefthandmiddle": "left_hand_middle",
    "righthandmiddle": "right_hand_middle",
    "headcenter": "head_center",
    "neckcenter": "neck_center",
    "trunkcenter": "trunk_center",
    "hipscenter": "hips_center",
    "centerofmass": "center_of_mass",
    "totalbodycom": "center_of_mass",
    "com": "center_of_mass",
    "leftwrist": "left_wrist",
    "rightwrist": "right_wrist",
    "leftfootindex": "left_foot_index",
    "rightfootindex": "right_foot_index",
})


def normalize_landmark_name(name, source=None, side=None):
    raw = str(name).strip().lstrip("#").strip()
    key = canonical_key(raw)
    base = ALIASES.get(key, raw.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_"))
    base = re.sub(r"__+", "_", base).strip("_")

    if source == "hand" and side in {"left", "right"}:
        if base.startswith(f"{side}_hand_"):
            return base
        if base.startswith(f"{side}_"):
            base = base[len(side) + 1:]
        return f"{side}_hand_{base}"

    if source == "face" and base not in POSE_NAMES:
        return f"face_{base}"

    return base


# =========================
# BVH skeleton definition
# =========================

PARENTS_SIMPLE = {
    "Hips": None,
    "Spine": "Hips",
    "Chest": "Spine",
    "Neck": "Chest",
    "Head": "Neck",

    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "LeftHandEnd": "LeftHand",

    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
    "RightHandEnd": "RightHand",

    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToe": "LeftFoot",

    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToe": "RightFoot",
}

PARENTS_MIXAMO = {
    "Hips": None,

    "Spine": "Hips",
    "Spine1": "Spine",
    "Spine2": "Spine1",
    "Neck": "Spine2",
    "Head": "Neck",

    "LeftShoulder": "Spine2",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",

    "RightShoulder": "Spine2",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",

    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToeBase": "LeftFoot",

    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToeBase": "RightFoot",
}

PARENTS = dict(PARENTS_MIXAMO)
CHILDREN = {}
CHANNEL_JOINTS = []


def make_children(parents):
    children = {}
    for joint, parent in parents.items():
        children.setdefault(joint, [])
        if parent is not None:
            children.setdefault(parent, []).append(joint)
    return children


def set_rig_preset(preset):
    global PARENTS, CHILDREN, CHANNEL_JOINTS
    if preset == "simple":
        PARENTS = dict(PARENTS_SIMPLE)
    elif preset == "mixamo":
        PARENTS = dict(PARENTS_MIXAMO)
    else:
        raise ValueError(f"Unknown rig preset: {preset}")
    CHILDREN = make_children(PARENTS)
    CHANNEL_JOINTS = list(PARENTS.keys())


set_rig_preset("mixamo")


# =========================
# Basic math
# =========================

def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def axis_angle_to_matrix(axis, angle):
    axis = normalize(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    return np.array([
        [t*x*x + c,     t*x*y - s*z,   t*x*z + s*y],
        [t*x*y + s*z,   t*y*y + c,     t*y*z - s*x],
        [t*x*z - s*y,   t*y*z + s*x,   t*z*z + c],
    ])


def rotation_matrix_from_vectors(a, b):
    a = normalize(a)
    b = normalize(b)

    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return np.eye(3)

    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if c > 0.999999:
        return np.eye(3)

    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return axis_angle_to_matrix(axis, math.pi)

    s = np.linalg.norm(v)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def orthonormalize(r):
    u, _, vh = np.linalg.svd(np.asarray(r, dtype=float))
    out = u @ vh
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def basis_from_yx(y_axis, x_hint):
    y = normalize(y_axis)
    if np.linalg.norm(y) < 1e-8:
        return None

    x = np.asarray(x_hint, dtype=float)
    x = x - y * np.dot(x, y)
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x, y)) > 0.9:
            x = np.array([0.0, 0.0, 1.0])
        x = x - y * np.dot(x, y)

    x = normalize(x)
    z = normalize(np.cross(x, y))
    x = normalize(np.cross(y, z))
    return orthonormalize(np.column_stack([x, y, z]))


def basis_from_yz(y_axis, z_hint):
    y = normalize(y_axis)
    if np.linalg.norm(y) < 1e-8:
        return None

    z = np.asarray(z_hint, dtype=float)
    z = z - y * np.dot(z, y)
    if np.linalg.norm(z) < 1e-8:
        z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(z, y)) > 0.9:
            z = np.array([1.0, 0.0, 0.0])
        z = z - y * np.dot(z, y)

    z = normalize(z)
    x = normalize(np.cross(y, z))
    z = normalize(np.cross(x, y))
    return orthonormalize(np.column_stack([x, y, z]))


def rotation_from_bases(rest_basis, current_basis):
    if rest_basis is None or current_basis is None:
        return np.eye(3)
    return orthonormalize(current_basis @ rest_basis.T)


def matrix_to_euler_zxy_degrees(r):
    r = orthonormalize(r)
    sx = max(-1.0, min(1.0, r[2, 1]))
    x = math.asin(sx)
    cx = math.cos(x)

    if abs(cx) > 1e-8:
        z = math.atan2(-r[0, 1], r[1, 1])
        y = math.atan2(-r[2, 0], r[2, 2])
    else:
        z = math.atan2(r[1, 0], r[0, 0])
        y = 0.0

    return np.degrees([z, x, y])


def moving_average(data, window):
    if window <= 1:
        return data

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(data, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(data)

    for i in range(data.shape[0]):
        out[i] = padded[i:i + window].mean(axis=0)

    return out


def euler_xyz_to_matrix_degrees(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0):
    rx = axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), math.radians(x_degrees))
    ry = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), math.radians(y_degrees))
    rz = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), math.radians(z_degrees))
    return rz @ ry @ rx


# =========================
# Loading and normalization
# =========================

def ensure_frame_count(frames, count):
    while len(frames) < count:
        frames.append({})


def merge_frames(base, extra):
    ensure_frame_count(base, len(extra))
    for i, frame in enumerate(extra):
        base[i].update(frame)


def parse_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out


def sniff_csv_columns(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or []


def read_csv_frames(path, source=None, side=None):
    columns = sniff_csv_columns(path)
    column_map = {canonical_key(c): c for c in columns}
    frame_col = column_map.get("frame") or column_map.get("framenumber") or column_map.get("frameindex")
    name_col = column_map.get("name") or column_map.get("landmark") or column_map.get("landmarkname")
    x_col = column_map.get("x")
    y_col = column_map.get("y")
    z_col = column_map.get("z")

    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    frames = []

    if name_col and x_col and y_col and z_col:
        for row_index, row in enumerate(rows):
            frame_index = int(parse_float(row.get(frame_col, row_index))) if frame_col else row_index
            ensure_frame_count(frames, frame_index + 1)
            name = normalize_landmark_name(row.get(name_col, ""), source=source, side=side)
            frames[frame_index][name] = np.array([
                parse_float(row.get(x_col)),
                parse_float(row.get(y_col)),
                parse_float(row.get(z_col)),
            ], dtype=float)
        return frames

    groups = defaultdict(dict)
    for col in columns:
        clean = col.strip().lstrip("#").strip()
        match = re.match(r"(.+?)[_\-\.\s]+([xyzXYZ])$", clean)
        if not match:
            continue
        stem, axis = match.group(1), match.group(2).lower()
        groups[stem][axis] = col

    for row_index, row in enumerate(rows):
        frame_index = int(parse_float(row.get(frame_col, row_index))) if frame_col else row_index
        ensure_frame_count(frames, frame_index + 1)

        for stem, axes in groups.items():
            if not {"x", "y", "z"}.issubset(axes):
                continue
            name = normalize_landmark_name(stem, source=source, side=side)
            frames[frame_index][name] = np.array([
                parse_float(row.get(axes["x"])),
                parse_float(row.get(axes["y"])),
                parse_float(row.get(axes["z"])),
            ], dtype=float)

    return frames


def read_npy_frames(path, source=None, side=None):
    data = np.load(path)

    if source == "com" and data.ndim == 2 and data.shape[1] == 3:
        return [{"center_of_mass": data[frame_index].astype(float)} for frame_index in range(data.shape[0])]

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected npy shape frames x landmarks x 3, got {data.shape}")

    if source == "hand":
        names = HAND_NAMES
    elif source == "com":
        names = ["center_of_mass"]
    elif source == "face":
        names = [f"face_{i}" for i in range(data.shape[1])]
    else:
        names = POSE_NAMES

    frames = []
    for frame_index in range(data.shape[0]):
        frame = {}
        for i in range(min(data.shape[1], len(names))):
            frame[normalize_landmark_name(names[i], source=source, side=side)] = data[frame_index, i].astype(float)
        frames.append(frame)
    return frames


def read_landmark_file(path, source=None, side=None):
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".npy":
        return read_npy_frames(path, source=source, side=side)
    return read_csv_frames(path, source=source, side=side)


def convert_coordinate_system(frames, scale=1.0, swap_yz=True, flip_x=False, flip_y=False, flip_z=False):
    out = []
    rot = np.array([1.0, 1.0, 1.0], dtype=float)
    if flip_x:
        rot[0] = -1.0
    if flip_y:
        rot[1] = -1.0
    if flip_z:
        rot[2] = -1.0

    for frame in frames:
        next_frame = {}
        for name, value in frame.items():
            point = np.asarray(value, dtype=float) * scale
            if swap_yz:
                point = point[[0, 2, 1]]
            next_frame[name] = point * rot
        out.append(next_frame)
    return out


def rotate_frames(frames, x_degrees=0.0, y_degrees=0.0, z_degrees=0.0):
    if abs(x_degrees) < 1e-8 and abs(y_degrees) < 1e-8 and abs(z_degrees) < 1e-8:
        return frames
    r = euler_xyz_to_matrix_degrees(x_degrees, y_degrees, z_degrees)
    return [{name: r @ point for name, point in frame.items()} for frame in frames]


def frame_keys(frames):
    keys = set()
    for frame in frames:
        keys.update(frame.keys())
    return keys


def fill_smooth_frames(frames, keys=None, window=1, max_joint_speed=None):
    if not frames:
        return frames

    keys = sorted(keys or frame_keys(frames))
    n = len(frames)
    arrays = {}

    for key in keys:
        data = np.full((n, 3), np.nan, dtype=float)
        for i, frame in enumerate(frames):
            if key in frame:
                data[i] = frame[key]

        valid = np.isfinite(data).all(axis=1)
        if not valid.any():
            continue

        idx = np.arange(n)
        for axis in range(3):
            data[:, axis] = np.interp(idx, idx[valid], data[valid, axis])

        if max_joint_speed is not None and max_joint_speed > 0:
            for i in range(1, n):
                if np.linalg.norm(data[i] - data[i - 1]) > max_joint_speed:
                    data[i] = data[i - 1]

        data = moving_average(data, window)
        arrays[key] = data

    out = [dict(frame) for frame in frames]
    for key, data in arrays.items():
        for i in range(n):
            out[i][key] = data[i]
    return out


# =========================
# Intermediate skeleton
# =========================

def point(frame, names, default=None):
    for name in names:
        if name in frame and np.isfinite(frame[name]).all():
            return frame[name]
    return default


def blend_points(points):
    valid = [p for p in points if p is not None and np.isfinite(p).all()]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def extend_from_wrist(wrist, fingertip, amount=0.35):
    if wrist is None or fingertip is None:
        return fingertip
    direction = fingertip - wrist
    if np.linalg.norm(direction) < 1e-6:
        return fingertip
    return fingertip + direction * amount


def hand_end(frame, side, wrist, body_fallback):
    prefix = f"{side}_hand_"
    candidates = [
        point(frame, [prefix + "middle_finger_tip", prefix + "middle_tip"]),
        point(frame, [prefix + "index_finger_tip", prefix + "index_tip"]),
        point(frame, [prefix + "ring_finger_tip", prefix + "ring_tip"]),
        point(frame, [prefix + "pinky_tip"]),
        point(frame, [prefix + "thumb_tip"]),
    ]
    avg = blend_points(candidates)
    if avg is not None:
        if wrist is not None and np.linalg.norm(avg - body_fallback) < 1e-6:
            return extend_from_wrist(wrist, body_fallback)
        return avg
    return extend_from_wrist(wrist, body_fallback)


def build_intermediate_positions(frame, com_blend=0.75):
    left_hip = point(frame, ["left_hip"])
    right_hip = point(frame, ["right_hip"])
    left_shoulder = point(frame, ["left_shoulder"])
    right_shoulder = point(frame, ["right_shoulder"])

    if any(p is None for p in [left_hip, right_hip, left_shoulder, right_shoulder]):
        raise ValueError("Missing required pose landmarks: hips and shoulders are needed.")

    hips_center = (left_hip + right_hip) / 2.0
    com = point(frame, ["center_of_mass", "total_body_com", "com"])
    hips = hips_center if com is None else com_blend * hips_center + (1.0 - com_blend) * com

    chest = (left_shoulder + right_shoulder) / 2.0
    spine = hips * 0.55 + chest * 0.45
    spine_mixamo = hips_center * 0.75 + chest * 0.25
    spine1_mixamo = hips_center * 0.50 + chest * 0.50
    spine2_mixamo = hips_center * 0.20 + chest * 0.80

    nose = point(frame, ["nose", "face_nose"], chest + np.array([0.0, 0.2, 0.0]))
    neck = point(frame, ["neck_center"], chest * 0.75 + nose * 0.25)
    head_center = blend_points([
        point(frame, ["head_center"]),
        nose,
        point(frame, ["left_eye", "face_left_eye"]),
        point(frame, ["right_eye", "face_right_eye"]),
    ])
    head = head_center if head_center is not None else nose

    left_wrist = point(frame, ["left_wrist"])
    right_wrist = point(frame, ["right_wrist"])
    left_hand = point(frame, ["left_index", "left_hand_index_finger_tip"], left_wrist)
    right_hand = point(frame, ["right_index", "right_hand_index_finger_tip"], right_wrist)
    left_hand_end = hand_end(frame, "left", left_wrist, left_hand)
    right_hand_end = hand_end(frame, "right", right_wrist, right_hand)

    left_ankle = point(frame, ["left_ankle"])
    right_ankle = point(frame, ["right_ankle"])
    left_toe = point(frame, ["left_foot_index"], left_ankle)
    right_toe = point(frame, ["right_foot_index"], right_ankle)

    positions = {
        "Hips": hips,
        "Spine": spine_mixamo,
        "Spine1": spine1_mixamo,
        "Spine2": spine2_mixamo,
        "Chest": chest,
        "Neck": neck,
        "Head": head,

        "LeftShoulder": left_shoulder,
        "LeftArm": point(frame, ["left_elbow"]),
        "LeftForeArm": left_wrist,
        "LeftHand": left_hand,
        "LeftHandEnd": left_hand_end,

        "RightShoulder": right_shoulder,
        "RightArm": point(frame, ["right_elbow"]),
        "RightForeArm": right_wrist,
        "RightHand": right_hand,
        "RightHandEnd": right_hand_end,

        "LeftUpLeg": left_hip,
        "LeftLeg": point(frame, ["left_knee"]),
        "LeftFoot": left_ankle,
        "LeftToe": left_toe,
        "LeftToeBase": left_toe,

        "RightUpLeg": right_hip,
        "RightLeg": point(frame, ["right_knee"]),
        "RightFoot": right_ankle,
        "RightToe": right_toe,
        "RightToeBase": right_toe,

        "_left_hip": left_hip,
        "_right_hip": right_hip,
        "_left_shoulder": left_shoulder,
        "_right_shoulder": right_shoulder,
        "_left_heel": point(frame, ["left_heel"], left_ankle),
        "_right_heel": point(frame, ["right_heel"], right_ankle),
        "_left_wrist": left_wrist,
        "_right_wrist": right_wrist,
        "_left_index": point(frame, ["left_index", "left_hand_index_finger_tip"], left_hand),
        "_right_index": point(frame, ["right_index", "right_hand_index_finger_tip"], right_hand),
        "_left_pinky": point(frame, ["left_pinky", "left_hand_pinky_mcp", "left_hand_pinky_tip"], left_hand),
        "_right_pinky": point(frame, ["right_pinky", "right_hand_pinky_mcp", "right_hand_pinky_tip"], right_hand),
        "_left_thumb": point(frame, ["left_thumb", "left_hand_thumb_tip"], left_hand),
        "_right_thumb": point(frame, ["right_thumb", "right_hand_thumb_tip"], right_hand),
        "_left_hand_index_mcp": point(frame, ["left_hand_index_finger_mcp"]),
        "_right_hand_index_mcp": point(frame, ["right_hand_index_finger_mcp"]),
        "_left_hand_pinky_mcp": point(frame, ["left_hand_pinky_mcp"]),
        "_right_hand_pinky_mcp": point(frame, ["right_hand_pinky_mcp"]),
        "_left_hand_middle": point(frame, ["left_hand_middle_finger_mcp", "left_hand_middle_finger_tip"]),
        "_right_hand_middle": point(frame, ["right_hand_middle_finger_mcp", "right_hand_middle_finger_tip"]),
        "_nose": nose,
        "_left_eye": point(frame, ["left_eye", "face_left_eye"]),
        "_right_eye": point(frame, ["right_eye", "face_right_eye"]),
        "_left_ear": point(frame, ["left_ear", "face_left_ear"]),
        "_right_ear": point(frame, ["right_ear", "face_right_ear"]),
    }

    missing = [joint for joint in PARENTS if positions.get(joint) is None]
    if missing:
        raise ValueError(f"Missing required landmarks for joints: {', '.join(missing)}")

    return positions


# =========================
# Rest pose and rotations
# =========================

def compute_offsets(all_positions, rest_start=0, rest_frames=30):
    n_frames = len(all_positions)
    a = max(0, rest_start)
    b = min(n_frames, rest_start + rest_frames)

    avg = {}
    for joint in PARENTS:
        avg[joint] = np.mean([all_positions[i][joint] for i in range(a, b)], axis=0)

    offsets = {}
    for joint, parent in PARENTS.items():
        if parent is None:
            offsets[joint] = np.array([0.0, 0.0, 0.0])
        else:
            offsets[joint] = avg[joint] - avg[parent]

    for joint in offsets:
        if joint != "Hips" and np.linalg.norm(offsets[joint]) < 1e-6:
            offsets[joint] = np.array([0.0, 0.01, 0.0])

    return offsets


def compute_tpose_offsets(measured_offsets):
    offsets = {}

    def length(joint, fallback=0.05):
        value = measured_offsets.get(joint)
        if value is None:
            return fallback
        out = float(np.linalg.norm(value))
        return out if out > 1e-6 else fallback

    for joint in PARENTS:
        offsets[joint] = np.array([0.0, 0.0, 0.0])

    for joint in ["Spine", "Spine1", "Spine2", "Chest", "Neck", "Head"]:
        if joint in offsets:
            offsets[joint] = np.array([0.0, length(joint), 0.0])

    if "LeftShoulder" in offsets:
        offsets["LeftShoulder"] = np.array([length("LeftShoulder"), 0.0, 0.0])
    if "RightShoulder" in offsets:
        offsets["RightShoulder"] = np.array([-length("RightShoulder"), 0.0, 0.0])

    for joint in ["LeftArm", "LeftForeArm", "LeftHand", "LeftHandEnd"]:
        if joint in offsets:
            offsets[joint] = np.array([length(joint), 0.0, 0.0])
    for joint in ["RightArm", "RightForeArm", "RightHand", "RightHandEnd"]:
        if joint in offsets:
            offsets[joint] = np.array([-length(joint), 0.0, 0.0])

    for joint in ["LeftUpLeg", "RightUpLeg"]:
        if joint in offsets:
            source = measured_offsets.get(joint, np.array([0.0, 0.0, 0.0]))
            x_sign = 1.0 if joint.startswith("Left") else -1.0
            offsets[joint] = np.array([x_sign * abs(source[0]), -abs(source[1]) * 0.2, 0.0])
            if np.linalg.norm(offsets[joint]) < 1e-6:
                offsets[joint] = np.array([x_sign * length(joint), 0.0, 0.0])

    for joint in ["LeftLeg", "RightLeg"]:
        if joint in offsets:
            offsets[joint] = np.array([0.0, -length(joint), 0.0])

    for joint in ["LeftFoot", "RightFoot"]:
        if joint in offsets:
            bone_len = length(joint)
            offsets[joint] = np.array([0.0, -bone_len * 0.25, -bone_len * 0.75])

    for joint in ["LeftToe", "RightToe", "LeftToeBase", "RightToeBase"]:
        if joint in offsets:
            offsets[joint] = np.array([0.0, 0.0, -length(joint)])

    return offsets


def compute_rest_positions(all_positions, rest_start=0, rest_frames=30):
    n_frames = len(all_positions)
    a = max(0, rest_start)
    b = min(n_frames, rest_start + rest_frames)
    keys = set()
    for positions in all_positions[a:b]:
        keys.update(positions.keys())
    rest = {}
    for key in keys:
        values = [
            all_positions[i][key]
            for i in range(a, b)
            if key in all_positions[i]
            and all_positions[i][key] is not None
            and np.isfinite(all_positions[i][key]).all()
        ]
        if values:
            rest[key] = np.mean(values, axis=0)
        else:
            rest[key] = None
    return rest


def torso_basis(positions):
    hip_right = normalize(positions["_right_hip"] - positions["_left_hip"])
    shoulder_right = normalize(positions["_right_shoulder"] - positions["_left_shoulder"])
    up = normalize(positions["Chest"] - positions["Hips"])
    right = normalize(0.5 * hip_right + 0.5 * shoulder_right)
    forward = normalize(np.cross(right, up))
    right = normalize(np.cross(up, forward))
    return orthonormalize(np.column_stack([right, up, forward]))


def head_basis(positions, fallback_basis=None, weight=0.7):
    if weight <= 0.0:
        return fallback_basis

    left_eye = positions.get("_left_eye")
    right_eye = positions.get("_right_eye")
    left_ear = positions.get("_left_ear")
    right_ear = positions.get("_right_ear")
    nose = positions.get("_nose")

    right_vec = None
    if left_eye is not None and right_eye is not None:
        right_vec = right_eye - left_eye
    elif left_ear is not None and right_ear is not None:
        right_vec = right_ear - left_ear

    center = blend_points([left_eye, right_eye, left_ear, right_ear])
    forward = None if center is None or nose is None else nose - center

    if right_vec is None or forward is None:
        return fallback_basis

    if np.linalg.norm(right_vec) < 1e-4 or np.linalg.norm(forward) < 1e-4:
        return fallback_basis

    up = normalize(np.cross(right_vec, forward))
    if np.linalg.norm(up) < 1e-4:
        return fallback_basis

    forward = normalize(np.cross(up, right_vec))
    face_basis = orthonormalize(np.column_stack([normalize(right_vec), up, normalize(forward)]))

    if fallback_basis is None:
        return face_basis
    return orthonormalize((1.0 - weight) * fallback_basis + weight * face_basis)


def hand_basis(positions, side):
    wrist = positions.get(f"_{side}_wrist")
    index_mcp = positions.get(f"_{side}_hand_index_mcp")
    if index_mcp is None:
        index_mcp = positions.get(f"_{side}_index")
    pinky_mcp = positions.get(f"_{side}_hand_pinky_mcp")
    if pinky_mcp is None:
        pinky_mcp = positions.get(f"_{side}_pinky")
    middle = positions.get(f"_{side}_hand_middle")
    if middle is None:
        middle = positions.get(f"{side.capitalize()}HandEnd")
    if middle is None:
        middle = positions.get(f"{side.capitalize()}Hand")
    if wrist is None or middle is None:
        return None
    palm_right = None if index_mcp is None or pinky_mcp is None else index_mcp - pinky_mcp
    forward = middle - wrist
    normal = np.cross(palm_right, forward) if palm_right is not None else np.array([0.0, 0.0, 1.0])
    return basis_from_yz(forward, normal)


def foot_basis(positions, side):
    foot = positions[f"{side.capitalize()}Foot"]
    toe = positions.get(f"{side.capitalize()}ToeBase")
    if toe is None:
        toe = positions[f"{side.capitalize()}Toe"]
    heel = positions.get(f"_{side}_heel", foot)
    forward = toe - foot
    sole = toe - heel
    return basis_from_yx(forward, sole)


def compute_world_rotations(positions, offsets, rest_positions, use_torso_frame=True, use_hand_rotation=False, head_rotation_weight=0.7):
    representative_child = {joint: (CHILDREN[joint][0] if CHILDREN[joint] else None) for joint in CHANNEL_JOINTS}

    world_rot = {}
    rest_torso = torso_basis(rest_positions)
    current_torso = torso_basis(positions)

    for joint in CHANNEL_JOINTS:
        if use_torso_frame and joint in {"Hips", "Spine", "Spine1", "Spine2", "Chest"}:
            world_rot[joint] = rotation_from_bases(rest_torso, current_torso)
            continue

        if joint == "Head":
            if head_rotation_weight <= 0.0:
                world_rot[joint] = world_rot.get("Neck", np.eye(3))
                continue

            rest_fallback = basis_from_yx(rest_positions["Head"] - rest_positions["Neck"], rest_positions["_right_shoulder"] - rest_positions["_left_shoulder"])
            current_fallback = basis_from_yx(positions["Head"] - positions["Neck"], positions["_right_shoulder"] - positions["_left_shoulder"])
            rest_head = head_basis(rest_positions, rest_fallback, weight=1.0)
            current_head = head_basis(positions, current_fallback, weight=head_rotation_weight)
            world_rot[joint] = rotation_from_bases(rest_head, current_head)
            continue

        if use_hand_rotation and joint in {"LeftForeArm", "RightForeArm"}:
            side = "left" if joint.startswith("Left") else "right"
            rest_hand = hand_basis(rest_positions, side)
            current_hand = hand_basis(positions, side)
            if rest_hand is not None and current_hand is not None:
                world_rot[joint] = rotation_from_bases(rest_hand, current_hand)
                continue

        if joint in {"LeftFoot", "RightFoot"}:
            side = "left" if joint.startswith("Left") else "right"
            rest_foot = foot_basis(rest_positions, side)
            current_foot = foot_basis(positions, side)
            world_rot[joint] = rotation_from_bases(rest_foot, current_foot)
            continue

        child = representative_child[joint]
        if child is None:
            world_rot[joint] = np.eye(3)
            continue

        rest_vec = offsets[child]
        current_vec = positions[child] - positions[joint]
        world_rot[joint] = rotation_matrix_from_vectors(rest_vec, current_vec)

    return world_rot


def compute_local_eulers(positions, offsets, rest_positions, use_torso_frame=True, use_hand_rotation=False, head_rotation_weight=0.7):
    world_rot = compute_world_rotations(
        positions,
        offsets,
        rest_positions,
        use_torso_frame=use_torso_frame,
        use_hand_rotation=use_hand_rotation,
        head_rotation_weight=head_rotation_weight,
    )
    local_eulers = {}

    for joint in CHANNEL_JOINTS:
        parent = PARENTS[joint]
        if parent is None:
            local_rot = world_rot[joint]
        else:
            local_rot = np.linalg.inv(world_rot[parent]) @ world_rot[joint]
        local_eulers[joint] = matrix_to_euler_zxy_degrees(local_rot)

    return local_eulers


# =========================
# BVH writer
# =========================

def format_vec(v):
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


def compute_end_site_offsets(offsets, scale=0.25):
    end_offsets = {}
    for joint in PARENTS:
        if CHILDREN[joint]:
            continue
        parent_offset = offsets[joint]
        if np.linalg.norm(parent_offset) < 1e-8:
            end_offsets[joint] = np.array([0.0, 0.0, 0.0])
        else:
            end_offsets[joint] = parent_offset * scale
    return end_offsets


def joint_angle_exceeds_threshold(euler_degrees, threshold):
    if threshold is None or threshold <= 0:
        return False
    return float(np.max(np.abs(euler_degrees))) > threshold


def soft_limit_euler(euler_degrees, threshold, soft_start_ratio=0.7):
    if threshold is None or threshold <= 0:
        return euler_degrees

    soft_start_ratio = max(0.0, min(0.99, soft_start_ratio))
    start = threshold * soft_start_ratio
    span = max(threshold - start, 1e-6)
    limited = np.asarray(euler_degrees, dtype=float).copy()

    for axis in range(3):
        value = limited[axis]
        magnitude = abs(value)
        if magnitude <= start:
            continue
        compressed = start + span * math.tanh((magnitude - start) / span)
        limited[axis] = math.copysign(min(compressed, threshold), value)

    return limited


def limit_euler_delta(current, previous, max_delta):
    if previous is None or max_delta is None or max_delta <= 0:
        return current

    limited = np.asarray(current, dtype=float).copy()
    for axis in range(3):
        delta = limited[axis] - previous[axis]
        if delta > max_delta:
            limited[axis] = previous[axis] + max_delta
        elif delta < -max_delta:
            limited[axis] = previous[axis] - max_delta
    return limited


def output_joint_name(joint, prefix=""):
    return f"{prefix}{joint}"


def write_joint_hierarchy(lines, joint, offsets, end_offsets, indent=0, output_prefix=""):
    sp = "  " * indent
    lines.append(f"{sp}{'ROOT' if joint == 'Hips' else 'JOINT'} {output_joint_name(joint, output_prefix)}")
    lines.append(f"{sp}{{")
    lines.append(f"{sp}  OFFSET {format_vec(offsets[joint])}")

    if joint == "Hips":
        lines.append(f"{sp}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{sp}  CHANNELS 3 Zrotation Xrotation Yrotation")

    for child in CHILDREN[joint]:
        write_joint_hierarchy(lines, child, offsets, end_offsets, indent + 1, output_prefix=output_prefix)

    if not CHILDREN[joint]:
        lines.append(f"{sp}  End Site")
        lines.append(f"{sp}  {{")
        lines.append(f"{sp}    OFFSET {format_vec(end_offsets[joint])}")
        lines.append(f"{sp}  }}")

    lines.append(f"{sp}}}")


def write_bvh(
    output_path,
    all_positions,
    offsets,
    rest_positions,
    fps=30.0,
    use_torso_frame=True,
    use_hand_rotation=False,
    head_rotation_weight=0.7,
    max_wrist_angle=100.0,
    max_ankle_angle=80.0,
    joint_limit_mode="soft",
    joint_limit_soft_start=0.7,
    max_limited_joint_delta=35.0,
    output_prefix="",
    prepend_tpose=False,
    tpose_frames=30,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end_offsets = compute_end_site_offsets(offsets)

    lines = ["HIERARCHY"]
    write_joint_hierarchy(lines, "Hips", offsets, end_offsets, output_prefix=output_prefix)

    frame_time = 1.0 / fps
    lines.append("MOTION")
    prepended_frames = tpose_frames if prepend_tpose else 0
    lines.append(f"Frames: {len(all_positions) + prepended_frames}")
    lines.append(f"Frame Time: {frame_time:.8f}")

    root_start = all_positions[0]["Hips"]
    previous_eulers = None

    if prepend_tpose:
        tpose_values = [0.0, 0.0, 0.0]
        for _joint in CHANNEL_JOINTS:
            tpose_values.extend([0.0, 0.0, 0.0])
        tpose_line = " ".join(f"{v:.6f}" for v in tpose_values)
        for _ in range(tpose_frames):
            lines.append(tpose_line)
        previous_eulers = {joint: np.zeros(3, dtype=float) for joint in CHANNEL_JOINTS}

    for positions in all_positions:
        values = []
        root_pos = positions["Hips"] - root_start
        values.extend(root_pos.tolist())

        local_eulers = compute_local_eulers(
            positions,
            offsets,
            rest_positions,
            use_torso_frame=use_torso_frame,
            use_hand_rotation=use_hand_rotation,
            head_rotation_weight=head_rotation_weight,
        )

        if previous_eulers is not None:
            for joint in CHANNEL_JOINTS:
                current = local_eulers[joint].copy()
                previous = previous_eulers[joint]
                for axis in range(3):
                    while current[axis] - previous[axis] > 180.0:
                        current[axis] -= 360.0
                    while current[axis] - previous[axis] < -180.0:
                        current[axis] += 360.0
                local_eulers[joint] = current

        limited_joints = {
            "LeftHand": max_wrist_angle,
            "RightHand": max_wrist_angle,
            "LeftFoot": max_ankle_angle,
            "RightFoot": max_ankle_angle,
        }
        for joint, threshold in limited_joints.items():
            if threshold is None or threshold <= 0 or joint_limit_mode == "off":
                continue

            if joint_limit_mode == "soft":
                local_eulers[joint] = soft_limit_euler(
                    local_eulers[joint],
                    threshold,
                    soft_start_ratio=joint_limit_soft_start,
                )
                if previous_eulers is not None:
                    previous = previous_eulers[joint]
                    for axis in range(3):
                        while local_eulers[joint][axis] - previous[axis] > 180.0:
                            local_eulers[joint][axis] -= 360.0
                        while local_eulers[joint][axis] - previous[axis] < -180.0:
                            local_eulers[joint][axis] += 360.0
                    local_eulers[joint] = limit_euler_delta(
                        local_eulers[joint],
                        previous,
                        max_limited_joint_delta,
                    )
            elif joint_angle_exceeds_threshold(local_eulers[joint], threshold):
                if previous_eulers is not None:
                    local_eulers[joint] = previous_eulers[joint].copy()
                else:
                    local_eulers[joint] = np.zeros(3, dtype=float)

        for joint in CHANNEL_JOINTS:
            values.extend(local_eulers[joint].tolist())

        lines.append(" ".join(f"{v:.6f}" for v in values))
        previous_eulers = {joint: local_eulers[joint].copy() for joint in CHANNEL_JOINTS}

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# Debug output
# =========================

def write_joint_map(path):
    mapping = {
        "Hips": ["left_hip", "right_hip", "center_of_mass"],
        "Spine": ["Hips", "Chest"],
        "Chest": ["left_shoulder", "right_shoulder"],
        "Neck": ["neck_center", "chest", "nose"],
        "Head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "LeftHandEnd": ["left_hand middle/index/ring/pinky/thumb tips", "left_index"],
        "RightHandEnd": ["right_hand middle/index/ring/pinky/thumb tips", "right_index"],
        "LeftToe": ["left_foot_index"],
        "RightToe": ["right_foot_index"],
    }
    Path(path).write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def write_preview_obj(path, positions):
    edges = [(parent, child) for child, parent in PARENTS.items() if parent is not None]
    vertices = []
    index = {}
    for joint in PARENTS:
        index[joint] = len(vertices) + 1
        vertices.append(positions[joint])

    lines = []
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for parent, child in edges:
        lines.append(f"l {index[parent]} {index[child]}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def debug_summary(frames, files):
    keys = sorted(frame_keys(frames))
    important = [
        "left_shoulder", "right_shoulder", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_foot_index", "right_foot_index",
        "center_of_mass", "left_hand_middle_finger_tip", "right_hand_middle_finger_tip",
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    ]

    print("Debug summary")
    print(f"  frames: {len(frames)}")
    print(f"  landmark names: {len(keys)}")
    print("  files:")
    for label, path in files.items():
        print(f"    {label}: {path or '(none)'}")
    print("  recognized important landmarks:")
    for name in important:
        print(f"    {name}: {'yes' if name in keys else 'no'}")
    print("  first 40 names:")
    print("    " + ", ".join(keys[:40]))


# =========================
# Main
# =========================

def existing_default(*paths):
    for path in paths:
        if Path(path).exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert FreeMoCap / MediaPipe landmarks to a BVH file.")

    parser.add_argument("--body", default=existing_default("input/all_frame_name_xyz.csv", "input/all_frame_name_xyz.npy", "input/body_frame_name_xyz.npy", "input/mediapipe_body_3d_xyz.csv", "input/body_trajectories.csv"))
    parser.add_argument("--left-hand", default=existing_default("input/left_hand_frame_name_xyz.csv", "input/left_hand_frame_name_xyz.npy"))
    parser.add_argument("--right-hand", default=existing_default("input/right_hand_frame_name_xyz.csv", "input/right_hand_frame_name_xyz.npy"))
    parser.add_argument("--face", default=existing_default("input/face_frame_name_xyz.csv"))
    parser.add_argument("--com", default=existing_default("input/center_of_mass_frame_name_xyz.csv", "input/center_of_mass_frame_name_xyz.npy"))
    parser.add_argument("--output", default="output/output_plus.bvh")
    parser.add_argument("--rig-preset", choices=["simple", "mixamo"], default="mixamo")
    parser.add_argument("--mixamo-prefix", default="mixamorig:")
    parser.add_argument("--rest-pose", choices=["data", "tpose"], default="data")
    parser.add_argument("--prepend-tpose", action="store_true")
    parser.add_argument("--tpose-frames", type=int, default=30)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--root-smooth", type=int, default=9)
    parser.add_argument("--hand-smooth", type=int, default=5)
    parser.add_argument("--face-smooth", type=int, default=5)
    parser.add_argument("--com-blend", type=float, default=0.75)
    parser.add_argument("--max-joint-speed", type=float, default=None)

    parser.add_argument("--no-swap-yz", action="store_true")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--rotate-x", type=float, default=0.0)
    parser.add_argument("--rotate-y", type=float, default=0.0)
    parser.add_argument("--rotate-z", type=float, default=0.0)

    parser.add_argument("--use-torso-frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-hand-rotation", action="store_true")
    parser.add_argument("--head-rotation-weight", type=float, default=0.0)
    parser.add_argument("--max-wrist-angle", type=float, default=100.0, help="Limit wrist local Euler angles. Use 0 to disable.")
    parser.add_argument("--max-ankle-angle", type=float, default=80.0, help="Limit ankle local Euler angles. Use 0 to disable.")
    parser.add_argument("--joint-limit-mode", choices=["soft", "hold", "off"], default="soft", help="soft compresses excessive local rotation toward the parent; hold reuses the previous valid value.")
    parser.add_argument("--joint-limit-soft-start", type=float, default=0.7, help="Ratio of the max angle where soft compression starts.")
    parser.add_argument("--max-limited-joint-delta", type=float, default=35.0, help="Maximum per-frame Euler change for limited wrist/ankle joints. Use 0 to disable.")
    parser.add_argument("--add-fingers", action="store_true", help="Reserved for future detailed finger BVH output.")

    parser.add_argument("--debug-summary", action="store_true")
    parser.add_argument("--dump-joint-map")
    parser.add_argument("--preview-first-frame")

    args = parser.parse_args()
    set_rig_preset(args.rig_preset)

    frames = []
    files = {
        "body": args.body,
        "left_hand": args.left_hand,
        "right_hand": args.right_hand,
        "face": args.face,
        "com": args.com,
    }

    merge_frames(frames, read_landmark_file(args.body, source="body"))
    merge_frames(frames, read_landmark_file(args.left_hand, source="hand", side="left"))
    merge_frames(frames, read_landmark_file(args.right_hand, source="hand", side="right"))
    merge_frames(frames, read_landmark_file(args.face, source="face"))
    merge_frames(frames, read_landmark_file(args.com, source="com"))

    if not frames:
        raise FileNotFoundError("No input landmark data was loaded. Use --body or another input path.")

    frames = convert_coordinate_system(
        frames,
        scale=args.scale,
        swap_yz=not args.no_swap_yz,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )
    frames = rotate_frames(frames, args.rotate_x, args.rotate_y, args.rotate_z)

    all_keys = frame_keys(frames)
    hand_keys = {k for k in all_keys if k.startswith("left_hand_") or k.startswith("right_hand_")}
    face_keys = {k for k in all_keys if k.startswith("face_") or k in FACE_NAMES}
    root_keys = {"center_of_mass", "com", "total_body_com", "left_hip", "right_hip"}
    body_keys = all_keys - hand_keys - face_keys

    frames = fill_smooth_frames(frames, body_keys, window=args.smooth, max_joint_speed=args.max_joint_speed)
    frames = fill_smooth_frames(frames, root_keys & frame_keys(frames), window=args.root_smooth, max_joint_speed=args.max_joint_speed)
    frames = fill_smooth_frames(frames, hand_keys, window=args.hand_smooth, max_joint_speed=args.max_joint_speed)
    frames = fill_smooth_frames(frames, face_keys, window=args.face_smooth, max_joint_speed=args.max_joint_speed)

    if args.debug_summary:
        debug_summary(frames, files)

    all_positions = []
    for frame in frames:
        all_positions.append(build_intermediate_positions(frame, com_blend=args.com_blend))

    measured_offsets = compute_offsets(all_positions)
    offsets = compute_tpose_offsets(measured_offsets) if args.rest_pose == "tpose" else measured_offsets
    rest_positions = compute_rest_positions(all_positions)
    output_prefix = args.mixamo_prefix if args.rig_preset == "mixamo" else ""

    write_bvh(
        args.output,
        all_positions,
        offsets,
        rest_positions,
        fps=args.fps,
        use_torso_frame=args.use_torso_frame,
        use_hand_rotation=args.use_hand_rotation,
        head_rotation_weight=args.head_rotation_weight,
        max_wrist_angle=args.max_wrist_angle,
        max_ankle_angle=args.max_ankle_angle,
        joint_limit_mode=args.joint_limit_mode,
        joint_limit_soft_start=args.joint_limit_soft_start,
        max_limited_joint_delta=args.max_limited_joint_delta,
        output_prefix=output_prefix,
        prepend_tpose=args.prepend_tpose,
        tpose_frames=args.tpose_frames,
    )

    if args.dump_joint_map:
        write_joint_map(args.dump_joint_map)
    if args.preview_first_frame:
        write_preview_obj(args.preview_first_frame, all_positions[0])

    print(f"Saved BVH: {args.output}")
    total_frames = len(all_positions) + (args.tpose_frames if args.prepend_tpose else 0)
    print(f"Frames: {total_frames}")
    print(f"FPS: {args.fps}")


if __name__ == "__main__":
    main()
