import argparse
import math
from pathlib import Path

import numpy as np


# =========================
# MediaPipe / FreeMoCap body landmark indices
# =========================

MP = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# =========================
# BVH skeleton definition
# =========================

PARENTS = {
    "Hips": None,

    "Spine": "Hips",
    "Chest": "Spine",
    "Neck": "Chest",
    "Head": "Neck",

    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",

    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",

    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToe": "LeftFoot",

    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToe": "RightFoot",
}

CHILDREN = {}
for joint, parent in PARENTS.items():
    CHILDREN.setdefault(joint, [])
    if parent is not None:
        CHILDREN.setdefault(parent, []).append(joint)


CHANNEL_JOINTS = list(PARENTS.keys())


# =========================
# Basic math
# =========================

def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix that rotates vector a to vector b.
    """
    a = normalize(a)
    b = normalize(b)

    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return np.eye(3)

    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if c > 0.999999:
        return np.eye(3)

    if c < -0.999999:
        # 180-degree rotation around an arbitrary orthogonal axis
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = normalize(axis)
        return axis_angle_to_matrix(axis, math.pi)

    s = np.linalg.norm(v)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

    r = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))
    return r


def orientation_matrix_from_primary_side(primary, side_hint):
    """
    Build an orthonormal world basis from a bone direction and a roll hint.
    The primary axis becomes local Y, and the side hint stabilizes local X.
    """
    y_axis = normalize(primary)
    if np.linalg.norm(y_axis) < 1e-8:
        return None

    x_axis = np.asarray(side_hint, dtype=float)
    x_axis = x_axis - y_axis * np.dot(x_axis, y_axis)

    if np.linalg.norm(x_axis) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, y_axis)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        x_axis = fallback - y_axis * np.dot(fallback, y_axis)

    x_axis = normalize(x_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_matrix_from_orientations(rest_primary, rest_side, current_primary, current_side):
    rest_basis = orientation_matrix_from_primary_side(rest_primary, rest_side)
    current_basis = orientation_matrix_from_primary_side(current_primary, current_side)

    if rest_basis is None or current_basis is None:
        return rotation_matrix_from_vectors(rest_primary, current_primary)

    return current_basis @ rest_basis.T


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


def euler_xyz_to_matrix_degrees(x_degrees=0.0, y_degrees=0.0, z_degrees=0.0):
    x = math.radians(x_degrees)
    y = math.radians(y_degrees)
    z = math.radians(z_degrees)

    rx = axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), x)
    ry = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), y)
    rz = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), z)

    return rz @ ry @ rx


def rotate_points(data, x_degrees=0.0, y_degrees=0.0, z_degrees=0.0):
    if abs(x_degrees) < 1e-8 and abs(y_degrees) < 1e-8 and abs(z_degrees) < 1e-8:
        return data

    r = euler_xyz_to_matrix_degrees(x_degrees, y_degrees, z_degrees)
    return data @ r.T


def matrix_to_euler_zxy_degrees(r):
    """
    Convert rotation matrix to BVH ZXY Euler angles in degrees.

    BVH line order below is:
    Zrotation Xrotation Yrotation

    This function returns:
    z, x, y
    """
    # R = Rz(z) * Rx(x) * Ry(y)
    # Matrix elements:
    # r[2,1] = sin(x)
    sx = max(-1.0, min(1.0, r[2, 1]))
    x = math.asin(sx)

    cx = math.cos(x)

    if abs(cx) > 1e-8:
        z = math.atan2(-r[0, 1], r[1, 1])
        y = math.atan2(-r[2, 0], r[2, 2])
    else:
        # Gimbal lock fallback
        z = math.atan2(r[1, 0], r[0, 0])
        y = 0.0

    return np.degrees([z, x, y])


def moving_average(data, window):
    if window <= 1:
        return data

    if window % 2 == 0:
        window += 1

    pad = window // 2
    padded = np.pad(data, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    out = np.zeros_like(data)

    for i in range(data.shape[0]):
        out[i] = padded[i:i + window].mean(axis=0)

    return out


# =========================
# FreeMoCap to intermediate joints
# =========================

def get_point(frame, name):
    return frame[MP[name]]


def build_intermediate_positions(frame):
    """
    Convert MediaPipe/FreeMoCap 33 landmarks into a simpler BVH skeleton.
    """

    left_hip = get_point(frame, "left_hip")
    right_hip = get_point(frame, "right_hip")
    left_shoulder = get_point(frame, "left_shoulder")
    right_shoulder = get_point(frame, "right_shoulder")

    hips = (left_hip + right_hip) / 2.0
    chest = (left_shoulder + right_shoulder) / 2.0
    spine = hips * 0.55 + chest * 0.45

    nose = get_point(frame, "nose")
    neck = chest * 0.75 + nose * 0.25
    head = nose

    positions = {
        "Hips": hips,

        "Spine": spine,
        "Chest": chest,
        "Neck": neck,
        "Head": head,

        "LeftShoulder": left_shoulder,
        "LeftArm": get_point(frame, "left_elbow"),
        "LeftForeArm": get_point(frame, "left_wrist"),
        "LeftHand": get_point(frame, "left_index"),

        "RightShoulder": right_shoulder,
        "RightArm": get_point(frame, "right_elbow"),
        "RightForeArm": get_point(frame, "right_wrist"),
        "RightHand": get_point(frame, "right_index"),

        "LeftUpLeg": left_hip,
        "LeftLeg": get_point(frame, "left_knee"),
        "LeftFoot": get_point(frame, "left_ankle"),
        "LeftToe": get_point(frame, "left_foot_index"),

        "RightUpLeg": right_hip,
        "RightLeg": get_point(frame, "right_knee"),
        "RightFoot": get_point(frame, "right_ankle"),
        "RightToe": get_point(frame, "right_foot_index"),

        "_left_hip": left_hip,
        "_right_hip": right_hip,
        "_left_shoulder": left_shoulder,
        "_right_shoulder": right_shoulder,
        "_left_wrist": get_point(frame, "left_wrist"),
        "_right_wrist": get_point(frame, "right_wrist"),
        "_left_index": get_point(frame, "left_index"),
        "_right_index": get_point(frame, "right_index"),
        "_left_pinky": get_point(frame, "left_pinky"),
        "_right_pinky": get_point(frame, "right_pinky"),
        "_left_thumb": get_point(frame, "left_thumb"),
        "_right_thumb": get_point(frame, "right_thumb"),
    }

    return positions


def convert_coordinate_system(data, scale=1.0, swap_yz=True, flip_x=False, flip_y=False, flip_z=False):
    """
    Convert FreeMoCap coordinate data to Blender-ish coordinate data.

    FreeMoCap coordinate convention can vary depending on export.
    This default often works as a first try:
      input  x, y, z
      output x, z, y

    If the BVH lies sideways or mirrored, try options:
      --no-swap-yz
      --flip-x
      --flip-y
      --flip-z
    """

    out = np.array(data, dtype=float) * scale

    if swap_yz:
        out = out[..., [0, 2, 1]]

    if flip_x:
        out[..., 0] *= -1.0
    if flip_y:
        out[..., 1] *= -1.0
    if flip_z:
        out[..., 2] *= -1.0

    return out


# =========================
# Rest pose and rotations
# =========================

def compute_offsets(all_positions, rest_start=0, rest_frames=30):
    """
    Compute BVH OFFSETs from average pose.
    """
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

    # Avoid zero-length bones without creating oversized limbs.
    for joint in offsets:
        if joint != "Hips" and np.linalg.norm(offsets[joint]) < 1e-6:
            offsets[joint] = np.array([0.0, 0.01, 0.0])

    return offsets


def compute_rest_positions(all_positions, rest_start=0, rest_frames=30):
    n_frames = len(all_positions)
    a = max(0, rest_start)
    b = min(n_frames, rest_start + rest_frames)

    keys = set()
    for positions in all_positions[a:b]:
        keys.update(positions.keys())

    return {
        key: np.mean([all_positions[i][key] for i in range(a, b) if key in all_positions[i]], axis=0)
        for key in keys
    }


def compute_end_site_offsets(offsets, scale=0.25):
    """
    Keep BVH leaf bones visible without adding fixed-size one-meter end bones.
    """
    end_offsets = {}

    for joint in PARENTS:
        if CHILDREN[joint]:
            continue

        parent = PARENTS[joint]
        parent_offset = offsets[joint]

        if parent is None or np.linalg.norm(parent_offset) < 1e-8:
            end_offsets[joint] = np.array([0.0, 0.0, 0.0])
        else:
            end_offsets[joint] = parent_offset * scale

    return end_offsets


def compute_world_rotations(positions, offsets, rest_positions):
    """
    Compute approximate world rotations for each joint.
    The rotation aligns each joint's rest child direction to the current child direction.

    For joints with multiple children, use a representative main child.
    """

    representative_child = {
        "Hips": "Spine",
        "Spine": "Chest",
        "Chest": "Neck",
        "Neck": "Head",
        "Head": None,

        "LeftShoulder": "LeftArm",
        "LeftArm": "LeftForeArm",
        "LeftForeArm": "LeftHand",
        "LeftHand": None,

        "RightShoulder": "RightArm",
        "RightArm": "RightForeArm",
        "RightForeArm": "RightHand",
        "RightHand": None,

        "LeftUpLeg": "LeftLeg",
        "LeftLeg": "LeftFoot",
        "LeftFoot": "LeftToe",
        "LeftToe": None,

        "RightUpLeg": "RightLeg",
        "RightLeg": "RightFoot",
        "RightFoot": "RightToe",
        "RightToe": None,
    }

    world_rot = {}

    for joint in CHANNEL_JOINTS:
        if joint == "Hips":
            world_rot[joint] = rotation_matrix_from_orientations(
                rest_positions["Spine"] - rest_positions["Hips"],
                rest_positions["_left_hip"] - rest_positions["_right_hip"],
                positions["Spine"] - positions["Hips"],
                positions["_left_hip"] - positions["_right_hip"],
            )
            continue

        if joint == "Chest":
            world_rot[joint] = rotation_matrix_from_orientations(
                rest_positions["Neck"] - rest_positions["Chest"],
                rest_positions["_left_shoulder"] - rest_positions["_right_shoulder"],
                positions["Neck"] - positions["Chest"],
                positions["_left_shoulder"] - positions["_right_shoulder"],
            )
            continue

        if joint == "LeftForeArm":
            world_rot[joint] = rotation_matrix_from_orientations(
                rest_positions["_left_index"] - rest_positions["_left_wrist"],
                rest_positions["_left_thumb"] - rest_positions["_left_pinky"],
                positions["_left_index"] - positions["_left_wrist"],
                positions["_left_thumb"] - positions["_left_pinky"],
            )
            continue

        if joint == "RightForeArm":
            world_rot[joint] = rotation_matrix_from_orientations(
                rest_positions["_right_index"] - rest_positions["_right_wrist"],
                rest_positions["_right_thumb"] - rest_positions["_right_pinky"],
                positions["_right_index"] - positions["_right_wrist"],
                positions["_right_thumb"] - positions["_right_pinky"],
            )
            continue

        child = representative_child.get(joint)

        if child is None:
            world_rot[joint] = np.eye(3)
            continue

        rest_vec = offsets[child]
        current_vec = positions[child] - positions[joint]

        world_rot[joint] = rotation_matrix_from_vectors(rest_vec, current_vec)

    return world_rot


def compute_local_eulers(positions, offsets, rest_positions):
    world_rot = compute_world_rotations(positions, offsets, rest_positions)
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


def write_joint_hierarchy(lines, joint, offsets, end_offsets, indent=0):
    sp = "  " * indent

    if joint == "Hips":
        lines.append(f"{sp}ROOT {joint}")
    else:
        lines.append(f"{sp}JOINT {joint}")

    lines.append(f"{sp}{{")
    lines.append(f"{sp}  OFFSET {format_vec(offsets[joint])}")

    if joint == "Hips":
        lines.append(f"{sp}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{sp}  CHANNELS 3 Zrotation Xrotation Yrotation")

    for child in CHILDREN[joint]:
        write_joint_hierarchy(lines, child, offsets, end_offsets, indent + 1)

    if len(CHILDREN[joint]) == 0:
        lines.append(f"{sp}  End Site")
        lines.append(f"{sp}  {{")
        lines.append(f"{sp}    OFFSET {format_vec(end_offsets[joint])}")
        lines.append(f"{sp}  }}")

    lines.append(f"{sp}}}")


def write_bvh(output_path, all_positions, offsets, rest_positions, fps=30.0):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end_offsets = compute_end_site_offsets(offsets)

    lines = []
    lines.append("HIERARCHY")
    write_joint_hierarchy(lines, "Hips", offsets, end_offsets, indent=0)

    frame_time = 1.0 / fps
    lines.append("MOTION")
    lines.append(f"Frames: {len(all_positions)}")
    lines.append(f"Frame Time: {frame_time:.8f}")

    root_start = all_positions[0]["Hips"]

    for positions in all_positions:
        values = []

        root_pos = positions["Hips"] - root_start
        values.extend(root_pos.tolist())

        local_eulers = compute_local_eulers(positions, offsets, rest_positions)

        for joint in CHANNEL_JOINTS:
            z, x, y = local_eulers[joint]
            values.extend([z, x, y])

        lines.append(" ".join(f"{v:.6f}" for v in values))

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================
# Loader
# =========================

def load_npy(path):
    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(f"Expected npy shape: frames x joints x 3, but got {data.shape}")

    if data.shape[2] != 3:
        raise ValueError(f"Expected last dimension to be xyz=3, but got {data.shape}")

    if data.shape[1] < 33:
        raise ValueError(
            f"Expected at least 33 body landmarks. Got {data.shape[1]} landmarks. "
            "This script assumes MediaPipe body landmark indices."
        )

    return data.astype(float)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Convert FreeMoCap / MediaPipe skeleton_3d.npy to a simple BVH file."
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="input/body_frame_name_xyz.npy",
        help="Path to skeleton_3d.npy. Default: input/body_frame_name_xyz.npy",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="output/output.bvh",
        help="Path to output .bvh. Default: output/output.bvh",
    )
    parser.add_argument("--input-file", dest="input_file", help="Path to skeleton_3d.npy.")
    parser.add_argument("--output-file", dest="output_file", help="Path to output .bvh.")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.0, help="Scale coordinates. Use 0.001 if input is mm.")
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window. Use 1 to disable.")

    parser.add_argument("--rest-start", type=int, default=0)
    parser.add_argument("--rest-frames", type=int, default=30)

    parser.add_argument("--no-swap-yz", action="store_true")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")
    parser.add_argument("--rotate-x", type=float, default=0.0, help="Rotate all input points around X axis in degrees.")
    parser.add_argument("--rotate-y", type=float, default=0.0, help="Rotate all input points around Y axis in degrees.")
    parser.add_argument("--rotate-z", type=float, default=0.0, help="Rotate all input points around Z axis in degrees.")

    args = parser.parse_args()
    input_path = args.input_file or args.input
    output_path = args.output_file or args.output

    data = load_npy(input_path)

    data = convert_coordinate_system(
        data,
        scale=args.scale,
        swap_yz=not args.no_swap_yz,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )

    data = rotate_points(
        data,
        x_degrees=args.rotate_x,
        y_degrees=args.rotate_y,
        z_degrees=args.rotate_z,
    )

    data = moving_average(data, args.smooth)

    all_positions = []
    for frame_idx in range(data.shape[0]):
        positions = build_intermediate_positions(data[frame_idx])
        all_positions.append(positions)

    offsets = compute_offsets(
        all_positions,
        rest_start=args.rest_start,
        rest_frames=args.rest_frames,
    )
    rest_positions = compute_rest_positions(
        all_positions,
        rest_start=args.rest_start,
        rest_frames=args.rest_frames,
    )

    write_bvh(
        output_path=output_path,
        all_positions=all_positions,
        offsets=offsets,
        rest_positions=rest_positions,
        fps=args.fps,
    )

    print(f"Saved BVH: {output_path}")
    print(f"Frames: {data.shape[0]}")
    print(f"FPS: {args.fps}")
    print("If the motion is sideways or mirrored, try --no-swap-yz, --flip-x, --flip-y, or --flip-z.")


if __name__ == "__main__":
    main()
