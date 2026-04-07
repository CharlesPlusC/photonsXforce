"""
Generate lamella (venetian blind) cube OBJ files.

Each face of the cube has N slats that can tilt around an axis parallel to the face.
This creates a 6-dimensional control space (one tilt angle per face).

The slats rotate like venetian blinds:
  - tilt = 0°: slats are flat (parallel to original face)
  - tilt = 45°: slats tilted 45°
  - tilt = 90°: slats perpendicular to original face (edge-on)

Usage:
    python generate_lamella.py                    # Default: all tilts = 0
    python generate_lamella.py 0 30 0 30 0 30     # Specify 6 tilt angles in degrees
"""

import numpy as np
import sys
from pathlib import Path
from typing import List


def generate_lamella_obj(
    tilt_angles_deg: list[float],
    n_slats: int = 4,
    output_path: str = "lamella.obj",
    cube_half_size: float = 0.5,
    gap_fraction: float = 0.05,  # Gap between slats as fraction of slat width
    slat_thickness: float = None,  # Defaults to a very thin slab if None
) -> str:
    """
    Generate OBJ file for a cube with lamellae (venetian blinds) on each face.

    Args:
        tilt_angles_deg: 6 tilt angles in degrees, one per face.
                        Order: +X, -X, +Y, -Y, +Z, -Z
        n_slats: Number of slats per face (default: 4)
        output_path: Output OBJ file path
        cube_half_size: Half-size of the cube (default: 0.5 for 1m cube)
        gap_fraction: Gap between slats as fraction of slat pitch
        slat_thickness: Physical thickness of each slat. If None, uses a small
            fraction of the slat pitch to avoid coplanar front/back faces.

    Returns:
        Path to generated OBJ file
    """
    assert len(tilt_angles_deg) == 6, "Need exactly 6 tilt angles"

    vertices = []
    normals = []
    faces = []  # List of (v1, v2, v3, normal_idx, group_name)

    h = cube_half_size

    # Face definitions: (center, normal, up_dir, right_dir, group_name)
    # - normal: outward normal of the face
    # - up_dir: direction along which slats are stacked
    # - right_dir: direction along which each slat extends (rotation axis)
    face_defs = [
        (np.array([h, 0, 0]),  np.array([1, 0, 0]),  np.array([0, 0, 1]), np.array([0, 1, 0]), "Face_PosX"),
        (np.array([-h, 0, 0]), np.array([-1, 0, 0]), np.array([0, 0, 1]), np.array([0, -1, 0]), "Face_NegX"),
        (np.array([0, h, 0]),  np.array([0, 1, 0]),  np.array([0, 0, 1]), np.array([-1, 0, 0]), "Face_PosY"),
        (np.array([0, -h, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0]), "Face_NegY"),
        (np.array([0, 0, h]),  np.array([0, 0, 1]),  np.array([0, 1, 0]), np.array([1, 0, 0]), "Face_PosZ"),
        (np.array([0, 0, -h]), np.array([0, 0, -1]), np.array([0, -1, 0]), np.array([1, 0, 0]), "Face_NegZ"),
    ]

    for face_idx, (center, normal, up_dir, right_dir, group_name) in enumerate(face_defs):
        tilt_rad = np.radians(tilt_angles_deg[face_idx])

        # Slat dimensions in local coordinates
        slat_pitch = (2 * h) / n_slats  # Distance between slat centers
        slat_width = slat_pitch * (1 - gap_fraction)  # Actual slat width
        slat_length = 2 * h  # Slats span full width of face
        thickness = slat_thickness if slat_thickness is not None else (0.001 * slat_pitch)

        # Rotation matrix: rotate around 'right_dir' axis by tilt angle
        # Using Rodrigues' formula
        cos_t = np.cos(tilt_rad)
        sin_t = np.sin(tilt_rad)
        K = np.array([
            [0, -right_dir[2], right_dir[1]],
            [right_dir[2], 0, -right_dir[0]],
            [-right_dir[1], right_dir[0], 0]
        ])
        R = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)

        # Compute rotated normal for this face's slats (front and back)
        rotated_normal = R @ normal
        rotated_normal = rotated_normal / np.linalg.norm(rotated_normal)

        # Add normals for front and back faces
        front_normal_idx = len(normals) + 1  # OBJ is 1-indexed
        normals.append(rotated_normal)
        back_normal_idx = len(normals) + 1
        normals.append(-rotated_normal)  # Opposite direction for back face

        for slat_idx in range(n_slats):
            # Center of this slat along 'up' direction (local coords)
            up_offset = -h + slat_pitch * (slat_idx + 0.5)
            slat_center = center + up_dir * up_offset

            # Slat corners before rotation (in plane perpendicular to original normal)
            half_len = slat_length / 2
            half_wid = slat_width / 2

            # Each slat corner offset from slat center (before rotation):
            corner_offsets_local = [
                right_dir * (-half_len) + up_dir * (-half_wid),
                right_dir * (half_len) + up_dir * (-half_wid),
                right_dir * (half_len) + up_dir * (half_wid),
                right_dir * (-half_len) + up_dir * (half_wid),
            ]

            # Apply rotation around right_dir
            world_corners = []
            for offset in corner_offsets_local:
                rotated_offset = R @ offset
                world_pos = slat_center + rotated_offset
                world_corners.append(world_pos)

            # Offset front/back faces along the rotated normal to add thickness
            half_thickness = 0.5 * thickness
            front_offset = rotated_normal * half_thickness
            back_offset = -front_offset
            front_corners = [corner + front_offset for corner in world_corners]
            back_corners = [corner + back_offset for corner in world_corners]

            # Add vertices (front then back)
            v_start_front = len(vertices) + 1  # OBJ is 1-indexed
            vertices.extend(front_corners)
            v_start_back = len(vertices) + 1
            vertices.extend(back_corners)

            # All slats on a face share the same group (6 DOF total - one per face)
            # Add two triangles for the quad (front face - CCW winding)
            faces.append((v_start_front, v_start_front+1, v_start_front+2, front_normal_idx, group_name))
            faces.append((v_start_front, v_start_front+2, v_start_front+3, front_normal_idx, group_name))

            # Add two triangles for the quad (back face - CW winding = reversed order)
            faces.append((v_start_back, v_start_back+2, v_start_back+1, back_normal_idx, group_name + "_back"))
            faces.append((v_start_back, v_start_back+3, v_start_back+2, back_normal_idx, group_name + "_back"))

    # Write MTL file
    mtl_path = output_path.replace('.obj', '.mtl')
    face_names = ['Face_PosX', 'Face_NegX', 'Face_PosY', 'Face_NegY', 'Face_PosZ', 'Face_NegZ']

    with open(mtl_path, 'w') as f:
        f.write("# Lamella cube materials\n")
        f.write("# Each face has its own material for rotation control\n\n")

        for face_name in face_names:
            # Front face material
            f.write(f"newmtl {face_name}\n")
            f.write("Kd 0.5 0.5 0.5\n")  # Diffuse gray
            f.write("Ks 0.3 0.3 0.3\n")  # Specular
            f.write("Ns 50\n\n")         # Glossiness

            # Back face material (same properties)
            f.write(f"newmtl {face_name}_back\n")
            f.write("Kd 0.5 0.5 0.5\n")
            f.write("Ks 0.3 0.3 0.3\n")
            f.write("Ns 50\n\n")

    # Write OBJ file
    mtl_filename = Path(mtl_path).name
    with open(output_path, 'w') as f:
        f.write("# Lamella cube model\n")
        f.write(f"# {n_slats} slats per face, 6 faces\n")
        f.write(f"# Tilt angles (deg): {tilt_angles_deg}\n")
        f.write(f"# Total: {len(vertices)} vertices, {len(faces)} triangles\n")
        f.write(f"mtllib {mtl_filename}\n")
        f.write("\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")

        # Write normals
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\n")

        # Write faces grouped by face, with material assignment
        current_group = None
        for v1, v2, v3, ni, group in faces:
            if group != current_group:
                f.write(f"\ng {group}\n")
                f.write(f"usemtl {group}\n")
                current_group = group
            f.write(f"f {v1}//{ni} {v2}//{ni} {v3}//{ni}\n")

    print(f"Generated {output_path}")
    print(f"  - {n_slats} slats per face")
    print(f"  - {len(vertices)} vertices")
    print(f"  - {len(faces)} triangles")
    print(f"  - Tilt angles: {tilt_angles_deg}")

    return output_path


def generate_training_set(
    output_dir: str,
    n_samples: int = 100,
    n_slats: int = 4,
    tilt_range: tuple[float, float] = (-60, 60),
    seed: int = 42
) -> list[tuple[str, list[float]]]:
    """
    Generate a set of lamella OBJ files with random tilt angles for training.

    Args:
        output_dir: Directory to save OBJ files
        n_samples: Number of samples to generate
        n_slats: Number of slats per face
        tilt_range: (min, max) tilt angle in degrees
        seed: Random seed

    Returns:
        List of (filepath, tilt_angles) tuples
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    for i in range(n_samples):
        tilt_angles = np.random.uniform(tilt_range[0], tilt_range[1], size=6).tolist()
        filename = f"lamella_{i:04d}.obj"
        filepath = output_dir / filename

        generate_lamella_obj(tilt_angles, n_slats=n_slats, output_path=str(filepath))
        samples.append((str(filepath), tilt_angles))

    # Save index file
    index_path = output_dir / "index.txt"
    with open(index_path, 'w') as f:
        f.write("# Lamella training set index\n")
        f.write("# Format: filename, tilt_+X, tilt_-X, tilt_+Y, tilt_-Y, tilt_+Z, tilt_-Z\n")
        for filepath, tilts in samples:
            tilt_str = ", ".join(f"{t:.2f}" for t in tilts)
            f.write(f"{Path(filepath).name}, {tilt_str}\n")

    print(f"\nGenerated {n_samples} samples in {output_dir}")
    print(f"Index saved to {index_path}")

    return samples


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default: all tilts = 0
        tilt_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif len(sys.argv) == 7:
        # 6 tilt angles provided
        tilt_angles = [float(x) for x in sys.argv[1:7]]
    elif len(sys.argv) == 2 and sys.argv[1] == "--training":
        # Generate training set
        generate_training_set(
            output_dir="training_samples",
            n_samples=100,
            n_slats=4
        )
        sys.exit(0)
    else:
        print(__doc__)
        sys.exit(1)

    output_path = Path(__file__).parent / "lamella.obj"
    generate_lamella_obj(tilt_angles, output_path=str(output_path))
