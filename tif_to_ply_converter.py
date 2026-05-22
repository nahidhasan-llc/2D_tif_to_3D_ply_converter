#!/usr/bin/env python3
"""
cyberware_to_ply.py
────────────────────────────────────────────────────────────────
Convert a Cyberware 3030/RGB range file + paired color TIF
into a colored PLY point cloud.

Scanner specs (Cyberware 3030 PS):
  - Scanning volume : 18" high × 18" diameter (457.2mm × 457.2mm)
  - Valid radius    : ≤ 9" = 228.6mm from center axis
  - ~131,000 points
  - Accuracy        : within 1mm

Usage
-----
    python cyberware_to_ply.py <range_file> <color_tif> [output.ply]

Dependencies
------------
    pip install numpy Pillow

Example
-------
    python cyberware_to_ply.py pat1day0C pat1day0C.tif
    python cyberware_to_ply.py pat1day0C pat1day0C.tif output/pat1day0C.ply
"""

import sys, math, os
import numpy as np
from PIL import Image

INVALID_SENTINEL  = 0x8000
SCANNER_HEIGHT_MM = 18 * 25.4   # 18 inches = 457.2mm
SCANNER_RADIUS_MM = 9  * 25.4   # 9 inches  = 228.6mm (valid scan volume radius)


def parse_header(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    if not raw.startswith(b"Cyberware"):
        raise ValueError(f"'{filepath}' is not a Cyberware range file.")
    idx = raw.find(b"DATA=\n")
    if idx == -1:
        raise ValueError("Could not find DATA= marker.")
    header_end = idx + len(b"DATA=\n")
    params = {}
    for line in raw[:header_end].decode("ascii", errors="replace").split("\n"):
        if "=" in line and not line.startswith("DATA"):
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params, header_end, raw


def cyberware_to_ply(range_path, color_path, output_path):
    print(f"\n{'─'*55}")
    print(f"  Cyberware 3030/RGB → PLY")
    print(f"  Range : {range_path}")
    print(f"  Color : {color_path}")
    print(f"  Out   : {output_path}")
    print(f"{'─'*55}\n")

    # ── 1. Parse header ───────────────────────────────────────────────────────
    params, header_end, raw = parse_header(range_path)
    NLG    = int(params["NLG"])     # rows = angular steps (360°)
    NLT    = int(params["NLT"])     # cols = height steps
    RSHIFT = int(params["RSHIFT"])
    LGINCR = int(params["LGINCR"])  # radius scale factor (1/32768 mm per unit)

    N_THETA    = NLG
    N_Z        = NLT
    # Radius: raw >> RSHIFT, scaled by LGINCR/32768 (mm)
    r_scale_mm = LGINCR / 32768.0
    # Z: derived from known scanner height (18") divided by number of steps
    z_scale_mm = SCANNER_HEIGHT_MM / N_Z
    theta_step = (2.0 * math.pi) / N_THETA

    print(f"  Angular steps : {N_THETA}  ({math.degrees(theta_step):.4f}°/step)")
    print(f"  Height steps  : {N_Z}  ({z_scale_mm:.4f} mm/step → {N_Z*z_scale_mm:.1f} mm total)")
    print(f"  Radius scale  : (raw >> {RSHIFT}) × {r_scale_mm:.6f} mm/unit")
    print(f"  Valid radius  : ≤ {SCANNER_RADIUS_MM:.1f} mm (9\" scanner boundary)\n")

    # ── 2. Read range data ────────────────────────────────────────────────────
    data = (np.frombuffer(raw[header_end:header_end + NLG*NLT*2], dtype=">u2")
              .reshape(NLG, NLT)
              .astype(np.float32))

    valid_mask = (data != INVALID_SENTINEL) & (data > 0)
    radius_mm  = np.where(valid_mask, (data / (2**RSHIFT)) * r_scale_mm, np.nan)

    # Filter to scanner volume boundary — removes chair, stand, background noise
    valid_mask = ~np.isnan(radius_mm) & (radius_mm > 0) & (radius_mm <= SCANNER_RADIUS_MM)

    n_valid = int(valid_mask.sum())
    print(f"  Valid points  : {n_valid:,} / {N_THETA*N_Z:,}  ({100*n_valid/(N_THETA*N_Z):.1f}%)")
    print(f"  Radius range  : {np.nanmin(radius_mm[valid_mask]):.1f} – {np.nanmax(radius_mm[valid_mask]):.1f} mm")

    if n_valid == 0:
        raise RuntimeError("No valid range points found.")

    # ── 3. Cylindrical → Cartesian ────────────────────────────────────────────
    # rows = angular (theta), cols = height (Z)
    Z_grid, THETA = np.meshgrid(
        np.arange(N_Z)     * z_scale_mm,
        np.arange(N_THETA) * theta_step
    )
    X = np.where(valid_mask, radius_mm * np.cos(THETA), np.nan)
    Y = np.where(valid_mask, radius_mm * np.sin(THETA), np.nan)
    Z = Z_grid

    print(f"  X : {np.nanmin(X):.1f} – {np.nanmax(X):.1f} mm")
    print(f"  Y : {np.nanmin(Y):.1f} – {np.nanmax(Y):.1f} mm")
    print(f"  Z : {np.nanmin(Z[valid_mask]):.1f} – {np.nanmax(Z[valid_mask]):.1f} mm\n")

    # ── 4. Color ──────────────────────────────────────────────────────────────
    color  = np.array(Image.open(color_path).convert("RGB"))
    ch, cw = color.shape[:2]

    # ── 5. Build colored point cloud ──────────────────────────────────────────
    rows, cols = np.where(valid_mask)
    pts        = np.column_stack([X[valid_mask], Y[valid_mask], Z[valid_mask]])
    colors     = color[
        (rows * ch / N_THETA).astype(int).clip(0, ch-1),
        (cols * cw / N_Z).astype(int).clip(0, cw-1)
    ].astype(np.uint8)

    # ── 6. Write binary PLY ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    packed = np.zeros(len(pts), dtype=[
        ("x","<f4"),("y","<f4"),("z","<f4"),
        ("r","u1"), ("g","u1"), ("b","u1")
    ])
    packed["x"],packed["y"],packed["z"] = pts[:,0],pts[:,1],pts[:,2]
    packed["r"],packed["g"],packed["b"] = colors[:,0],colors[:,1],colors[:,2]

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(packed.tobytes())

    print(f"  ✓ Saved : {output_path}")
    print(f"    {len(pts):,} points  |  {os.path.getsize(output_path)/1e6:.2f} MB")
    print(f"{'─'*55}\n")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    range_path  = sys.argv[1]
    color_path  = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.splitext(range_path)[0] + ".ply"

    for path in [range_path, color_path]:
        if not os.path.isfile(path):
            print(f"Error: file not found: '{path}'")
            sys.exit(1)

    try:
        cyberware_to_ply(range_path, color_path, output_path)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



# python .\tif_to_ply_converter.py .\pat1day28A .\pat1day28A.tif