#!/usr/bin/env python3
"""Downsize PNG and GIF figures in results/ for git inclusion.

PNGs: Resize so max dimension <= 600px, convert RGBA->RGB, then lossy compress with pngquant.
GIFs: Use ImageMagick to aggressively resize and optimize.
Processes in-place (originals can be regenerated from scripts).
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def get_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def downsize_png(path: str, max_dim: int = 600) -> tuple[float, float]:
    """Resize PNG, convert RGBA->RGB, then lossy compress with pngquant."""
    before = get_file_size_mb(path)
    img = Image.open(path)
    w, h = img.size

    # Resize if needed
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert RGBA to RGB (smaller file, white background)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    img.save(path, optimize=True)

    # Apply lossy compression with pngquant (needs temp file for in-place)
    tmp_path = path + ".tmp.png"
    try:
        subprocess.run(
            ["pngquant", "--force", "--quality=40-80", "--speed=1",
             "--output", tmp_path, path],
            check=True, capture_output=True, timeout=30
        )
        os.replace(tmp_path, path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    after = get_file_size_mb(path)
    return before, after


def downsize_gif(
    path: str, max_dim: int = 400, max_frames: int = 40
) -> tuple[float, float]:
    """Use ImageMagick to aggressively resize and optimize GIF."""
    before = get_file_size_mb(path)

    # Skip tiny GIFs
    if before < 0.01:
        return before, before

    # Get frame count
    try:
        result = subprocess.run(
            ["identify", "-format", "%n\n", path],
            capture_output=True, text=True, timeout=30
        )
        n_frames = int(result.stdout.strip().split("\n")[0])
    except Exception:
        n_frames = 1

    tmp_path = path + ".tmp.gif"
    resize_arg = f"{max_dim}x{max_dim}>"

    cmd = ["convert", path, "-coalesce"]

    # Drop frames for large GIFs
    if n_frames > max_frames:
        step = n_frames // max_frames
        frames_to_keep = set(range(0, n_frames, step))
        frames_to_delete = sorted(set(range(n_frames)) - frames_to_keep)
        if frames_to_delete:
            cmd.extend(["-delete", ",".join(str(f) for f in frames_to_delete)])

    cmd.extend([
        "-resize", resize_arg,
        "-fuzz", "5%",
        "-layers", "OptimizePlus",
        "-colors", "64",
        "+dither",
        tmp_path,
    ])

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        os.replace(tmp_path, path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return before, before

    after = get_file_size_mb(path)
    return before, after


def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Error: results/ directory not found")
        sys.exit(1)

    # Process PNGs
    pngs = sorted(glob.glob("results/**/*.png", recursive=True))
    total_before_png = 0.0
    total_after_png = 0.0

    print(f"Processing {len(pngs)} PNG files...")
    for path in pngs:
        before, after = downsize_png(path)
        total_before_png += before
        total_after_png += after
        reduction = (1 - after / before) * 100 if before > 0 else 0
        if reduction > 5:
            print(f"  {path}: {before:.2f}MB -> {after:.2f}MB ({reduction:.0f}%)")

    print(f"\nPNG total: {total_before_png:.1f}MB -> {total_after_png:.1f}MB "
          f"({(1 - total_after_png / total_before_png) * 100:.0f}% reduction)")

    # Process GIFs (already compressed from previous run, just report)
    gifs = sorted(glob.glob("results/**/*.gif", recursive=True))
    total_gif = sum(get_file_size_mb(g) for g in gifs)
    print(f"\nGIF total: {total_gif:.1f}MB ({len(gifs)} files)")

    total = total_after_png + total_gif
    print(f"\nTotal figures: {total:.1f}MB")


if __name__ == "__main__":
    main()
