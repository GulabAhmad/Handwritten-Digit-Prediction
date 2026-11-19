from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from digit_core import PROJECT_DIR

OUT = PROJECT_DIR / "sample_images"

# Stroke paths in a 0..1 box (x, y), drawn like a pen would.
STROKES = {
    0: [[(0.5, 0.05), (0.2, 0.2), (0.12, 0.5), (0.2, 0.8), (0.5, 0.95), (0.8, 0.8),
         (0.88, 0.5), (0.8, 0.2), (0.5, 0.05)]],
    1: [[(0.3, 0.2), (0.55, 0.05), (0.55, 0.95)]],
    2: [[(0.15, 0.25), (0.35, 0.07), (0.65, 0.07), (0.82, 0.25), (0.75, 0.48),
         (0.15, 0.93), (0.88, 0.93)]],
    3: [[(0.15, 0.1), (0.8, 0.1), (0.45, 0.45), (0.8, 0.6), (0.82, 0.8), (0.55, 0.95),
         (0.15, 0.88)]],
    4: [[(0.65, 0.95), (0.65, 0.05), (0.1, 0.65), (0.9, 0.65)]],
    5: [[(0.8, 0.07), (0.25, 0.07), (0.2, 0.45), (0.6, 0.4), (0.82, 0.6), (0.75, 0.88),
         (0.45, 0.95), (0.15, 0.85)]],
    6: [[(0.75, 0.07), (0.4, 0.2), (0.2, 0.55), (0.25, 0.85), (0.5, 0.95), (0.78, 0.8),
         (0.75, 0.58), (0.5, 0.5), (0.22, 0.62)]],
    7: [[(0.12, 0.08), (0.88, 0.08), (0.4, 0.95)]],
    8: [[(0.5, 0.48), (0.2, 0.3), (0.3, 0.08), (0.7, 0.08), (0.8, 0.3), (0.5, 0.48),
         (0.18, 0.7), (0.3, 0.93), (0.7, 0.93), (0.82, 0.7), (0.5, 0.48)]],
    9: [[(0.8, 0.4), (0.5, 0.5), (0.22, 0.35), (0.3, 0.1), (0.6, 0.05), (0.8, 0.2),
         (0.8, 0.4), (0.7, 0.95)]],
}


def draw_digit(d: int, size=(320, 400), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed + d)
    w, h = size
    paper = rng.normal(225, 6, (h, w)).clip(0, 255).astype(np.uint8)   # textured paper
    img = Image.fromarray(paper, "L").convert("RGB")
    draw = ImageDraw.Draw(img)
    box_w, box_h = int(w * 0.45), int(h * 0.6)
    x0 = int(w * 0.2 + rng.integers(0, int(w * 0.15)))   # deliberately off-centre
    y0 = int(h * 0.12 + rng.integers(0, int(h * 0.15)))
    for path in STROKES[d]:
        pts = [(x0 + px * box_w, y0 + py * box_h) for px, py in path]
        draw.line(pts, fill=(35, 35, 60), width=18, joint="curve")
        for p in pts:
            draw.ellipse([p[0] - 9, p[1] - 9, p[0] + 9, p[1] + 9], fill=(35, 35, 60))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for d in range(10):
        img = draw_digit(d)
        img.save(OUT / f"digit_{d}_drawn.png")
        img.save(OUT / f"digit_{d}_photo.jpg", quality=85)
    blank = np.random.default_rng(1).normal(228, 4, (400, 320)).clip(0, 255).astype(np.uint8)
    Image.fromarray(blank, "L").save(OUT / "blank.png")
    (OUT / "broken.png").write_bytes(b"this is not really a PNG file\n" * 10)
    print(f"Wrote demo images to {OUT}")


if __name__ == "__main__":
    main()
