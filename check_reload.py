"""Fresh-process check: reload both checkpoints (no training), re-score the saved
test split, and run the upload pipeline on the demo images.

Usage:  python check_reload.py
"""
from __future__ import annotations

import numpy as np
import torch

from digit_core import (CNN_PATH, PROJECT_DIR, SPLIT_PATH, VIT_PATH, ImageRejected,
                        image_to_patches, load_checkpoint, load_scaled_digits, open_upload,
                        predict_both, prepare_image)


def main() -> None:
    cnn, cnn_ckpt = load_checkpoint(CNN_PATH)
    vit, vit_ckpt = load_checkpoint(VIT_PATH)
    images, labels = load_scaled_digits()
    test_idx = np.load(SPLIT_PATH)["test"]
    x = torch.tensor(images[test_idx]).unsqueeze(1)
    y = labels[test_idx]
    with torch.no_grad():
        cnn_acc = (cnn(x).argmax(1).numpy() == y).mean()
        vit_acc = (vit(image_to_patches(x)).argmax(1).numpy() == y).mean()
    print(f"Reloaded CNN test accuracy {cnn_acc:.4f} (saved {cnn_ckpt['test_accuracy']:.4f})")
    print(f"Reloaded ViT test accuracy {vit_acc:.4f} (saved {vit_ckpt['test_accuracy']:.4f})")
    assert abs(cnn_acc - cnn_ckpt["test_accuracy"]) < 1e-9
    assert abs(vit_acc - vit_ckpt["test_accuracy"]) < 1e-9

    sample_dir = PROJECT_DIR / "sample_images"
    if not sample_dir.exists():
        print("No sample_images/ folder - run python make_samples.py to test uploads.")
        return
    print("\nUpload pipeline on sample_images/:")
    for path in sorted(sample_dir.iterdir()):
        try:
            arr = prepare_image(open_upload(path.read_bytes()))
            p = predict_both(cnn, vit, arr)
            print(f"  {path.name:20s} range {arr.min():.2f}-{arr.max():.2f}  "
                  f"CNN {p['CNN'][0]}  ViT {p['ViT'][0]}")
        except ImageRejected as exc:
            print(f"  {path.name:20s} REJECTED: {exc}")
    for label, data in (("missing (empty bytes)", b""),):
        try:
            open_upload(data)
        except ImageRejected as exc:
            print(f"  {label:20s} REJECTED: {exc}")


if __name__ == "__main__":
    main()
