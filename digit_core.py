"""Shared code for the handwritten digit project.

Everything that BOTH training (train.py) and the app (app.py) need lives here,
so the data scaling, patching, class mapping and model definitions can never
drift apart between the two paths.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Constants shared by training and the app
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
CNN_PATH = MODEL_DIR / "cnn_digits.pt"
VIT_PATH = MODEL_DIR / "vit_digits.pt"
SPLIT_PATH = MODEL_DIR / "split_indices.npz"

SEED = 42
CLASS_NAMES = [str(d) for d in range(10)]  # class index i  <->  digit i
SKLEARN_MAX_PIXEL = 16.0   # sklearn digits use 0..16
UPLOAD_MAX_PIXEL = 255.0   # 8-bit photos use 0..255
IMAGE_SIZE = 8
PATCH_SIZE = 2
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 16
PATCH_DIM = PATCH_SIZE * PATCH_SIZE            # 4


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# 1. Dataset + one shared stratified split
# --------------------------------------------------------------------------- #
def load_scaled_digits():
    """Return images [N, 8, 8] scaled to 0..1 (pixel / 16) and labels [N]."""
    digits = load_digits()
    images = digits.images.astype(np.float32) / SKLEARN_MAX_PIXEL
    labels = digits.target.astype(np.int64)
    return images, labels


def make_split(labels: np.ndarray, seed: int = SEED):
    """One stratified 70 / 15 / 15 train / validation / test split (indices)."""
    all_idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.30, stratify=labels, random_state=seed)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=seed)
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


# --------------------------------------------------------------------------- #
# 2. CNN: [N, 1, 8, 8] -> 10 logits
# --------------------------------------------------------------------------- #
class DigitCNN(nn.Module):
    def __init__(self, channels: int = 32, hidden: int = 128, dropout: float = 0.3,
                 num_classes: int = 10):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),        # [N, c, 8, 8]
            nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, kernel_size=3, padding=1),    # [N, 2c, 8, 8]
            nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.MaxPool2d(2),                                  # [N, 2c, 4, 4]
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, padding=1),  # [N, 4c, 4, 4]
            nn.BatchNorm2d(4 * c), nn.ReLU(),
            nn.MaxPool2d(2),                                  # [N, 4c, 2, 2]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(4 * c * 2 * 2, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or tuple(x.shape[1:]) != (1, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"DigitCNN expects [N, 1, 8, 8], got {tuple(x.shape)}")
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# 3. Vision Transformer: [N, 16, 4] -> 10 logits
# --------------------------------------------------------------------------- #
def image_to_patches(images: torch.Tensor) -> torch.Tensor:
    """Split 8x8 images into sixteen non-overlapping 2x2 patches.

    Accepts [N, 8, 8] or [N, 1, 8, 8]; returns [N, 16, 4].
    Patches are ordered row-major (top-left, then left-to-right, top-to-bottom)
    and each patch is flattened row-major: [p00, p01, p10, p11].
    """
    if images.dim() == 4:
        images = images.squeeze(1)
    n = images.shape[0]
    g = IMAGE_SIZE // PATCH_SIZE  # 4 patches per side
    patches = images.reshape(n, g, PATCH_SIZE, g, PATCH_SIZE)  # [N, gr, pr, gc, pc]
    patches = patches.permute(0, 1, 3, 2, 4)                   # [N, gr, gc, pr, pc]
    return patches.reshape(n, g * g, PATCH_DIM)                # [N, 16, 4]


class DigitViT(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, depth: int = 2,
                 mlp_dim: int = 128, dropout: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, embed_dim)          # trainable projection
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, embed_dim))  # positions
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth,
                                             enable_nested_tensor=False)  # self-attention
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 3 or tuple(patches.shape[1:]) != (NUM_PATCHES, PATCH_DIM):
            raise ValueError(f"DigitViT expects [N, 16, 4], got {tuple(patches.shape)}")
        x = self.patch_embed(patches)                         # [N, 16, D]
        cls = self.cls_token.expand(x.shape[0], -1, -1)       # [N, 1, D]
        x = torch.cat([cls, x], dim=1) + self.pos_embed       # [N, 17, D]
        x = self.encoder(self.dropout(x))                     # attention over patches
        return self.head(self.norm(x[:, 0]))                  # classify from [CLS]


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
MODEL_CLASSES = {"cnn": DigitCNN, "vit": DigitViT}


def save_checkpoint(path: Path, model: nn.Module, model_type: str, config: dict,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": model_type,
        "config": config,
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "input_range": [0.0, 1.0],
        "train_pixel_divisor": SKLEARN_MAX_PIXEL,
        "upload_pixel_divisor": UPLOAD_MAX_PIXEL,
        "input_shape": [1, 8, 8] if model_type == "cnn" else [NUM_PATCHES, PATCH_DIM],
        **(extra or {}),
    }, path)


def load_checkpoint(path: Path):
    """Rebuild a model from a checkpoint. Returns (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = MODEL_CLASSES[ckpt["model_type"]](**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt["class_names"] != CLASS_NAMES:
        raise ValueError(f"Class mapping in {path.name} does not match the app.")
    return model, ckpt


@torch.no_grad()
def predict_both(cnn: nn.Module, vit: nn.Module, image01: np.ndarray):
    """image01: [8, 8] float array in 0..1. Returns dict of (digit, probabilities)."""
    x = torch.as_tensor(image01, dtype=torch.float32).reshape(1, 1, 8, 8)
    out = {}
    for name, model, inp in (("CNN", cnn, x), ("ViT", vit, image_to_patches(x))):
        model.eval()
        probs = torch.softmax(model(inp), dim=1)[0].numpy()
        out[name] = (int(probs.argmax()), probs)
    return out


# --------------------------------------------------------------------------- #
# 5. Upload preparation (Pillow)
# --------------------------------------------------------------------------- #
class ImageRejected(ValueError):
    """Raised with a user-friendly message when an upload cannot be used."""


ALLOWED_FORMATS = {"PNG", "JPEG", "MPO"}  # MPO = JPEG variant some phones produce


def open_upload(data: bytes) -> Image.Image:
    """Open PNG/JPEG bytes with Pillow, or raise ImageRejected."""
    if not data:
        raise ImageRejected("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                    # detects truncated / corrupt files
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageRejected(f"The file could not be read as an image ({exc.__class__.__name__}).") from exc
    if img.format not in ALLOWED_FORMATS:
        raise ImageRejected(f"Only PNG or JPEG images are supported (got {img.format}).")
    if min(img.size) < 8:
        raise ImageRejected(f"The image is too small ({img.size[0]}x{img.size[1]}).")
    return ImageOps.exif_transpose(img)       # respect phone rotation


def to_grayscale(img: Image.Image) -> Image.Image:
    """Convert to 8-bit grayscale; transparent areas become white paper."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(paper, rgba)
    return img.convert("L")


def prepare_image(img: Image.Image, return_steps: bool = False):
    """Turn a photo of one digit into an 8x8 float array in 0..1 like sklearn digits.

    grayscale -> invert if background is bright -> remove background ->
    crop strokes -> thicken -> pad to centred square -> resize 8x8 -> /255
    """
    gray = to_grayscale(img)
    gray.thumbnail((512, 512))                         # speed; keeps aspect ratio
    arr = np.asarray(gray, dtype=np.float32)

    # Background level estimated from the image border.
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if np.median(border) > 127:                        # bright paper, dark ink
        arr = 255.0 - arr                              # -> bright strokes on dark
    background = float(np.median(np.concatenate(
        [arr[0], arr[-1], arr[:, 0], arr[:, -1]])))

    contrast = float(np.percentile(arr, 99.5) - background)
    if contrast < 40:
        raise ImageRejected("The image looks blank (no visible pen strokes).")

    # Remove the background: everything below a threshold becomes 0.
    threshold = background + 0.35 * contrast
    stroke_mask = arr > threshold
    ink_fraction = stroke_mask.mean()
    if stroke_mask.sum() < 15 or ink_fraction < 0.001:
        raise ImageRejected("The image looks blank (too few stroke pixels).")
    if ink_fraction > 0.5:
        raise ImageRejected("The image is mostly ink - upload one dark digit on plain paper.")
    clean = np.where(stroke_mask, arr - background, 0.0)
    clean = np.clip(clean / max(clean.max(), 1e-6) * 255.0, 0, 255)

    # Crop to the stroke bounding box.
    rows = np.where(stroke_mask.any(axis=1))[0]
    cols = np.where(stroke_mask.any(axis=0))[0]
    crop = clean[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = crop.shape
    crop_img = Image.fromarray(crop.astype(np.uint8))

    # Thicken thin pen strokes so they survive downsampling (sklearn strokes are thick).
    side = max(h, w)
    k = int(side / 12) | 1                             # odd kernel size
    if k >= 3:
        crop_img = crop_img.filter(ImageFilter.MaxFilter(k))

    # Centre the digit in a square with a small margin (sklearn digits fill the frame).
    margin = int(round(side * 0.12))
    square = Image.new("L", (side + 2 * margin, side + 2 * margin), 0)
    square.paste(crop_img, ((square.width - w) // 2, (square.height - h) // 2))

    small = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BOX)
    small = ImageOps.autocontrast(small)               # strongest stroke pixel -> 255
    image01 = np.asarray(small, dtype=np.float32) / UPLOAD_MAX_PIXEL  # same 0..1 scale as /16

    if image01.max() <= 0:
        raise ImageRejected("The image looks blank after preparation.")
    if return_steps:
        return image01, {"grayscale": gray, "cleaned_crop": crop_img, "centred": square}
    return image01
