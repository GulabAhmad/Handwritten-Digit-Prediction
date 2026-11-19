"""Train the CNN and the ViT on sklearn digits, evaluate fairly, save checkpoints.

Usage:  python train.py            (about 1-2 minutes on a laptop CPU)
"""
from __future__ import annotations

import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from digit_core import (CLASS_NAMES, CNN_PATH, RESULTS_DIR, SEED, SPLIT_PATH, VIT_PATH,
                        DigitCNN, DigitViT, image_to_patches, load_scaled_digits,
                        make_split, save_checkpoint, set_seed)

CNN_CONFIG = dict(channels=32, hidden=128, dropout=0.3)
VIT_CONFIG = dict(embed_dim=64, num_heads=4, depth=2, mlp_dim=128, dropout=0.1)
TRAIN_SETTINGS = {
    "cnn": dict(lr=1e-3, weight_decay=1e-4, max_epochs=80, patience=15, batch_size=64),
    "vit": dict(lr=1e-3, weight_decay=1e-4, max_epochs=200, patience=30, batch_size=64),
}


def to_model_input(model_type: str, images: torch.Tensor) -> torch.Tensor:
    """images [N, 8, 8] in 0..1 -> CNN [N, 1, 8, 8] or ViT [N, 16, 4]."""
    x = images.unsqueeze(1)
    return x if model_type == "cnn" else image_to_patches(x)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x).argmax(dim=1).numpy()


def train_model(model_type: str, model: nn.Module, data: dict) -> tuple[nn.Module, dict]:
    """Train on the training split; pick the epoch with the best VALIDATION accuracy."""
    s = TRAIN_SETTINGS[model_type]
    x_tr, y_tr = data["train"]
    x_val, y_val = data["val"]
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=s["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["max_epochs"])
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0, "state": None}
    t0 = time.time()
    for epoch in range(1, s["max_epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            val_loss = loss_fn(logits, y_val).item()
            val_acc = (logits.argmax(1) == y_val).float().mean().item()
        history["train_loss"].append(total / len(x_tr))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Model selection uses validation data only (ties broken by lower val loss).
        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_type}] epoch {epoch:3d}  train_loss {history['train_loss'][-1]:.4f}"
                  f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")
        if epoch - best["epoch"] >= s["patience"]:
            print(f"  [{model_type}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best["state"])
    model.eval()
    info = {"best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
            "epochs_run": epoch, "train_seconds": round(time.time() - t0, 1),
            "history": history, "settings": s}
    print(f"  [{model_type}] best epoch {best['epoch']}  val_acc {best['val_acc']:.4f}")
    return model, info


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(10)).tolist()}


def top_confusions(cm: np.ndarray, k: int = 3) -> list[tuple[int, int, int]]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(int(off[i, j]), i, j) for i in range(10) for j in range(10) if off[i, j] > 0]
    return [(t, p, c) for c, t, p in sorted(pairs, reverse=True)[:k]]


def plot_confusion_matrices(results: dict, path) -> None:
    names = ["Majority baseline", "CNN", "ViT"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name in zip(axes, names):
        cm = np.array(results[name]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        r = results[name]["test"]
        ax.set_title(f"{name}\nacc {r['accuracy']:.3f} | macro F1 {r['macro_f1']:.3f}")
        ax.set_xlabel("Predicted digit")
        ax.set_ylabel("True digit")
        ax.set_xticks(range(10), CLASS_NAMES)
        ax.set_yticks(range(10), CLASS_NAMES)
    fig.suptitle("Test-set confusion matrices (same held-out split for every model)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(infos: dict, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, info in infos.items():
        h = info["history"]
        axes[0].plot(h["train_loss"], label=f"{name} train")
        axes[0].plot(h["val_loss"], "--", label=f"{name} val")
        axes[1].plot(h["val_acc"], label=f"{name} val")
        axes[1].axvline(info["best_epoch"] - 1, ls=":", color="gray")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Validation accuracy (dotted = selected epoch)")
    axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_errors(x_test, y_test, preds: dict, path, max_per_model: int = 6) -> list[dict]:
    rows, listed = [], []
    for name, p in preds.items():
        wrong = np.where(p != y_test)[0][:max_per_model]
        rows.append((name, wrong, p))
        listed += [{"model": name, "test_position": int(i), "true": int(y_test[i]),
                    "predicted": int(p[i])} for i in wrong]
    cols = max(max_per_model, 1)
    fig, axes = plt.subplots(len(rows), cols, figsize=(2 * cols, 2.3 * len(rows)), squeeze=False)
    for r, (name, wrong, p) in enumerate(rows):
        for c in range(cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(wrong):
                i = wrong[c]
                ax.imshow(x_test[i], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{name}: true {y_test[i]} -> {p[i]}", fontsize=9)
        if len(wrong) == 0:
            axes[r][0].text(0, 0.5, f"{name}: no test errors", fontsize=10)
    fig.suptitle("Misclassified test digits")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return listed


def write_report(results: dict, infos: dict, split_sizes: dict, errors: list, path) -> None:
    lines = ["# Evaluation results", "",
             "All numbers below come from `python train.py` (seed 42).", "",
             "## Data and split", "",
             "- Dataset: `sklearn.datasets.load_digits()` - 1,797 images, 8x8, labels 0-9.",
             "- Scaling: pixel / 16 -> 0..1 (uploads use pixel / 255 -> 0..1).",
             f"- One stratified split shared by every model: train {split_sizes['train']}, "
             f"validation {split_sizes['val']}, test {split_sizes['test']} "
             "(indices saved in `models/split_indices.npz`).",
             "- Training uses the train split only; the checkpoint epoch is chosen by validation "
             "accuracy; the test split is used once, for the final numbers.", "",
             "## Test-set comparison", "",
             "| Model | Val accuracy | Test accuracy | Test macro F1 | Selected epoch | Params |",
             "|---|---|---|---|---|---|"]
    for name in ["Majority baseline", "CNN", "ViT"]:
        r = results[name]
        info = infos.get(name, {})
        lines.append(f"| {name} | {r['val']['accuracy']:.4f} | {r['test']['accuracy']:.4f} | "
                     f"{r['test']['macro_f1']:.4f} | {info.get('best_epoch', '-')} | "
                     f"{r.get('parameters', '-')} |")
    lines += ["", f"Majority baseline always predicts digit **{results['Majority baseline']['predicts']}** "
              "(the most frequent class in the training split).", "",
              "Confusion matrices: `results/confusion_matrices.png`. "
              "Training curves: `results/training_curves.png`. "
              "Misclassified examples: `results/misclassified.png`.", "",
              "## Mistakes", ""]
    for name in ["CNN", "ViT"]:
        cm = np.array(results[name]["test"]["confusion_matrix"])
        n_err = int(cm.sum() - np.trace(cm))
        conf = top_confusions(cm)
        desc = ", ".join(f"{t} read as {p} ({c}x)" for t, p, c in conf) or "none"
        lines.append(f"- **{name}**: {n_err} of {int(cm.sum())} test digits wrong. "
                     f"Most frequent confusions: {desc}.")
    lines += ["", "Example errors (true -> predicted):", ""]
    lines += [f"- {e['model']}: test image #{e['test_position']} true {e['true']} -> predicted "
              f"{e['predicted']}" for e in errors[:8]]
    lines += ["", "## Interpretation", "",
              "- Both networks beat the ~10% majority baseline by a wide margin, so they learn "
              "real digit shape, not class frequency.",
              "- Errors concentrate on digits whose 8x8 versions share strokes (see the confusions "
              "above and `results/misclassified.png`): a narrow 8 with squeezed loops looks like a "
              "1, an 8 with a faint lower loop looks like a 9, and a 2 with a curled base looks "
              "like an 8. At 8x8 resolution one loop is only one or two pixels wide.",
              "- The CNN has a built-in bias for local, shift-tolerant patterns (small kernels + "
              "pooling), which suits tiny images and small data. The ViT must learn spatial "
              "relations between its 16 patches from only ~1,250 training images through "
              "attention and positional embeddings, so it usually needs more epochs and is a "
              "little less accurate.",
              "- Real photos are harder than this dataset: different pen thickness, lighting, "
              "shadows, paper texture, off-centre or tilted digits, and writing styles not in the "
              "sklearn writers. The preprocessing maps photos to the same 0..1 scale and framing, "
              "but matching the scale does not guarantee the same accuracy.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed()
    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_scaled_digits()
    train_idx, val_idx, test_idx = make_split(labels)
    assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx)
                or set(val_idx) & set(test_idx)), "split leakage"
    SPLIT_PATH.parent.mkdir(exist_ok=True)
    np.savez(SPLIT_PATH, train=train_idx, val=val_idx, test=test_idx)
    split_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    print(f"Split sizes: {split_sizes}  (pixel range {images.min():.1f}..{images.max():.1f})")

    imgs_t = torch.tensor(images)
    labels_t = torch.tensor(labels)
    parts = {"train": train_idx, "val": val_idx, "test": test_idx}

    results, infos, test_preds = {}, {}, {}

    # Majority-class baseline, "fit" on the training labels only.
    majority = int(np.bincount(labels[train_idx]).argmax())
    results["Majority baseline"] = {
        "predicts": majority,
        "val": metrics(labels[val_idx], np.full(len(val_idx), majority)),
        "test": metrics(labels[test_idx], np.full(len(test_idx), majority))}

    for name, mtype, cls, cfg, path in (("CNN", "cnn", DigitCNN, CNN_CONFIG, CNN_PATH),
                                        ("ViT", "vit", DigitViT, VIT_CONFIG, VIT_PATH)):
        print(f"\nTraining {name} ...")
        set_seed()
        data = {k: (to_model_input(mtype, imgs_t[idx]), labels_t[idx]) for k, idx in parts.items()}
        print(f"  input shape: {tuple(data['train'][0].shape)}")
        model = cls(**cfg)
        model, info = train_model(mtype, model, data)
        n_params = sum(p.numel() for p in model.parameters())
        val_pred = predict(model, data["val"][0])
        test_pred = predict(model, data["test"][0])
        results[name] = {"parameters": n_params,
                         "val": metrics(labels[val_idx], val_pred),
                         "test": metrics(labels[test_idx], test_pred)}
        infos[name] = info
        test_preds[name] = test_pred
        save_checkpoint(path, model, mtype, cfg, extra={
            "best_epoch": info["best_epoch"], "val_accuracy": info["best_val_acc"],
            "test_accuracy": results[name]["test"]["accuracy"],
            "test_macro_f1": results[name]["test"]["macro_f1"], "seed": SEED})
        print(f"  saved {path.name}: test acc {results[name]['test']['accuracy']:.4f}, "
              f"macro F1 {results[name]['test']['macro_f1']:.4f}")

    plot_confusion_matrices(results, RESULTS_DIR / "confusion_matrices.png")
    plot_curves(infos, RESULTS_DIR / "training_curves.png")
    errors = plot_errors(images[test_idx], labels[test_idx], test_preds,
                         RESULTS_DIR / "misclassified.png")
    summary = {"split_sizes": split_sizes, "results": results,
               "training": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                            for k, v in infos.items()},
               "example_errors": errors}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(results, infos, split_sizes, errors, RESULTS_DIR / "RESULTS.md")

    print("\n=== Test set (held out) ===")
    for name in ["Majority baseline", "CNN", "ViT"]:
        r = results[name]["test"]
        print(f"{name:18s} accuracy {r['accuracy']:.4f}   macro F1 {r['macro_f1']:.4f}")
    print(f"\nWrote checkpoints to {CNN_PATH.parent} and reports to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
