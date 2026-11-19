# Module 04 – Handwritten Digit Prediction (CNN vs. ViT)

Trains a CNN and a small Vision Transformer on `sklearn.datasets.load_digits()` and serves
both in a Streamlit app that turns an uploaded photo into an 8x8 input and shows both predictions.

## Files

| File | Purpose |
|---|---|
| `digit_core.py` | Shared code: data loading/scaling, the one stratified split, `DigitCNN`, `DigitViT`, `image_to_patches`, checkpoint save/load, upload preparation (`open_upload`, `prepare_image`) |
| `train.py` | Trains both models, picks epochs on validation data, evaluates on the test set, saves checkpoints and reports |
| `app.py` | Streamlit app (loads the saved checkpoints; never trains) |
| `check_reload.py` | Fresh-process check: reloads both checkpoints, re-scores the test split, runs the upload pipeline on the demo images |
| `make_samples.py` | Creates demo uploads in `sample_images/` (clear digits, a blank page, a broken file) |
| `models/` | `cnn_digits.pt`, `vit_digits.pt` (weights + model settings + class mapping + scaling), `split_indices.npz` |
| `results/` | `RESULTS.md` (metrics + interpretation), `metrics.json`, `confusion_matrices.png`, `training_curves.png`, `misclassified.png` |

## Run instructions

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py          # ~1-2 min on CPU; writes models/ and results/
python check_reload.py   # optional: confirms the checkpoints reload in a new process
python make_samples.py   # optional: demo images for the app
streamlit run app.py
```

The checkpoints are included, so `streamlit run app.py` works straight away without training.

## How it meets the brief

- **Data + split**: pixels / 16 → 0..1. One stratified 70/15/15 split (seed 42) used by the
  baseline, CNN and ViT; indices are saved. Training uses the train split only, the saved epoch
  is the one with the best validation accuracy, and the test split is used once for the final numbers.
- **CNN**: input `[N, 1, 8, 8]` → 3 Conv2d blocks (BatchNorm, ReLU, MaxPool) → 10 logits.
  CrossEntropyLoss + Adam.
- **ViT**: `image_to_patches` gives sixteen non-overlapping 2x2 patches `[N, 16, 4]` →
  trainable `Linear(4, 64)` patch embedding → [CLS] token + learnable positional embedding →
  2-layer `TransformerEncoder` (4-head self-attention) → 10 logits.
- **Evaluation**: majority-class baseline, test accuracy, macro F1 and a confusion matrix for each
  model, plus misclassified examples and a short interpretation in `results/RESULTS.md`.
- **Upload preparation** (Pillow): open PNG/JPEG → grayscale (transparency → white) → invert if the
  background is bright → remove background → crop to strokes → thicken thin pen lines →
  centre in a square → resize to 8x8 → / 255 → 0..1, the same range as training (/ 16).
- **App**: shows the uploaded photo, an enlarged 8x8 preview of the exact model input, and both
  predicted digits with probabilities. If no file is uploaded, the file is empty, unreadable, not
  PNG/JPEG, blank or almost all ink, the app shows a message instead of crashing.
- **Save + reproduce**: checkpoints hold `state_dict`, model settings, class names, input shape and
  pixel divisors. The app checks that the class mapping matches before using a model.

## Why real photos are harder

The sklearn digits are 8x8 scans from a few dozen writers, tightly cropped, with thick, even
strokes. Photos have lighting gradients, shadows, paper texture, thin or thick pens, tilt and
different writing styles. Preprocessing matches the scale and framing, but a matching scale does
not guarantee the same accuracy as on the test set.
