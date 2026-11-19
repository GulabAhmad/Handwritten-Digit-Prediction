# Evaluation results

All numbers below come from `python train.py` (seed 42).

## Data and split

- Dataset: `sklearn.datasets.load_digits()` - 1,797 images, 8x8, labels 0-9.
- Scaling: pixel / 16 -> 0..1 (uploads use pixel / 255 -> 0..1).
- One stratified split shared by every model: train 1257, validation 270, test 270 (indices saved in `models/split_indices.npz`).
- Training uses the train split only; the checkpoint epoch is chosen by validation accuracy; the test split is used once, for the final numbers.

## Test-set comparison

| Model | Val accuracy | Test accuracy | Test macro F1 | Selected epoch | Params |
|---|---|---|---|---|---|
| Majority baseline | 0.1000 | 0.1037 | 0.0188 | - | - |
| CNN | 0.9926 | 0.9926 | 0.9926 | 31 | 160074 |
| ViT | 0.9852 | 0.9741 | 0.9737 | 59 | 69194 |

Majority baseline always predicts digit **3** (the most frequent class in the training split).

Confusion matrices: `results/confusion_matrices.png`. Training curves: `results/training_curves.png`. Misclassified examples: `results/misclassified.png`.

## Mistakes

- **CNN**: 2 of 270 test digits wrong. Most frequent confusions: 8 read as 1 (2x).
- **ViT**: 7 of 270 test digits wrong. Most frequent confusions: 2 read as 8 (2x), 8 read as 9 (1x), 8 read as 1 (1x).

Example errors (true -> predicted):

- CNN: test image #21 true 8 -> predicted 1
- CNN: test image #231 true 8 -> predicted 1
- ViT: test image #72 true 6 -> predicted 0
- ViT: test image #77 true 2 -> predicted 8
- ViT: test image #78 true 2 -> predicted 8
- ViT: test image #91 true 5 -> predicted 9
- ViT: test image #115 true 4 -> predicted 6
- ViT: test image #171 true 8 -> predicted 9

## Interpretation

- Both networks beat the ~10% majority baseline by a wide margin, so they learn real digit shape, not class frequency.
- Errors concentrate on digits whose 8x8 versions share strokes (see the confusions above and `results/misclassified.png`): a narrow 8 with squeezed loops looks like a 1, an 8 with a faint lower loop looks like a 9, and a 2 with a curled base looks like an 8. At 8x8 resolution one loop is only one or two pixels wide.
- The CNN has a built-in bias for local, shift-tolerant patterns (small kernels + pooling), which suits tiny images and small data. The ViT must learn spatial relations between its 16 patches from only ~1,250 training images through attention and positional embeddings, so it usually needs more epochs and is a little less accurate.
- Real photos are harder than this dataset: different pen thickness, lighting, shadows, paper texture, off-centre or tilted digits, and writing styles not in the sklearn writers. The preprocessing maps photos to the same 0..1 scale and framing, but matching the scale does not guarantee the same accuracy.
