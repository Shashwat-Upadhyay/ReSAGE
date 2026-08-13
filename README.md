# ReSAGE: Resolution-Enhanced Super-resolution of Advanced Genomic Images

ReSAGE is a deep learning project aimed at restoring and super-resolving noisy genomic/microscopy images.

## Project Structure

```
ReSAGE/
├── data/
│   ├── test/
│   │   └── NoisyLR/
│   └── train/
│       ├── NoisyLR/
│       └── GT/
│
├── src/
│   └── datasets/
│       ├── inspect_dataset.py
│       └── dataset.py
│
├── scripts/
├── checkpoints/
├── outputs/
├── reports/
├── experiments/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Details
- NoisyLR: Noisy Low Resolution NumPy arrays (`.npy`)
- GT: High Resolution Ground Truth NumPy arrays (`.npy`)
