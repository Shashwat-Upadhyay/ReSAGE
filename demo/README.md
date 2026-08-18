# ReSAGE — Streamlit Demonstration UI

This directory contains the standalone **Streamlit demonstration application** for the ReSAGE semiconductor image restoration project.

## Overview

The application visualizes the ReSAGE restoration pipeline on semiconductor images:
- **Degraded Input**: Low-resolution noisy image (`128 × 128`, 1-channel)
- **Joint SAE Restored Output**: High-resolution feature-restored output (`256 × 256`, 1-channel)
- **Ground Truth**: Original clean image (`256 × 256`, 1-channel)

## How to Run

Launch the Streamlit app from the project root directory:

```bash
streamlit run demo/app.py
```

## Features

- **Random Sample Selection**: Click "Random Sample" or "Next Sample" to interactively explore paired `.npy` dataset samples.
- **Side-by-Side 3-Column Visualization**: Visual comparison with clear emphasis on the central Joint SAE restored output.
- **Image Restoration Comparison**: Feature table detailing exact NumPy array statistics (Min, Max, Mean, Std Dev).
- **Restoration Pipeline Flow**: Interactive UI diagram illustrating the network processing steps.
- **Validation Results**: Metric cards displaying PSNR (28.3865 dB) and SSIM (0.7761) improvements over baseline.
- **About Model Details**: Collapsible section explaining the Sparse Autoencoder architecture.
