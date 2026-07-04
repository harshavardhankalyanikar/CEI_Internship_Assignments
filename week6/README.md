# MNIST Image Denoising using Autoencoders

## Project Overview

This project implements and compares three Deep Learning-based Denoising Autoencoders using the MNIST handwritten digit dataset. Artificial Gaussian noise is added to clean images, and each model learns to reconstruct the original images from noisy inputs.

The objective is to evaluate different autoencoder architectures for image denoising.

---

## Dataset

Dataset: MNIST Handwritten Digits

Source: Kaggle

Dataset Structure

```
data/
└── MNIST/
    ├── raw/
    │   ├── training/
    │   └── testing/
    │
    └── processed/
```

Training Images : 60,000

Testing Images : 10,000

Image Size : 28 × 28

Classes : 10 (Digits 0–9)

---

## Project Structure

```
AutoEncoder/
│
├── assets/
├── data/
├── models/
├── AutoEncoder.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Implemented Models

### 1. Feed Forward Autoencoder

Architecture

```
784
↓
512
↓
256
↓
64
↓
256
↓
512
↓
784
```

---

### 2. Convolutional Autoencoder (Conv2DTranspose)

Architecture

```
Input
↓
Conv2D
↓
MaxPooling
↓
Conv2D
↓
MaxPooling
↓
Latent Space
↓
Conv2DTranspose
↓
Conv2DTranspose
↓
Output
```

---

### 3. Convolutional Autoencoder (UpSampling2D)

Architecture

```
Input
↓
Conv2D
↓
MaxPooling
↓
Conv2D
↓
MaxPooling
↓
Latent Space
↓
Conv2D
↓
UpSampling
↓
Conv2D
↓
UpSampling
↓
Output
```

---

## Methodology

1. Load the MNIST dataset.
2. Normalize pixel values.
3. Generate Gaussian noisy images.
4. Train three denoising autoencoders.
5. Generate denoised images.
6. Compare model performance using:
   - Binary Crossentropy Loss
   - Mean Squared Error (MSE)
   - Peak Signal-to-Noise Ratio (PSNR)

---

## Results

The notebook includes:

- Original Images
- Noisy Images
- Reconstructed Images
- Training Loss Curves
- Model Comparison
- MSE Comparison
- PSNR Comparison

---

## Performance Metrics

- Binary Crossentropy Loss
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)

---

## Libraries Used

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

---

## How to Run

1. Clone the repository.

```
git clone <repository-url>
```

2. Install dependencies.

```
pip install -r requirements.txt
```

3. Place the Kaggle MNIST dataset inside

```
data/MNIST/raw/
```

4. Open

```
AutoEncoder.ipynb
```

5. Run all notebook cells sequentially.

---

## Results Summary

| Model | Advantages |
|--------|------------|
| Feed Forward Autoencoder | Simple architecture and faster training |
| Conv2DTranspose Autoencoder | Best reconstruction quality and preserves spatial information |
| UpSampling Autoencoder | Stable reconstruction with smooth output images |

---

## Future Improvements

- Variational Autoencoder (VAE)
- Residual Autoencoder
- U-Net based Autoencoder
- Different noise types
- CIFAR-10 image denoising

---

## Author

Deep Learning Assignment

MNIST Denoising Autoencoder
