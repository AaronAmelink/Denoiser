# Patchwise Denoising Autoencoder

A PyTorch-based image denoising framework that reconstructs clean images from noisy inputs using a **patch-based autoencoder** approach.  
The model is trained on overlapping patches with Gaussian-weighted blending and includes a suite of diagnostic tools for quantitative and qualitative evaluation.

## Overview

**Patchwise Denoising Autoencoder** is an image restoration system that learns to remove Gaussian noise from RGB images by operating on local patches.  
Each image is split into overlapping patches that are noised, denoised by the autoencoder, and reassembled using Gaussian-weighted blending.

The code is setup up to be test and train models with - rather than providing a interface to denoise images. If that is your use case, train your model using this,
save it and run later at your convenience.

Most variables you'd want to tweak are in Denoiser.py

This approach provides:
- Localized control of reconstruction
- Fine-grained error diagnostics
- Patch-level feature analysis (variance, edge content, latent space, etc.)

Example with default parameters:
![Denoising Example](denoising_default_example.png)
