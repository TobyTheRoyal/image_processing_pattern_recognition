# Image Processing and Pattern Recognition Projects

## Overview

This repository contains various projects completed for the Image Processing and Pattern Recognition course. Each project focuses on implementing different image processing algorithms and techniques, followed by detailed analysis and reporting of the results.

## Projects

### 1. Guided Image Filtering

#### Task: Implement the Guided Image Filter and Detail Enhancement
- **Guided Image Filter**: Implemented the guided image filter using a sliding window approach and calculated the filtering output based on local linear models.
- **Detail Enhancement**: Enhanced image details by decomposing the image into structure and detail components and recombining them with a detail enhancement coefficient.

### 2. Dehazing and Flash/No-Flash Denoising

#### Task: Implement Dehazing Algorithm and Flash/No-Flash Denoising
- **Dehazing**: Implemented the dark channel prior-based dehazing algorithm, computed the transmission map, refined it using a guided filter, and restored the haze-free image.
- **Flash/No-Flash Denoising**: Applied the guided image filter to denoise a no-flash image using a flash image as guidance, preserving natural colors while reducing noise.

### 3. Coherence Enhancing Diffusion

#### Task: Implement Coherence Enhancing Diffusion Algorithm
- **Implementation**: Implemented the coherence enhancing diffusion (CED) algorithm using structure tensor analysis, performed Eigendecomposition, and applied anisotropic diffusion based on eigenvalues.
- **Discussion**: Analyzed the effects of various parameters on the diffusion process and compared the results with different settings to achieve optimal image enhancement.

### 4. Mean Shift Denoising

#### Task: Implement Mean Shift Denoising Algorithm
- **Derivation**: Derived the update equations for the mean shift iteration based on kernel density estimation (KDE) with DCT coefficients.
- **Implementation**: Implemented the mean shift algorithm, calculated DCT coefficient matrices, and optimized the parameters for image denoising.
- **Visualization**: Visualized the KDE approximation and the trajectory of coefficients during the mean shift process, showing the denoising effect and PSNR development over iterations.

### 5. Total Variation Algorithm for MRI Reconstruction

#### Task: Apply Total Variation Algorithm to Undersampled MRI Data
- **Implementation**: Implemented the total variation (TV) algorithm to reconstruct MRI images from undersampled data.
- **Parameter Tuning**: Explored the effects of the regularization parameter λ on the reconstruction quality, balancing data fidelity and smoothness.
- **Analysis**: Discussed the trade-offs between artifact reduction and detail preservation, and suggested potential improvements using advanced techniques.

Each project involved detailed implementation of the specified algorithms, followed by testing and evaluation using provided datasets. The results were analyzed to understand the impact of different parameters and to optimize the performance of the implemented methods.
