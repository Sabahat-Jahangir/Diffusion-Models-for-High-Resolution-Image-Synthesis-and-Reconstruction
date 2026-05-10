# Diffusion Models for High-Resolution Image Synthesis and Reconstruction

This repository contains a complete implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** built entirely with **PyTorch** for high-resolution image generation and reconstruction tasks.
The project focuses on learning how images are progressively corrupted with noise and then reconstructed through a learned reverse denoising process.

The implementation follows the assignment constraints by using **custom PyTorch modules only** without relying on pretrained diffusion pipelines or external diffusion frameworks.

---

# Project Overview

The goal of this project is to train a diffusion model capable of:

* Learning the forward noising process
* Predicting and removing noise during reverse diffusion
* Generating realistic images from random Gaussian noise
* Reconstructing target images through iterative denoising
* Visualizing intermediate diffusion stages
* Evaluating reconstruction quality using PSNR and SSIM

The project demonstrates the complete workflow of a diffusion-based generative model, including preprocessing, training, sampling, reconstruction, evaluation, and deployment.

---

# Main Features

## Forward Diffusion Process

* Gradual corruption of images using Gaussian noise
* Linear beta noise schedule
* Visualization of noisy images across multiple timesteps

## Reverse Diffusion Process

* Learned denoising using a custom U-Net backbone
* Step-by-step reconstruction of clean images
* Sampling images starting from pure noise

## U-Net Architecture

The denoising network includes:

* Residual convolution blocks
* Time-step embeddings
* Encoder-decoder structure
* Downsampling and upsampling layers
* Channel progression:

  * 64 → 128 → 256

## Training Pipeline

* Mixed Precision Training (AMP)
* AdamW optimizer
* Mean Squared Error (MSE) loss
* Gradient clipping support
* GPU training on Kaggle T4 environment

## Evaluation & Visualization

The notebook provides:

* Forward diffusion previews
* Reverse denoising previews
* Generated image samples
* Target vs reconstructed comparisons
* Training loss curves
* PSNR and SSIM metrics

---

# Dataset

The model can be trained using any of the following datasets:

* CelebA-HQ
* FFHQ
* WikiArt Dataset

Images are:

* Resized to the required resolution
* Normalized to the range [-1, 1]
* Loaded using PyTorch DataLoader with batching support

---

# Project Structure

```bash
├── notebooks/
│   └── diffusion_model.ipynb
│
├── models/
│   └── unet.py
│
├── checkpoints/
│   └── model_checkpoint.pth
│
├── outputs/
│   ├── generated_images/
│   ├── reconstructions/
│   └── training_plots/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# Training Configuration

| Component     | Configuration     |
| ------------- | ----------------- |
| Framework     | PyTorch           |
| Model         | DDPM              |
| Backbone      | Simplified U-Net  |
| Optimizer     | AdamW             |
| Loss Function | MSE Loss          |
| Timesteps     | 200–500           |
| Image Size    | 128×128 / 256×256 |
| Precision     | Mixed Precision   |
| Batch Size    | 16–32             |

---

# Forward Diffusion

The forward process gradually adds Gaussian noise to an image over several timesteps.

The implementation includes:

* Noise schedule generation
* Sampling noisy images
* Visualization of intermediate corruption stages

This helps the model understand how images degrade over time.

---

# Reverse Diffusion

The reverse diffusion process is learned through training.

The U-Net receives:

* A noisy image
* A timestep embedding

The model predicts the noise component, which is removed iteratively to recover a clean image.

This process enables:

* Image generation from random noise
* Image reconstruction from corrupted samples

---

# Reconstruction Task

The repository also supports reconstruction experiments where:

1. A target image is selected
2. Noise is progressively added
3. The reverse diffusion process reconstructs the image

The final reconstructed output is compared with the original image using:

* PSNR
* SSIM

---

# Streamlit / Gradio Application

An interactive app is included for demonstrating the trained diffusion model.

Features:

* Generate images from random noise
* Visualize denoising steps
* Upload custom images
* Perform reconstruction experiments
* Display intermediate outputs interactively

Run the app using:

```bash
streamlit run app.py
```

OR

```bash
gradio app.py
```

(depending on the implementation used)

---

# Installation

Clone the repository:

```bash
git clone <your-repository-link>
cd <repository-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Notebook

Open the notebook in Jupyter or Kaggle:

```bash
jupyter notebook
```

Run all cells sequentially to:

* Train the model
* Generate images
* Visualize diffusion
* Evaluate reconstruction quality

---

# Results

The project includes:

* Generated samples from pure noise
* Reconstruction outputs
* Intermediate denoising visualizations
* Loss vs Epoch plots
* Quantitative evaluation metrics

Generated outputs become progressively sharper as training improves.

---

# Important Notes

* The implementation is fully custom and built using base PyTorch layers only.
* No pretrained diffusion pipelines are used.
* No HuggingFace Diffusers library is used.
* Training high-resolution diffusion models requires significant GPU resources and training time.
* Better image quality can be achieved with longer training and larger datasets.

---

# Future Improvements

Possible extensions for the project include:

* Faster sampling methods
* Cosine noise schedules
* Class-conditioned generation
* Style-conditioned diffusion
* Higher resolution training (256×256+)
* Improved U-Net architecture
* DDIM sampling implementation

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Streamlit / Gradio
* scikit-image

---

# Acknowledgment

This project was developed as part of a Generative AI assignment focused on understanding diffusion-based image generation models and implementing DDPM from scratch using PyTorch.
