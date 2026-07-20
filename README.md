# Low-Dose to Standard-Dose Brain MRI Synthesis Using Pix2Pix

A research-oriented PyTorch implementation of a conditional Generative Adversarial Network for synthesizing **standard-dose-like contrast-enhanced brain MRI** from MRI acquired using approximately **20% of the standard gadolinium contrast dose**.

The current implementation uses a Pix2Pix-style architecture consisting of a compact U-Net generator and a PatchGAN discriminator.

> [!CAUTION]
> This project is intended exclusively for research and educational purposes. The generated images are synthetic model outputs and must not be used as a substitute for standard contrast-enhanced MRI, radiological interpretation, clinical diagnosis, treatment planning, or medical decision-making.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Research Objective](#research-objective)
* [Input and Output Definition](#input-and-output-definition)
* [Dataset](#dataset)
* [MRI Acquisition Information](#mri-acquisition-information)
* [Image Pairing and Registration](#image-pairing-and-registration)
* [Data Preprocessing](#data-preprocessing)
* [Data Splitting](#data-splitting)
* [Model Architecture](#model-architecture)
* [Loss Functions](#loss-functions)
* [Training Configuration](#training-configuration)
* [Training Environment](#training-environment)
* [Quantitative Evaluation](#quantitative-evaluation)
* [Qualitative Results](#qualitative-results)
* [Failure Cases](#failure-cases)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Running the Notebook](#running-the-notebook)
* [Model Checkpoints](#model-checkpoints)
* [Limitations](#limitations)
* [Most Important Research Consideration](#most-important-research-consideration)
* [Medical and Ethical Considerations](#medical-and-ethical-considerations)
* [Future Work](#future-work)
* [Reference](#reference)
* [Author](#author)

---

## Project Overview

Gadolinium-based contrast agents can improve the visualization of tumors, lesions, vascular structures, and abnormal tissue enhancement in brain MRI.

This project investigates whether a conditional image-to-image translation model can learn to generate a synthetic standard-dose-like contrast-enhanced MRI image from a corresponding low-dose scan acquired using approximately 20% of the standard gadolinium dose.

The translation task can be summarized as:

```text
20% gadolinium-dose brain MRI
                │
                ▼
        Pix2Pix Generator
                │
                ▼
Synthetic standard-dose-like brain MRI
```

Although the source imaging data may contain non-contrast, low-dose, and standard-dose acquisitions, the current model uses only the following paired domains:

* Low-dose contrast-enhanced MRI
* Standard-dose contrast-enhanced MRI

Non-contrast MRI is not used as an input or target in the current training pipeline.

---

## Research Objective

The primary objective is to evaluate whether a paired conditional GAN can learn the mapping:

```text
Low-dose contrast-enhanced MRI
                    ↓
Synthetic standard-dose-like MRI
```

The model is designed to increase the apparent contrast enhancement of the low-dose input while preserving anatomical structures.

This is a preliminary proof-of-concept experiment. It does not establish clinical equivalence between the generated image and an actual MRI acquired using the standard gadolinium dose.

---

## Input and Output Definition

### Model Input

The input to the generator is:

* One two-dimensional brain MRI slice
* Acquired using approximately 20% of the standard gadolinium dose
* Converted to a three-channel RGB image
* Resized to `256 × 256` pixels
* Normalized to the range `[-1, 1]`

Tensor shape:

```text
Batch × 3 × 256 × 256
```

### Training Target

The target image is:

* The spatially corresponding standard-dose contrast-enhanced MRI slice
* Converted to RGB
* Resized to `256 × 256`
* Normalized to `[-1, 1]`

### Model Output

The generator produces:

* A three-channel `256 × 256` synthetic MRI image
* Intended to resemble the corresponding standard-dose contrast-enhanced MRI
* Represented internally in the range `[-1, 1]` because of the final `Tanh` activation

The output should be described as:

```text
Synthetic standard-dose-like MRI
```

It should not be described as an actual standard-dose acquisition.

---

## Dataset

The dataset contains paired contrast-enhanced brain MRI data from **8 patients**.

After image preparation, the dataset used by the current notebook contains:

| Dataset property       |          Value |
| ---------------------- | -------------: |
| Number of patients     |              8 |
| Total paired 2D slices |          2,000 |
| Training pairs         |          1,500 |
| Validation pairs       |            500 |
| Independent test pairs |              0 |
| Number of MRI series   | Not documented |
| Slices per patient     |   Not reported |
| Slices per series      |   Not reported |

Each paired sample consists of:

| Domain | Description                                                   |
| ------ | ------------------------------------------------------------- |
| `A`    | MRI acquired using approximately 20% gadolinium dose          |
| `B`    | Corresponding MRI acquired using the standard gadolinium dose |

The model learns the transformation:

```text
A → B
```

where:

```text
A    = low-dose input
B    = real standard-dose target
G(A) = synthetic standard-dose-like output
```

### Expected Dataset Structure

```text
pix2pix_dataset/
├── train/
│   ├── A/
│   │   ├── patient0_*.jpg
│   │   ├── patient1_*.jpg
│   │   └── ...
│   └── B/
│       ├── patient0_*.jpg
│       ├── patient1_*.jpg
│       └── ...
└── val/
    ├── A/
    │   ├── patient0_*.jpg
    │   ├── patient1_*.jpg
    │   └── ...
    └── B/
        ├── patient0_*.jpg
        ├── patient1_*.jpg
        └── ...
```

### Dataset Availability

The medical imaging dataset is not included in this repository.

Medical imaging data may be subject to:

* Patient privacy requirements
* Institutional approval
* Ethical review
* Data-use agreements
* Licensing restrictions
* De-identification requirements

No protected health information should be committed to this repository.

---

## MRI Acquisition Information

The exact MRI acquisition parameters are not included in the current notebook or repository.

| Acquisition property         | Status         |
| ---------------------------- | -------------- |
| MRI sequence                 | Not documented |
| Scanner manufacturer         | Not documented |
| Scanner model                | Not documented |
| Magnetic field strength      | Not documented |
| Repetition time              | Not documented |
| Echo time                    | Not documented |
| Flip angle                   | Not documented |
| Voxel spacing                | Not documented |
| Slice thickness              | Not documented |
| Gadolinium agent             | Not documented |
| Standard dose in mmol/kg     | Not documented |
| Injection timing             | Not documented |
| Number of acquisition series | Not documented |

The MRI sequence must not be assumed solely from image appearance. For example, the data should only be described as post-contrast T1-weighted MRI if this is verified using the original DICOM or acquisition metadata.

Future documentation should report:

* Sequence name
* Scanner manufacturer and model
* Field strength
* Acquisition plane
* Spatial resolution
* Slice thickness
* Contrast agent
* Exact low and standard doses
* Injection-to-scan delay
* Number of series per patient

---

## Image Pairing and Registration

### Pairing Implemented in the Notebook

The notebook extracts a patient number from each filename using the following naming pattern:

```text
patient<number>
```

Images are then:

1. Grouped by patient identifier
2. Sorted lexicographically within each patient
3. Paired sequentially between domains `A` and `B`
4. Truncated to the smaller domain size when the numbers of images differ

Conceptually:

```text
sorted low-dose images    → A1, A2, A3, ...
sorted standard-dose data → B1, B2, B3, ...

paired samples            → (A1, B1), (A2, B2), ...
```

### Registration Status

The notebook does **not** perform image registration.

No rigid, affine, deformable, or learning-based registration method is implemented in the current pipeline.

The model therefore assumes that:

* Low-dose and standard-dose images have already been spatially aligned
* Corresponding filenames represent the same anatomical slice
* Sequentially sorted images have correct anatomical correspondence

The original registration method used to create the paired dataset is currently **not documented**.

This is an important methodological limitation because incorrect alignment can cause the model to learn anatomical displacement rather than only contrast transformation.

Future versions should report:

* Registration software
* Registration type
* Similarity metric
* Interpolation method
* Reference image
* Transformation parameters
* Quality-control procedure
* Criteria for rejecting misregistered pairs

---

## Data Preprocessing

The preprocessing implemented in the notebook consists of the following operations.

### 1. Image Loading

Images are loaded using Pillow:

```python
Image.open(path).convert("RGB")
```

Therefore, the current model operates on RGB image files rather than original DICOM or NIfTI volumes.

### 2. Spatial Resizing

Every image is resized to:

```text
256 × 256 pixels
```

The notebook does not explicitly configure the interpolation method, so the default interpolation behavior of the installed torchvision version is used.

### 3. Tensor Conversion

Images are converted to PyTorch tensors using:

```python
transforms.ToTensor()
```

This initially scales image intensities to approximately:

```text
[0, 1]
```

### 4. Intensity Normalization

All three channels are normalized using:

```python
mean = (0.5, 0.5, 0.5)
std  = (0.5, 0.5, 0.5)
```

The resulting intensity range is approximately:

```text
[-1, 1]
```

The normalization equation is:

```text
normalized = (pixel - 0.5) / 0.5
```

### Preprocessing Not Implemented

The current notebook does not document or implement:

* DICOM intensity calibration
* NIfTI volume processing
* Bias-field correction
* N4 correction
* Skull stripping
* Brain extraction
* Denoising
* Artifact correction
* Histogram matching
* Z-score normalization
* Percentile clipping
* Voxel-spacing standardization
* Slice-thickness standardization
* Cropping to a brain bounding box
* Data augmentation
* Random rotations
* Random flipping
* Elastic deformation
* Registration
* Lesion segmentation
* Background masking

---

## Data Splitting

The current dataset is divided as follows:

| Split      | Paired slices | Percentage |
| ---------- | ------------: | ---------: |
| Training   |         1,500 |        75% |
| Validation |           500 |        25% |
| Test       |             0 |         0% |
| Total      |         2,000 |       100% |

### Splitting Unit

The split is performed at the **slice level**, not at the patient level.

This means that slices from the same patient may appear in both the training and validation subsets.

```text
8 patients
    │
    ▼
2,000 paired slices
    │
    ▼
Slice-level split
    ├── 1,500 training pairs
    └──   500 validation pairs
```

### Important Data-Leakage Risk

Because the split is slice-based, anatomically similar slices from the same patient may occur in both subsets.

As a result:

* Validation performance may be optimistically biased
* The model may learn patient-specific anatomy
* The model may learn scanner- or acquisition-specific characteristics
* The results do not demonstrate generalization to unseen patients
* The validation set is not patient-independent

There is currently no independent test set.

A stronger evaluation should use one of the following:

* Patient-level train, validation, and test splits
* Leave-one-patient-out cross-validation
* Patient-wise k-fold cross-validation
* External validation on another scanner or institution

---

## Model Architecture

The project uses a conditional GAN inspired by Pix2Pix.

The system contains:

* A U-Net-style generator
* A conditional PatchGAN discriminator

---

## Generator

The generator receives a low-dose MRI slice and generates a synthetic standard-dose-like image.

### Encoder

| Layer  | Input channels | Output channels | Kernel | Stride | Normalization | Activation    |
| ------ | -------------: | --------------: | -----: | -----: | ------------- | ------------- |
| Down 1 |              3 |              64 |  4 × 4 |      2 | None          | LeakyReLU 0.2 |
| Down 2 |             64 |             128 |  4 × 4 |      2 | BatchNorm     | LeakyReLU 0.2 |
| Down 3 |            128 |             256 |  4 × 4 |      2 | BatchNorm     | LeakyReLU 0.2 |

### Decoder

| Layer | Input channels | Output channels | Kernel | Stride | Normalization | Activation |
| ----- | -------------: | --------------: | -----: | -----: | ------------- | ---------- |
| Up 1  |            256 |             128 |  4 × 4 |      2 | BatchNorm     | ReLU       |
| Up 2  |            256 |              64 |  4 × 4 |      2 | BatchNorm     | ReLU       |
| Up 3  |            128 |               3 |  4 × 4 |      2 | None          | Tanh       |

Skip connections concatenate:

```text
Down 2 → Up 1 output
Down 1 → Up 2 output
```

These skip connections help preserve spatial and anatomical information.

### Generator Flow

```text
Input: 3 × 256 × 256
        │
        ▼
64 × 128 × 128
        │
        ▼
128 × 64 × 64
        │
        ▼
256 × 32 × 32
        │
        ▼
128 × 64 × 64
        │
        ├── concatenate encoder feature
        ▼
64 × 128 × 128
        │
        ├── concatenate encoder feature
        ▼
Output: 3 × 256 × 256
```

This is a compact U-Net-style model rather than the full eight-level U-Net commonly used in the original Pix2Pix implementation.

---

## Discriminator

The discriminator receives two images:

* The low-dose input
* Either the real standard-dose target or the generated output

The two RGB images are concatenated to form a six-channel tensor:

```text
3 low-dose channels + 3 target/generated channels = 6 channels
```

### Discriminator Layers

| Layer  | Input channels | Output channels | Kernel | Stride | Normalization | Activation    |
| ------ | -------------: | --------------: | -----: | -----: | ------------- | ------------- |
| Conv 1 |              6 |              64 |  4 × 4 |      2 | None          | LeakyReLU 0.2 |
| Conv 2 |             64 |             128 |  4 × 4 |      2 | BatchNorm     | LeakyReLU 0.2 |
| Conv 3 |            128 |             256 |  4 × 4 |      2 | BatchNorm     | LeakyReLU 0.2 |
| Output |            256 |               1 |  4 × 4 |      1 | None          | Sigmoid       |

For a `256 × 256` input, the discriminator produces a spatial prediction map rather than one global real/fake value.

This encourages the model to learn local:

* Contrast patterns
* Edges
* Anatomical textures
* Tissue boundaries
* Enhancement characteristics

---

## Loss Functions

### Generator Loss

The generator is optimized using:

```text
L_G = L_GAN + λL1 × L_L1
```

where:

* `L_GAN` is binary cross-entropy adversarial loss
* `L_L1` is pixel-wise L1 reconstruction loss
* `λL1 = 100`

The implemented generator loss is:

```python
loss_GAN = BCE(D(G(A), A), 1)
loss_L1  = L1(G(A), B) × 100
loss_G   = loss_GAN + loss_L1
```

The adversarial loss encourages realistic local appearance.

The L1 loss encourages correspondence with the real standard-dose target and reduces large pixel-level differences.

### Discriminator Loss

The discriminator loss is:

```text
L_D = 0.5 × (L_real + L_fake)
```

where:

```python
L_real = BCE(D(B, A), 1)
L_fake = BCE(D(G(A), A), 0)
```

The generated image is detached during discriminator optimization.

---

## Training Configuration

| Hyperparameter              |                Value |
| --------------------------- | -------------------: |
| Framework                   |              PyTorch |
| Input size                  |            256 × 256 |
| Input channels              |                    3 |
| Output channels             |                    3 |
| Batch size                  |                    4 |
| Training epochs             |                   60 |
| Training samples            |  1,500 paired slices |
| Validation samples          |    500 paired slices |
| Generator optimizer         |                 Adam |
| Discriminator optimizer     |                 Adam |
| Generator learning rate     |               0.0002 |
| Discriminator learning rate |               0.0002 |
| Adam β₁                     |                  0.5 |
| Adam β₂                     |                0.999 |
| L1 coefficient              |                  100 |
| GAN loss                    | Binary cross-entropy |
| Reconstruction loss         |                   L1 |
| Training shuffle            |              Enabled |
| Validation shuffle          |             Disabled |
| DataLoader workers          |                    2 |
| Random seed                 |              Not set |
| Learning-rate scheduler     |             Not used |
| Mixed-precision training    |             Not used |
| Early stopping              |             Not used |

### Weight Initialization

No explicit custom weight initialization is applied.

The networks therefore use the default initialization behavior of the installed PyTorch version.

### Model Selection

The notebook saves `best_generator.pth` when the generator loss from the final processed batch of an epoch is lower than the previously stored value.

This is not equivalent to selecting the model using:

* Mean epoch loss
* Validation loss
* Validation SSIM
* LPIPS
* Patient-level performance

The final quantitative evaluation loads `generator_final.pth`, not necessarily `best_generator.pth`.

---

## Training Environment

The notebook was designed to run in Google Colab and mounts the dataset from Google Drive.

The model is explicitly assigned to:

```python
torch.device("cuda")
```

Therefore, the current code requires a CUDA-enabled environment unless it is modified to support a CPU fallback.

| Equipment property | Status                  |
| ------------------ | ----------------------- |
| Platform           | Google Colab            |
| Accelerator type   | CUDA GPU                |
| Exact GPU model    | NVIDIA Tesla T4         |
| GPU memory         | 16 GB GDDR6             |
| CPU model          | typically an Intel Xeon virtual CPU |
| System memory      | usually about 12–13 GB  |

---

## Quantitative Evaluation

### Structural Similarity Index Measure

The current notebook evaluates generated images using SSIM.

Before evaluation:

1. Generated and target tensors are moved to the CPU
2. Tensor dimensions are converted from `B × C × H × W` to `B × H × W × C`
3. Images are denormalized from `[-1, 1]` to `[0, 1]`
4. SSIM is calculated separately for every validation slice

The evaluation configuration is:

| SSIM parameter   | Value                                   |
| ---------------- | --------------------------------------- |
| Implementation   | `skimage.metrics.structural_similarity` |
| Data range       | 1.0                                     |
| Channel handling | `channel_axis=2`                        |
| Window size      | 7 for 256 × 256 images                  |
| Evaluation unit  | Individual 2D slice                     |
| Evaluation split | Slice-level validation set              |

### Current Results

| Metric                       |        Result |
| ---------------------------- | ------------: |
| Mean validation SSIM         |    **0.8615** |
| Maximum validation SSIM      |    **0.9857** |
| Index of maximum-SSIM sample |       **155** |

### Interpretation

The mean SSIM indicates substantial structural similarity between generated and target images within the current validation setup.

However, this result does not establish:

* Diagnostic accuracy
* Preservation of tumor enhancement
* Preservation of lesion boundaries
* Clinical interchangeability
* Generalization to unseen patients
* Generalization to other scanners
* Generalization to other MRI protocols

Because the validation split is slice-based, the reported SSIM may be optimistically biased.

### LPIPS

LPIPS has not yet been implemented or reported.

Future evaluation should specify:

* LPIPS package and version
* Backbone network
* Input intensity range
* Grayscale-to-RGB handling
* Slice-level and patient-level aggregation
* Mean and standard deviation

---

## Qualitative Results

Each sample image contains three panels from left to right:

| Position | Content                                                    |
| -------- | ---------------------------------------------------------- |
| Left     | Input MRI acquired using approximately 20% gadolinium dose |
| Center   | Real standard-dose MRI target                              |
| Right    | AI-generated standard-dose-like MRI                        |

### Sample 1

<p align="center">
  <img
    src="output/sample1.png"
    alt="Low-dose input, real standard-dose target, and generated output for sample 1"
    width="900"
  >
</p>

### Sample 2

<p align="center">
  <img
    src="output/sample2.png"
    alt="Low-dose input, real standard-dose target, and generated output for sample 2"
    width="900"
  >
</p>

### Sample 3

<p align="center">
  <img
    src="output/sample3.png"
    alt="Low-dose input, real standard-dose target, and generated output for sample 3"
    width="900"
  >
</p>

### Sample 4

<p align="center">
  <img
    src="output/sample4.png"
    alt="Low-dose input, real standard-dose target, and generated output for sample 4"
    width="900"
  >
</p>

These images are selected qualitative examples.

They should not be interpreted as a complete or unbiased representation of model performance.

---

## Failure Cases

A dedicated set of failure cases has not yet been included in this repository.

The current `output` folder contains four selected qualitative samples and does not provide a balanced analysis of poor-performing or clinically concerning outputs.

This is an important reporting limitation.

Future failure-case analysis should include examples with:

* Low SSIM
* Weak or missing enhancement
* Artificial enhancement
* Hallucinated structures
* Suppressed lesions
* Distorted tumor boundaries
* Misregistration
* Motion artifacts
* Low signal-to-noise ratio
* Anatomy near the superior and inferior ends of the brain
* Unusual pathology
* Out-of-distribution acquisition conditions

Each failure example should preferably contain:

```text
Low-dose input | Real standard-dose target | Generated output | Error map
```

Failure cases should be selected using predefined criteria rather than only visual preference.

---

## Repository Structure

```text
low-dose-mri-to-standard-dose-pix2pix/
├── low_dose_to_standard_dose_brain_mri_pix2pix.ipynb
├── output/
│   ├── sample1.png
│   ├── sample2.png
│   ├── sample3.png
│   └── sample4.png
├── README.md
└── .gitignore
```

The following files may be created during training but should generally not be committed directly when they are large:

```text
best_generator.pth
generator_final.pth
generator_epoch_10.pth
generator_epoch_20.pth
generator_epoch_30.pth
generator_epoch_40.pth
generator_epoch_50.pth
generator_epoch_60.pth

discriminator_epoch_10.pth
discriminator_epoch_20.pth
discriminator_epoch_30.pth
discriminator_epoch_40.pth
discriminator_epoch_50.pth
discriminator_epoch_60.pth
```

---

## Installation

### Requirements

* Python 3
* PyTorch
* torchvision
* NumPy
* Matplotlib
* Pillow
* scikit-image
* Jupyter Notebook or Google Colab
* CUDA-compatible GPU recommended

Install the required packages using:

```bash
pip install torch torchvision numpy matplotlib pillow scikit-image jupyter
```

For a reproducible research repository, package versions should be pinned in a `requirements.txt` file.

Example:

```text
torch==<version>
torchvision==<version>
numpy==<version>
matplotlib==<version>
Pillow==<version>
scikit-image==<version>
```

The exact versions used for the current experiment were not recorded.

---

## Running the Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/mohammadmolavi/low-dose-mri-to-standard-dose-pix2pix.git
cd low-dose-mri-to-standard-dose-pix2pix
```

### 2. Open the Notebook

Open:

```text
low_dose_to_standard_dose_brain_mri_pix2pix.ipynb
```

The notebook can be run using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 3. Prepare the Dataset

The expected dataset folders are:

```text
pix2pix_dataset/train/A
pix2pix_dataset/train/B
pix2pix_dataset/val/A
pix2pix_dataset/val/B
```

In the current Colab workflow, the dataset archive is loaded from:

```text
/content/drive/MyDrive/MRI/pix2pix_dataset.zip
```

Update this path when using another environment.

### 4. Execute the Notebook

Run the cells in order to:

1. Import dependencies
2. Mount Google Drive
3. Extract the dataset
4. Define the paired dataset loader
5. Resize and normalize the images
6. Create training and validation DataLoaders
7. Define the generator
8. Define the discriminator
9. Configure loss functions and optimizers
10. Train the model for 60 epochs
11. Save model checkpoints
12. Evaluate the final generator using SSIM
13. Export generated comparison images

---

## Model Checkpoints

During training, generator and discriminator checkpoints are saved every ten epochs.

The final generator is saved as:

```text
generator_final.pth
```

A generator checkpoint called:

```text
best_generator.pth
```

is also saved according to the final-batch generator loss at the end of each epoch.

Because this selection criterion does not use validation performance, `best_generator.pth` should not automatically be interpreted as the best generalizing model.

Large model files can be stored using:

* Git LFS
* Hugging Face Hub
* Zenodo
* A dedicated model-release page
* Institutional storage

--
---

## Limitations

The current project has several significant limitations.

### Dataset Limitations

* The dataset contains only eight patients
* The exact number of MRI series is not documented
* The number of slices per patient is not reported
* The acquisition sequence is not documented
* Scanner and protocol information are not documented
* There is no independent test set
* There is no external validation dataset

### Splitting Limitations

* The train-validation split is performed at the slice level
* Slices from the same patient may appear in both subsets
* Patient-level data leakage may inflate validation performance
* Generalization to unseen patients has not been demonstrated

### Pairing and Registration Limitations

* Registration is not performed in the notebook
* The original registration method is not documented
* Images are paired by patient identifier and sorted filename order
* Sequential filename pairing may fail if files are missing or incorrectly ordered
* Registration quality is not evaluated

### Image-Processing Limitations

* The original volumetric data are converted to 2D image files
* Images are converted to three-channel RGB
* Original physical voxel spacing is not used
* DICOM metadata are not used
* Images are resized to a fixed square resolution
* No brain mask is applied
* No intensity calibration is performed
* No data augmentation is used

### Model Limitations

* The generator is shallower than the standard full Pix2Pix U-Net
* The model operates independently on 2D slices
* Inter-slice and volumetric consistency are not modeled
* The model may hallucinate enhancement
* The model may suppress true enhancement
* The model may alter lesion boundaries
* The model may reproduce patient-specific patterns

### Reproducibility Limitations

* No random seed is set
* Package versions are not pinned
* Exact GPU hardware is not recorded
* Training duration is not reliably recorded
* Weight initialization is not explicitly controlled
* Best-checkpoint selection is based on a final training batch rather than validation performance

---

## Most Important Research Consideration

The most important research consideration is that the reported SSIM was obtained using a **slice-level validation split across data from only eight patients**.

Therefore, the current result primarily demonstrates that the model can learn the low-dose-to-standard-dose mapping within this limited dataset.

It does not demonstrate reliable generalization to a completely unseen patient.

The reported mean SSIM of `0.8615` should therefore be interpreted as a preliminary slice-level result rather than evidence of clinical validity.

The most important next methodological step is to perform patient-independent evaluation, preferably using:

* Leave-one-patient-out cross-validation
* Patient-wise cross-validation
* An independent patient-level test set
* External multi-center validation

Improving the evaluation design is currently more important than simply increasing model complexity.

---

## Medical and Ethical Considerations

This repository is intended exclusively for:

* Machine-learning research
* Medical image synthesis experiments
* Educational use
* Academic portfolio presentation
* Methodological investigation

### Clinical Disclaimer

The generated images:

* Are synthetic
* Are not real standard-dose MRI acquisitions
* May contain plausible but incorrect structures
* May hallucinate contrast enhancement
* May suppress clinically relevant enhancement
* May distort lesions or anatomical boundaries
* Have not been clinically validated
* Must not be used for diagnosis
* Must not be used for treatment planning
* Must not be used for patient management
* Must not replace standard MRI acquisition
* Must not replace interpretation by a qualified radiologist or physician

A visually realistic image is not necessarily medically accurate.

### Patient Privacy

Any medical data used in this project should be:

* Properly de-identified
* Stored securely
* Accessed only by authorized individuals
* Used according to applicable ethical approvals
* Used according to institutional policies
* Shared only when licensing and consent permit

No names, identifiers, dates, accession numbers, or embedded metadata containing protected health information should be published.

### Requirements Before Clinical Investigation

Any future clinical investigation would require:

* Larger and more diverse datasets
* Patient-independent testing
* External validation
* Multi-scanner evaluation
* Protocol robustness analysis
* Lesion-level evaluation
* Radiologist assessment
* Prospective validation
* Ethical approval
* Regulatory review
* Risk analysis for hallucination and omission errors

---

## Future Work

Future development should prioritize:

1. Patient-level train, validation, and test splits
2. Leave-one-patient-out cross-validation
3. Independent external validation
4. Documentation of MRI sequence and acquisition metadata
5. Documentation and verification of registration
6. Original DICOM or NIfTI processing
7. Grayscale or single-channel model training
8. Three-dimensional or multi-slice architectures
9. LPIPS evaluation
10. PSNR, MAE, and MSE evaluation
11. Per-patient metric reporting
12. Lesion-level and tumor-region analysis
13. Difference and error maps
14. Systematic failure-case reporting
15. Radiologist evaluation
16. Reproducible random seeds
17. Pinned software dependencies
18. Hardware and training-time logging
19. Validation-based checkpoint selection
20. Testing across scanners and acquisition protocols

---

## Reference

This project is based on the Pix2Pix conditional image-to-image translation framework:

> Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.
> **Image-to-Image Translation with Conditional Adversarial Networks.**
> IEEE Conference on Computer Vision and Pattern Recognition, 2017.

---

## Author

**Mohammad Molavi**

GitHub: [mohammadmolavi](https://github.com/mohammadmolavi)

---

## Final Disclaimer

This repository presents an experimental machine-learning pipeline.

The generated standard-dose-like MRI images are not clinically validated and must never be considered equivalent to real standard-dose contrast-enhanced MRI.

The output of this model must not replace:

* A physician
* A radiologist
* Standard imaging protocols
* Clinical judgment
* Diagnostic testing

