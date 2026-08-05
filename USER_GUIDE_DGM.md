This document provides a detailed usage guide for the Deep Generative Model (DGM) framework used for reconstructing quantum Wigner functions from experimentally accessible marginal distributions.

The repository includes:

- Wigner function dataset generation
- Dataset preprocessing and merging
- Deep generative model training
- Prediction and reconstruction
- Visualization and comparison tools

---
# Table of Contents

- [1. Overall Workflow](#1-overall-workflow)
- [2. Project Structure](#2-project-structure)
- [3. Dataset Generation](#3-dataset-generation)
- [4. Dataset Merging](#4-dataset-merging)
- [5. Model Training](#5-model-training)
- [6. Prediction and Reconstruction](#6-prediction-and-reconstruction)
- [7. Visualization and Comparison](#7-visualization-and-comparison)
- [8. Model Architecture](#8-model-architecture)
- [9. Important Parameters](#9-important-parameters)
- [10. Example Workflow](#10-example-workflow)
- [11. Output Files](#11-output-files)
- [12. Notes and Troubleshooting](#12-notes-and-troubleshooting)

---
# 1. Overall Workflow

The complete workflow of this project is:

```text
WF.py
   ↓
data_merge.py
   ↓
train.py
   ↓
prediction / visualization
```

Or directly execute:

```bash
python main_code.py
```

which automatically performs the complete workflow.

---
# 2. Project Structure

```text
─ DGM/
  │
  ├── code/
  │   ├── main_code.py
  │   ├── WF.py
  │   ├── train.py
  │   ├── data_merge.py
  │   ├── ResNet.py
  │   └── cmap.py
  │
  ├── npy_quantum/
  │   ├── distribution_JCM/
  │   └── distribution_TJCM/
  │
  └── test_visualization_Wigner/

```

---

# 3. Dataset Generation

Dataset generation is implemented in:

```text
WF.py
```

Main function:

```python
build_data(
    dataset_filename,
    types,
    num,
    resolution,
    length
)
```

---

## Supported Quantum Systems

Currently supported:

- MJCM (multiphoton Jaynes-Cummings model)
- TMJCM (two-atom extension)

---

## Example

```python
build_data(
    dataset_filename="npy_quantum",
    types="MJCM",
    num=30050,
    resolution=0.00625,
    length=7
)
```

---

## Parameters

| Parameter | Description |
|---|---|
| `dataset_filename` | Output dataset folder |
| `types` | Quantum model type (`MJCM` or `TMJCM`) |
| `num` | Number of generated samples |
| `resolution` | Phase-space resolution |
| `length` | Phase-space range |

---

## Generated Outputs

The program automatically generates:

```text
main_data/
x_data/
y_data/
u_data/
```

These correspond to:

| Folder | Description |
|---|---|
| `main_data` | Wigner function |
| `x_data` | x marginal |
| `y_data` | p marginal |
| `u_data` | u marginal |

---

# 4. Dataset Merging

Dataset merging is implemented in:

```text
data_merge.py
```

Main function:

```python
build_combined_data2(...)
```

This step:

- combines multiple datasets
- generates training/testing sets
- creates labels


---

## Example

```python
build_combined_data2(
    dataset_filename="npy_quantum",
    train_filename="dataset_quantum_ResNet184_test",
    JCM_num=30050,
    TJCM_num=30050,
    test_data_num=150,
    split_num=4,
    vmin=-0.45,
    vmax=0.45,
    channel=1
)
```
# 5. Model Training

Training is implemented in:

```text
train.py
```

Main function:

```python
training2(...)
```

---

## Example

```python
training2(
    dataset_filename="dataset_quantum_ResNet184_test",
    model_name="decoder_model_ResNet184_test",
    epochs=100,
    batch_size=32,
    data_patch=4
)
```

---

## Training Features

The framework supports:

- mixed precision training (AMP)
- automatic checkpoint saving
- resume training
- validation split
- automatic filtering of invalid samples
- best model saving

---

## Generated Model Files

Training automatically generates:

```text
decoder_model_xxx.py
decoder_model_xxx_best.pth
decoder_model_xxx_checkpoint.pth
```

---
# 6. Prediction and Reconstruction

Prediction is implemented using:

```python
prediction2(...)
```

This function loads a trained model and reconstructs Wigner functions from input marginals.

---

## Example

```python
prediction2(
    model_name="decoder_model_ResNet184_test",
    prediction_npy_name="prediction.npy",
    x_test=x_test,
    y_test=y_test
)
```

---

# 7. Visualization and Comparison

Visualization and reconstruction comparison are implemented using:

```python
test_comparison2(...)
```

---

## Example

```python
test_comparison2(
    train_filename="dataset_quantum_ResNet184_test",
    model_name="decoder_model_ResNet184_test",
    image_num=5,
    vmin=-0.15,
    vmax=0.15,
    point=2241,
    step=7,
    channel=1,
    a=2
)
```

---

## Generated Comparison Results

The comparison includes:

- Ground Truth Wigner function
- Predicted Wigner function
- Absolute difference
- x marginal comparison
- p marginal comparison
- u marginal comparison

---

## Dataset Selection Option

| `a` value | Description |
|---|---|
| `a = 0` | TMJCM test dataset |
| `a = 1` | MJCM test dataset |
| `a = 2` | all test dataset |
| `a = 3` | training dataset |

---

# 8. Model Architecture

The main architecture is implemented in:

```text
ResNet.py
```

Current architecture:

```text
ResNet184 Decoder
```

---

## Decoder Structure

```text
Latent Vector
    ↓
Fully Connected Layer
    ↓
8×8×2048 Feature Map
    ↓
Residual Deconvolution Stages
    ↓
256×256 Wigner Function
```

---

## Main Components

The architecture contains:

- Fully connected expansion layer
- Residual identity blocks
- Residual deconvolution blocks
- Batch normalization
- ReLU activation
- Final reconstruction head

---
---

# 9. Important Parameters

## Phase-Space Parameters

| Parameter | Meaning |
|---|---|
| `vmin` | minimum WF visualization value |
| `vmax` | maximum WF visualization value |
| `step` | phase-space range |
| `point` | marginal interpolation points |

---

## Training Parameters

| Parameter | Meaning |
|---|---|
| `epochs` | number of training epochs |
| `batch_size` | training batch size |
| `data_patch` | number of dataset partitions |

---
# 10. Example Workflow

## Full Example

```python
# Step 1. Generate dataset
build_data(...)

# Step 2. Merge dataset
build_combined_data2(...)

# Step 3. Train model
training2(...)

# Step 4. Compare reconstruction
test_comparison2(...)
```

---

## Direct Execution

You may directly run:

```bash
python main_code.py
```

which executes the entire workflow automatically.

---

# 11. Output Files

## Dataset Files

```text
x_train.npy
y_train.npy
x_test.npy
y_test.npy
```

---

## Model Files

```text
*_checkpoint.pth
*_best.pth
```

---

## Prediction Results

```text
prediction/
```

Contains:

- reconstructed images
- comparison plots
- saved prediction arrays

---

# Author


National Cheng Kung University (NCKU)  
Department of Engineering Science