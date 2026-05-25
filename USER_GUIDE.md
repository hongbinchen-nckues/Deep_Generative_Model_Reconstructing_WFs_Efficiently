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
project/
│
├── main_code.py
├── WF.py
├── train.py
├── data_merge.py
├── ResNet.py
├── cmap.py
│
├── npy_quantum/
│   ├── distribution_JCM/
│   └── distribution_TJCM/
│
├── dataset_quantum_ResNet184/
│
├── prediction/
│
├── README.md
└── USER_GUIDE.md
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
- TMJCM (wo-atom extension)

---

## Example

```python
build_data(
    dataset_filename="npy_quantum",
    types="TJCM",
    num=30050,
    resolution=0.00625,
    length=7
)
```

---
