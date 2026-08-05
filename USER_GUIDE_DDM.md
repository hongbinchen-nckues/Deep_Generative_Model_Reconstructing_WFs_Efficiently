This document provides a detailed usage guide for the Deep Diffusion Model (DGM) framework used for reconstructing quantum Wigner functions from experimentally accessible marginal distributions.

The repository includes:

- Deep diffusion model training
- Prediction and reconstruction

---
# Table of Contents

- [1. Overall Workflow](#1-overall-workflow)
- [2. Project Structure](#2-project-structure)
- [3. Data Generation and Merging](#3-data-generation-and-merging)
- [4. Model Training and Testing](#4-model-training-and-testing)

---
# 1. Overall Workflow

The complete workflow of this project is:

```text
WF.py (On DGM Folder)
   ↓
data_merge.py (On DGM Folder)
   ↓
Diffusion_Base.py
   ↓
prediction / visualization
```

---

# 2. Project Structure

```text
─ DDM/
  │
  ├── DiffusionGenerateImage/
  │
  ├── DiffusionModel/
  │
  ├── Code/
  │   ├── ClassicalModel.py
  │   ├── Function.py
  │   └── Load_State.py
  │
  ├── npy_quantum/
  │   ├── distribution_JCM/
  │   └── distribution_TJCM/
  │
  └── Diffusion_Base.py

```

---

# 3. Data Generation and Merging

See in DGM_user_guide.md.

---

# 4. Model Training and Testing

Run the code in Diffusion_Base.py to train the model.
You can check the test prediction result in DiffusionGenerateImage floder.