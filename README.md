# Deep_Generative_Model_Reconstructing_WFs_Efficiently

This project presents a deep generative model (DGM) framework for efficient reconstruction of quantum Wigner functions from experimentally accessible marginal distributions.
The repository includes dataset generation, model training, testing, and visualization tools for MJCM and TMJCM quantum systems.

## Contents

- [Getting Started](#getting-started)
  - [Configuration Requirements](#configuration-requirements)
  - [Installation Steps](#installation-steps)
- [File Directory Description](#file-directory-description)

---

# Dataset

Download dataset from:
- MJCM
https://doi.org/10.5281/zenodo.20391988

- TMJCM

## Getting Started

### Recommended Environment
> **Note**
>
> Although the current system Python version is 3.13.7,  
> Python 3.10 is recommended for better compatibility with scientific computing libraries and PyTorch-related dependencies.

- OS: Ubuntu 22.04.5 LTS
- NVIDIA Driver: 580.126.09
- CUDA: 12.9
- cuDNN: 9.10.2
- Python: 3.13.7
- PyTorch: 2.8.0+cu129

---

### Installation Steps

#### 1. Check recommended NVIDIA drivers

```bash
ubuntu-drivers devices
```

#### 2. Install NVIDIA Driver 580

```bash
sudo apt update
sudo apt install nvidia-driver-580
```

#### 3. Reboot the system

```bash
sudo reboot
```

#### 4. Verify NVIDIA driver installation

```bash
nvidia-smi
```

If successful, the GPU model and driver version will be displayed.

---

#### 5. Install CUDA 13.0

Download CUDA 13.0 from the official NVIDIA CUDA Toolkit website.

---

#### 6. Run CUDA installer

Move to the download directory:

```bash
cd ~/Downloads
```

Run the installer:

```bash
sudo sh cuda_13.0.XXX_linux.run
```

---

#### 7. CUDA installation options

During the installation process:

- Select `accept`
- Deselect the NVIDIA Driver installation  
  (the driver has already been installed)
- Select `Install`

---

#### 8. Add CUDA environment variables

Open `.bashrc`:

```bash
sudo gedit ~/.bashrc
```

Add the following lines at the end of the file:

```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

Apply the changes:

```bash
source ~/.bashrc
```

---

#### 9. Verify CUDA installation

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery

sudo make

./deviceQuery
```

If successful, the CUDA device information will be displayed.
---

#### 10. Install cuDNN 9.10.2

Download cuDNN 9.10.2 compatible with CUDA 13.0 from the NVIDIA Developer website.

---

#### 11. Extract cuDNN package

Move to the download directory:

```bash
cd ~/Downloads
```

Extract the downloaded file:

```bash
tar -zxvf cudnn-linux-x86_64-9.10.2_cuda13-archive.tar.xz
```

---

#### 12. Copy cuDNN files to CUDA directory

```bash
sudo cp <extracted-folder>/lib/* /usr/local/cuda/lib64/

sudo cp <extracted-folder>/include/* /usr/local/cuda/include/

sudo chmod a+r /usr/local/cuda/include/cudnn.h \
/usr/local/cuda/lib64/libcudnn*
```

---

#### 13. Verify cuDNN installation

```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

If successful, the installed cuDNN version information will be displayed.

---

#### 14. Install Anaconda

Download the latest Anaconda installer from:

```text
https://www.anaconda.com/download
```

#### 15. Run Anaconda installer

Move to the download directory:

```bash
cd ~/Downloads
```

Run the installer:

```bash
sudo sh Anaconda3-5.3.1-Linux-x86_64.sh
```

---

#### 16. Installation options

During the installation process:

- Most options can be answered with `yes`
- Select `no` for Microsoft VSCode integration

---

#### 17. Check existing conda environments

```bash
conda env list
```

---

#### 18. Create a new virtual environment

```bash
conda create --name torch_gpu python=3.10
```

You may replace `torch_gpu` with your preferred environment name.

---

#### 19. Activate the virtual environment

```bash
conda activate torch_gpu
```

---

#### 20. Verify installed packages

```bash
conda list
```

#### 21. Install PyTorch

Install PyTorch with CUDA 12.9 support:

```bash
pip install torch torchvision torchaudio
```

---

#### 22. Verify PyTorch installation

```bash
python -c "import torch; print(torch.__version__)"

python -c "import torch; print(torch.cuda.is_available())"
```

If successful, the installed PyTorch version and CUDA availability will be displayed.
## File Directory Description
