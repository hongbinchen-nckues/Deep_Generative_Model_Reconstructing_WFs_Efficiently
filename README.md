# Deep_Generative_Model_Reconstructing_WFs_Efficiently

## Contents

- [Getting Started](#getting-started)
  - [Configuration Requirements](#configuration-requirements)
  - [Installation Steps](#installation-steps)
- [File Directory Description](#file-directory-description)

---

## Getting Started

### Configuration Requirements

- OS: Ubuntu 18.04
- NVIDIA Driver: 460
- CUDA: 11.0
- cuDNN: 8.0.4
- Python: 3.7
- TensorFlow: 2.4.0

---

### Installation Steps

#### 1. Check recommended NVIDIA drivers

```bash
ubuntu-drivers devices
```

#### 2. Install NVIDIA Driver 460

```bash
sudo apt update
sudo apt install nvidia-driver-460
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

#### 5. Install CUDA 11.0

Download CUDA 11.0 from the official NVIDIA CUDA Toolkit website.

---

#### 6. Run CUDA installer

Move to the download directory:

```bash
cd ~/Downloads
```

Run the installer:

```bash
sudo sh cuda_11.0.XXX_linux.run
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
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
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

#### 10. Install cuDNN 8.0.4

Download cuDNN 8.0.4 compatible with CUDA 11.0 from the NVIDIA Developer website.

---

#### 11. Extract cuDNN package

Move to the download directory:

```bash
cd ~/Downloads
```

Extract the downloaded file:

```bash
tar -zxvf cudnn-11.0-linux-x64-v8.0.4.30.tgz
```

---

#### 12. Copy cuDNN files to CUDA directory

```bash
sudo cp cuda/lib64/* /usr/local/cuda/lib64/

sudo cp cuda/include/* /usr/local/cuda/include/

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

#### 14. Install Anaconda (Python 3.7)

The recommended Anaconda version for this project is:

```text
Anaconda3-5.3.1-Linux-x86_64.sh
```

Download from the official Anaconda archive:

```text
https://repo.anaconda.com/archive/
```

---

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
conda create --name tf_gpu python=3.7
```

You may replace `tf_gpu` with your preferred environment name.

---

#### 19. Activate the virtual environment

```bash
source activate tf_gpu
```

---

#### 20. Verify installed packages

```bash
conda list
```

## File Directory Description
