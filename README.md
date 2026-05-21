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


## File Directory Description
