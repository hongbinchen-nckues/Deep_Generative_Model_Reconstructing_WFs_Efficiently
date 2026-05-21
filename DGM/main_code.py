import os, torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from JCM2 import build_data
from data_merge import build_combined_data2
from train2_one import training2, prediction2, test_comparison2
from cmap import rgb_cmap, rgb_cmap_inverse

# ===== safe setting =====
os.environ["TORCH_COMPILE_DISABLE"] = "1"      
# environment variable
os.environ["TORCHINDUCTOR_CUDAGRAPH"] = "0"
os.environ["TORCHINDUCTOR_ASSERT_SIZE_STRIDE"] = "0"

# =============================================================================
#    auto_sample = auto combine all data (default = True)
#    test_data_num = numbers of test data (default = 100)
#    epochs = numbers of iteration (default = 100)
#    batch_size = numbers of data sampling per iteration (default = 32)
#    image_num = numbers of prediction image (default = 5) must < test_data_num/7
#    data_patch = number of divide data into training set (default = 4)
# =============================================================================
# build_data(dataset_filename = "npy_quantum", types = 'TJCM', num = 100, resolution = 0.00625, length = 7)
# build_data(dataset_filename = "npy_quantum", types = 'JCM', num = 1, resolution = 0.00625, length = 7)

build_combined_data2(dataset_filename = "npy_quantum", train_filename = "dataset_quantum_ResNet152_2", 
                     JCM_num = 80050, TJCM_num = 80050,
                     test_data_num = 150, split_num = 4, vmin = -0.45, vmax = 0.45, channel = 1) 

training2(dataset_filename = "dataset_quantum_ResNet152_2",  model_name = 'decoder_model_ResNet152_nmix_2', epochs = 100, batch_size =64, data_patch = 4)

test_comparison2(train_filename = "dataset_quantum_ResNet152_2",  model_name = 'decoder_model_ResNet152_nmix_2',
                   image_num = 5, vmin = -0.15, vmax = 0.15, point = 2241, step = 7, channel = 1, a = 2)
