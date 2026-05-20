import os, torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wigner_thermal import build_data3
from JCM2 import build_data
from new_data_merge2 import build_combined_data2
from train2_one import training2, prediction2, test_comparison2
from cmap import rgb_cmap, rgb_cmap_inverse

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #gpu 0 or 1

# ===== safe setting =====
os.environ["TORCH_COMPILE_DISABLE"] = "1"      
# ===================================

# environment variable
os.environ["TORCHINDUCTOR_CUDAGRAPH"] = "0"
os.environ["TORCHINDUCTOR_ASSERT_SIZE_STRIDE"] = "0"


#    auto_sample = auto combine all data (default = True)
#    test_data_num = numbers of test data (default = 100)
#    epochs = numbers of iteration (default = 100)
#    batch_size = numbers of data sampling per iteration (default = 32)
#    image_num = numbers of prediction image (default = 5) must < test_data_num/7
#    data_patch = number of divide data into training set (default = 4)
# =============================================================================


# =============================================================================
# build_data3(dataset_filename = "npy_quantum", types = 'harmonic', num =400,resolution = 0.00625, length = 7)
# build_data3(dataset_filename = "npy_quantum", types = 'coherent', num = 10000, resolution = 0.00625, length = 7)
# build_data3(dataset_filename = "npy_quantum", types = 'cat', num = 3000, resolution = 0.00625, length = 7)
# build_data3(dataset_filename = "npy_quantum", types = 'squeezed', num = 5000, resolution = 0.00625, length = 7)
# build_data3(dataset_filename = "npy_quantum", types = 'fock', num = 10000, resolution = 0.00625, length = 7)

# build_data3(dataset_filename = "npy_quantum", types = 'biphoton', num = 20050, resolution = 0.0125, length = 4.5)
# build_data(dataset_filename = "npy_quantum", types = 'Entanglement', num =1, resolution = 0.00625, length = 7)
# build_data(dataset_filename = "npy_quantum", types = 'TJCM', num = 100, resolution = 0.00625, length = 7)
# build_data(dataset_filename = "npy_quantum", types = 'JCM', num = 1, resolution = 0.00625, length = 7)

# build_combined_data2(dataset_filename = "npy_quantum", train_filename = "dataset_quantum_ResNet152_PCS_1", 
#                    harmonic_num =50,coherent_num = 0, cat_num = 0, squeezed_num = 0,fock_num = 0, biphoton_num = 0, Entanglement_num = 80050, TJCM_num = 0,JCM_num = 0,
#                     test_data_num = 150, split_num = 4,
#                     vmin = -0.45, vmax = 0.45, channel = 1) 

training2(dataset_filename = "dataset_quantum_ResNet152_2",  model_name = 'decoder_model_Densenet169_mix_2', epochs = 100, batch_size =64, data_patch = 4)

test_comparison2(train_filename = "dataset_quantum_ResNet152_2",  model_name = 'decoder_model_Densenet169_mix_2',
                   image_num = 5, vmin = -0.15, vmax = 0.15, point = 2241, step = 7, channel = 1, a = 1)
# =============================================================================


# Predict data (quantum data)--> two spectral density
# 1. Super-ohmics [1b+3]
# 2. Drude lorentz [1b+2]

# x_test need to be prepare into shape (None, 2163) 
# which included three marginal distribution
# each of marginal distribution has sampling the data 
# between -18 and 18 (x coordination) with interval 0.05
# one marginal shape = (None, 720) ==> three marginals shape = (None, 2163)
# And the prediction will be shape (None, 256, 256, 3)
# =============================================================================

# Super-ohmics 

# case 1
# labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
# types = ['T = 0.1', 'T = 0.2', 'T = 0.3', 'T = 0.4', 'T = 0.5']

# case 2
# labels = ['0.6', '0.7', '0.8', '0.9', '1.']
# types = ['T = 0.6', 'T = 0.7', 'T = 0.8', 'T = 0.9', 'T = 1.0']

# case 3
# labels = ['1.1', '1.2', '1.3', '1.4', '1.5']
# types = ['T = 1.1', 'T = 1.2', 'T = 1.3', 'T = 1.4', 'T = 1.5']

# case 4
# labels = ['1.6', '1.7', '1.8', '1.9', '2.']
# types = ['T = 1.6', 'T = 1.7', 'T = 1.8', 'T = 1.9', 'T = 2.0']

# case 5
# labels = ['2.1', '2.2', '2.3', '2.4', '2.5']
# types = ['T = 2.1', 'T = 2.2', 'T = 2.3', 'T = 2.4', 'T = 2.5']

# case 6
# labels = ['2.6', '2.7', '2.8', '2.9', '3.']
# types = ['T = 2.6', 'T = 2.7', 'T = 2.8', 'T = 2.9', 'T = 3.0']

# case 7
# labels = ['3.1', '3.2', '3.3', '3.4', '3.5']
# types = ['T = 3.1', 'T = 3.2', 'T = 3.3', 'T = 3.4', 'T = 3.5']

# case 8
# labels = ['3.6', '3.7', '3.8', '3.9', '4.']
# types = ['T = 3.6', 'T = 3.7', 'T = 3.8', 'T = 3.9', 'T = 4.0']

# case 9
# labels = ['4.1', '4.2', '4.3', '4.4', '4.5']
# types = ['T = 4.1', 'T = 4.2', 'T = 4.3', 'T = 4.4', 'T = 4.5']

# case 10
# labels = ['4.6', '4.7', '4.8', '4.9', '5.']
# types = ['T = 4.6', 'T = 4.7', 'T = 4.8', 'T = 4.9', 'T = 5.0']

# case 11
# labels = ['2.6', '2.9', '3.2', '3.6', '4.']
# types = ['T = 2.6', 'T = 2.9', 'T = 3.2', 'T = 3.6', 'T = 4.0']

# x_test = []
# for i in labels:
#     x_test_per = pd.read_csv(r'/home/quantum666/Desktop/super_ohmic/s=3.5/T=%s.csv'%i)
#     # x_test_per = pd.read_csv(r'C:\Users\Quantum\Documents\T=%s.csv'%i)
#     x_test.extend([x_test_per])
# x_test = np.array(x_test)
# print('x_test shape is', x_test.shape)

# prediction(model_name = 'decoder_model_shift_super_ohmics_one3', prediction_npy_name = 'prediction_SO.npy', x_test = x_test)
# result_comparison(density = 'SO', para = 3.5, types = types, prediction_npy_name = 'prediction_SO.npy', x_test = x_test,
#                     vmin = -0.01, vmax = 0.045)

# x_test = []
# for i in labels:
#     x_test_per = pd.read_csv(r'/home/quantum666/Desktop/super ohmic/s=2.5/T=%s.csv'%i)
#     # x_test_per = pd.read_csv(r'C:\Users\Quantum\Documents\T=%s.csv'%i)
#     x_test.extend([x_test_per])
# x_test = np.array(x_test)
# print('x_test shape is', x_test.shape)

# prediction(model_name = 'decoder_model_shift_super_ohmics_one5', prediction_npy_name = 'prediction_SO.npy', x_test = x_test)
# result_comparison(density = 'SO', para = 2.5, types = types, prediction_npy_name = 'prediction_SO.npy', x_test = x_test,
#                     vmin = -0.01, vmax = 0.045)

# =============================================================================
# Drude lorentz

# case 1
# labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
# types = ['T = 0.1', 'T = 0.2', 'T = 0.3', 'T = 0.4', 'T = 0.5']

# case 2
# labels = ['0.6', '0.7', '0.8', '0.9', '1.']
# types = ['T = 0.6', 'T = 0.7', 'T = 0.8', 'T = 0.9', 'T = 1.0']

# case 3
# labels = ['1.1', '1.2', '1.3', '1.4', '1.5']
# types = ['T = 1.1', 'T = 1.2', 'T = 1.3', 'T = 1.4', 'T = 1.5']

# case 4
# labels = ['1.6', '1.7', '1.8', '1.9', '2.']
# types = ['T = 1.6', 'T = 1.7', 'T = 1.8', 'T = 1.9', 'T = 2.0']

# case 5
# labels = ['2.1', '2.2', '2.3', '2.4', '2.5']
# types = ['T = 2.1', 'T = 2.2', 'T = 2.3', 'T = 2.4', 'T = 2.5']

# case 6
# labels = ['2.6', '2.7', '2.8', '2.9', '3.']
# types = ['T = 2.6', 'T = 2.7', 'T = 2.8', 'T = 2.9', 'T = 3.0']

# case 7
# labels = ['3.1', '3.2', '3.3', '3.4', '3.5']
# types = ['T = 3.1', 'T = 3.2', 'T = 3.3', 'T = 3.4', 'T = 3.5']

# case 8
# labels = ['3.6', '3.7', '3.8', '3.9', '4.']
# types = ['T = 3.6', 'T = 3.7', 'T = 3.8', 'T = 3.9', 'T = 4.0']

# case 9
# labels = ['4.1', '4.2', '4.3', '4.4', '4.5']
# types = ['T = 4.1', 'T = 4.2', 'T = 4.3', 'T = 4.4', 'T = 4.5']

# case 10
# labels = ['4.6', '4.7', '4.8', '4.9', '5.']
# types = ['T = 4.6', 'T = 4.7', 'T = 4.8', 'T = 4.9', 'T = 5.0']


# x_test = []
# for i in labels:
#     x_test_per = pd.read_csv(r'/home/quantum666/Desktop/drude lorentz/eta=1.5/T=%s.csv'%i)
#     # x_test_per = pd.read_csv(r'C:\Users\Quantum\Documents\T=%s.csv'%i)
#     x_test.extend([x_test_per])
# x_test = np.array(x_test)
# print(x_test.shape)

# prediction(model_name = 'decoder_model_shift_drude_lorentz_one1', prediction_npy_name = 'prediction_DL.npy', x_test = x_test)
# result_comparison(density = 'DL', para = 1.5, types = types, prediction_npy_name = 'prediction_DL.npy', x_test = x_test,
#                     vmin = -0.01, vmax = 0.045)

# =============================================================================

# x_train = []
# y_train = []
# osd = os.getcwd()
# front_path = os.path.abspath(os.path.join(osd, os.path.pardir))
# path_dataset = os.path.join(front_path, dataset_filename="dataset_shift_SO")
     
# for i in range(data_patch=4):
#     x_train += list(np.load(os.path.join(path_dataset, 'x_train%d.npy'%(i+1)), allow_pickle=True))
#     y_train += list(np.load(os.path.join(path_dataset, 'y_train%d.npy'%(i+1)), allow_pickle=True))
# print(x_train.shape)
# print(y_train.shape)



# train_filename = "dataset_shift_SO"
# front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
# path_combined_dataset = os.path.join(front_path, train_filename)

# x_test = np.load(os.path.join(path_combined_dataset, 'x_test.npy'))
# y_test = np.load(os.path.join(path_combined_dataset, 'y_test.npy'))
# print("max y test is ", np.max(y_test))
# print("min y test is ",np.min(y_test))
# print(y_test)



# others

# x_test = []
# for i in labels1:
#     x_test_per = pd.read_csv(r'C:\Users\quantum01\Desktop\Research Paper\result\quantum dynamics data\New_ohmic\T=%s.csv'%i)
#     #x_test_per = pd.read_csv(r'C:\Users\quantum01\Desktop\Research Paper\result\quantum dynamics data\new_data_4_7\s=2.0\T=%s.csv'%i)
#     #x_test_per = np.array(x_test_per.iloc[:2161, 0:1]).flatten().reshape((2160, 1))
#     x_test_per1 = np.array(np.append(np.array(x_test_per)[:720], 0))
#     x_test_per2 = np.array(np.append(np.array(x_test_per)[720:1440], 0))
#     x_test_per3 = np.array(np.append(np.array(x_test_per)[1440:], 0))
#     x_test_per_t = np.concatenate((x_test_per1, x_test_per2, x_test_per3), 0)
#     x_test.extend([x_test_per_t])  
# x_test = np.array(x_test)
# print(x_test.shape)
