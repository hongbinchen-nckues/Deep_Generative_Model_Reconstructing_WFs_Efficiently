import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import os
from cmap import rgb_cmap2
    
def build_combined_data2(dataset_filename, train_filename, TMJCM_num, MJCM_num, test_data_num = 150, split_num = 8, vmin = -0.01, vmax = 0.045, channel = 1):

    def read_type_data(distribution_num, type_filename, offset=0):
        x_train = list([])
        y_train1 = list([])
        y_train2 = list([])
        y_train3 = list([])

        available = distribution_num - offset
        per_split = available // split_num
        start = per_split * k + offset
        end   = per_split * (k + 1) + offset

        for i in tqdm(range(start, end), desc="Loading dataset"):
            m0 = np.load(os.path.join(osd, dataset_filename, type_filename, main_distribution_filename, '{}p0.npy').format(i), allow_pickle=True)
            m1 = np.load(os.path.join(osd, dataset_filename, type_filename, x_distribution_filename, '{}p1.npy').format(i), allow_pickle=True)
            m2 = np.load(os.path.join(osd, dataset_filename, type_filename, y_distribution_filename, '{}p2.npy').format(i), allow_pickle=True)
            m3 = np.load(os.path.join(osd, dataset_filename, type_filename, u_distribution_filename, '{}p3.npy').format(i), allow_pickle=True)
            if channel == 3:
                m0 = rgb_cmap2(m0, vmin, vmax)
            x_train.extend([m0])
            y_train1.extend([m1])
            y_train2.extend([m2])
            y_train3.extend([m3])

        x_train = np.array(x_train)
        y_train1 = np.array(y_train1)
        y_train2 = np.array(y_train2)
        y_train3 = np.array(y_train3)

        print('y_train_shape', y_train1.shape, y_train2.shape, y_train3.shape)
        y_train_per = np.concatenate((y_train1, y_train2, y_train3), axis = 1)

        return x_train, y_train_per
    

    # set load path of dataset
    osd = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))   
    TMJCM_filename ="full_distribution_TMJCM"
    MJCM_filename = "distribution_MJCM"

    
    main_distribution_filename = 'main_data'
    x_distribution_filename = 'x_data'
    y_distribution_filename = 'y_data'
    u_distribution_filename = 'u_data'
    
    # get the front path
    front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    
    path_combined_dataset = os.path.join(front_path, train_filename)
    
    print("Combined data save in {path_combined_dataset}".format(path_combined_dataset = path_combined_dataset))
    
    if os.path.exists(path_combined_dataset) == False:     
        os.mkdir(path_combined_dataset)

    print("TMJCM data number : ", TMJCM_num)
    print("MJCM data number : ", MJCM_num)


    # =========================
    # First, create a test set before training.
    # If both TMJCM and MJCM are used, split evenly (test_data_num // 2 each).
    # If only one type is used, take all test_data_num from that type.
    # =========================

    use_tmjcm = (TMJCM_num > 0)
    use_mjcm  = (MJCM_num  > 0)

    if use_tmjcm and use_mjcm:
        tmjcm_test_num = test_data_num // 2
        mjcm_test_num  = test_data_num - tmjcm_test_num
    elif use_tmjcm:
        tmjcm_test_num = test_data_num
        mjcm_test_num  = 0
    elif use_mjcm:
        tmjcm_test_num = 0
        mjcm_test_num  = test_data_num
    else:
        tmjcm_test_num = 0
        mjcm_test_num  = 0

    def read_test_data(index_list, type_filename):
        x_test = []
        y1, y2, y3 = [], [], []

        for i in index_list:
            m0 = np.load(os.path.join(osd, dataset_filename, type_filename, main_distribution_filename, '{}p0.npy').format(i), allow_pickle=True)
            m1 = np.load(os.path.join(osd, dataset_filename, type_filename, x_distribution_filename, '{}p1.npy').format(i), allow_pickle=True)
            m2 = np.load(os.path.join(osd, dataset_filename, type_filename, y_distribution_filename, '{}p2.npy').format(i), allow_pickle=True)
            m3 = np.load(os.path.join(osd, dataset_filename, type_filename, u_distribution_filename, '{}p3.npy').format(i), allow_pickle=True)

            if channel == 3:
                m0 = rgb_cmap2(m0, vmin, vmax)

            x_test.append(m0)
            y1.append(m1)
            y2.append(m2)
            y3.append(m3)

        x_test = np.array(x_test)
        y_test = np.concatenate((np.array(y1), np.array(y2), np.array(y3)), axis=1)

        return x_test, y_test

    x_test_parts, y_test_parts, label_list = [], [], []

    if tmjcm_test_num > 0:
        tmjcm_idx = np.arange(0, tmjcm_test_num)
        tmjcm_x_test, tmjcm_y_test = read_test_data(tmjcm_idx, TMJCM_filename)
        x_test_parts.append(tmjcm_x_test)
        y_test_parts.append(tmjcm_y_test)
        label_list += [0] * tmjcm_test_num

    if mjcm_test_num > 0:
        mjcm_idx = np.arange(0, mjcm_test_num)
        mjcm_x_test, mjcm_y_test = read_test_data(mjcm_idx, MJCM_filename)
        x_test_parts.append(mjcm_x_test)
        y_test_parts.append(mjcm_y_test)
        label_list += [1] * mjcm_test_num

    if len(x_test_parts) > 0:
        x_test = np.concatenate(x_test_parts, axis=0)
        y_test = np.concatenate(y_test_parts, axis=0)

        test_label = np.array(label_list)

        x_test, y_test, test_label = shuffle(x_test, y_test, test_label)

        if channel == 1:
            print("Before reshape, x_test shape:", x_test.shape)
            x_test = x_test.reshape(-1, 256, 256, 1)

        np.save(os.path.join(path_combined_dataset, 'x_test.npy'), y_test)
        np.save(os.path.join(path_combined_dataset, 'y_test.npy'), x_test)
        np.save(os.path.join(path_combined_dataset, 'test_label.npy'), test_label)

        print("MJCM/TMJCM test set done")
    else:
        print("Skip MJCM/TMJCM test set because TMJCM_num = 0 and MJCM_num = 0")

    
    x_data_num = 0
    for k in range(split_num):

# =============================================================================
#         1. choose states
# =============================================================================

        if TMJCM_num == 0 :
            xn1_train, yn1_train = read_type_data(MJCM_num, MJCM_filename,  offset=mjcm_test_num)

            x_train = xn1_train
            y_train = yn1_train
        else:
            xn1_train, yn1_train = read_type_data(MJCM_num, MJCM_filename, offset=mjcm_test_num)
            xn2_train, yn2_train = read_type_data(TMJCM_num, TMJCM_filename, offset=tmjcm_test_num)            

            x_train = np.concatenate((xn1_train, xn2_train), axis=0)
            y_train = np.concatenate((yn1_train, yn2_train), axis=0)


        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')

        X_train, Y_train = shuffle(x_train, y_train)
        
        del x_train, y_train
        
        x_train = X_train
        y_train = Y_train
                  
        
        np.save(os.path.join(path_combined_dataset, 'x_train%d.npy'%(k+1)), y_train)
        np.save(os.path.join(path_combined_dataset, 'y_train%d.npy'%(k+1)), x_train)
        print(X_train.shape,Y_train.shape)
        x_data_num += y_train.shape[0]



    if len(x_test_parts) > 0:
        print('y_test_shape = ', y_test.shape)
        print('x_test_shape = ', x_test.shape)
        print('test_data_num = ', x_test.shape[0])
    else:
        print('y_test_shape = skipped')
        print('x_test_shape = skipped')
        print('test_data_num = 0')
        
    print('x_data_shape = ', y_train.shape)
    print('y_data_shape = ', x_train.shape)
    print('train_data_num = ', x_data_num)

