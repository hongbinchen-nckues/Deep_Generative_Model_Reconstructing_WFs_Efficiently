import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import os
from cmap import rgb_cmap2
from scipy.ndimage import zoom
import cv2
    
def build_combined_data2(dataset_filename, train_filename, harmonic_num, coherent_num, cat_num, squeezed_num, fock_num, biphoton_num, Entanglement_num, TJCM_num, JCM_num, test_data_num = 150, split_num = 8, vmin = -0.01, vmax = 0.045, channel = 1):
    # reading definition
    # offset: number of files reserved for test at the start of the index range.
    #   - TJCM / JCM : offset = test_each  (test indices 0 .. test_each-1 are held out)
    #   - harmonic / others : offset = 0   (no test files reserved for these types)
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
    harmonic_filename = "distribution_harmonic"
    coherent_filename = "distribution_coherent"
    cat_filename = "distribution_cat"
    squeezed_filename = "distribution_squeezed"
    fock_filename = "distribution_fock"
    biphoton_filename = "distribution_biphoton"
    Entanglement_filename = "distribution_Entanglement"
    TJCM_filename ="full_distribution_TJCM"
    JCM_filename = "distribution_JCM"

    
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

    print("harmoic data number : ", harmonic_num)
    print("coherent data number : ", coherent_num)
    print("cat data number : ", cat_num)
    print("squeezed data number : ", squeezed_num)
    print("fock data number : ", fock_num)
    print("biphoton data number : ", biphoton_num)
    print("Entanglement data number : ", Entanglement_num)
    print("TJCM data number : ", TJCM_num)
    print("JCM data number : ", JCM_num)


    # =========================
    # First, create a test set before training.
    # If both TJCM and JCM are used, split evenly (test_data_num // 2 each).
    # If only one type is used, take all test_data_num from that type.
    # =========================

    use_tjcm = (TJCM_num > 0)
    use_jcm  = (JCM_num  > 0)

    if use_tjcm and use_jcm:
        tjcm_test_num = test_data_num // 2
        jcm_test_num  = test_data_num - tjcm_test_num
    elif use_tjcm:
        tjcm_test_num = test_data_num
        jcm_test_num  = 0
    elif use_jcm:
        tjcm_test_num = 0
        jcm_test_num  = test_data_num
    else:
        tjcm_test_num = 0
        jcm_test_num  = 0

    # test_each is used as training-index offset for TJCM to avoid test/train overlap
    test_each = tjcm_test_num

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

    if tjcm_test_num > 0:
        tjcm_idx = np.arange(0, tjcm_test_num)
        tjcm_x_test, tjcm_y_test = read_test_data(tjcm_idx, TJCM_filename)
        x_test_parts.append(tjcm_x_test)
        y_test_parts.append(tjcm_y_test)
        label_list += [0] * tjcm_test_num

    if jcm_test_num > 0:
        jcm_idx = np.arange(0, jcm_test_num)
        jcm_x_test, jcm_y_test = read_test_data(jcm_idx, JCM_filename)
        x_test_parts.append(jcm_x_test)
        y_test_parts.append(jcm_y_test)
        label_list += [1] * jcm_test_num

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
        print("Skip MJCM/TMJCM test set because TJCM_num = 0 and JCM_num = 0")

    
    x_data_num = 0
    for k in range(split_num):
        
        # four training case
        # 1. pure gauss 
        # 2. pure gauss added noise
        # 3. shifted pure gauss
        # 4. shifted pure gauss added noise(limited)

# =============================================================================
#         1. choose states
# =============================================================================

        if harmonic_num == 0 :
            # xn1_train, yn1_train = read_type_data(coherent_num, coherent_filename, offset=0)
            # xn2_train, yn2_train = read_type_data(cat_num, cat_filename, offset=0)
            # xn3_train, yn3_train = read_type_data(squeezed_num, squeezed_filename, offset=0)
            # xn5_train, yn5_train = read_type_data(biphoton_num, biphoton_filename, offset=0)
            # xn6_train, yn6_train = read_type_data(Entanglement_num, Entanglement_filename, offset=0)
            xn7_train, yn7_train = read_type_data(TJCM_num, TJCM_filename, offset=test_each)
            # xn8_train, yn8_train = read_type_data(JCM_num, JCM_filename, offset=test_each)

            x_train = xn7_train
            y_train = yn7_train
        else:
            x_train, y_train = read_type_data(harmonic_num, harmonic_filename, offset=0)
            # xn1_train, yn1_train = read_type_data(coherent_num, coherent_filename, offset=0)
            # xn2_train, yn2_train = read_type_data(cat_num, cat_filename, offset=0)
            # xn3_train, yn3_train = read_type_data(squeezed_num, squeezed_filename, offset=0)
            # xn4_train, yn4_train = read_type_data(fock_num, fock_filename, offset=0)
            # xn5_train, yn5_train = read_type_data(biphoton_num, biphoton_filename, offset=0)
            # xn6_train, yn6_train = read_type_data(Entanglement_num, Entanglement_filename, offset=0)
            xn7_train, yn7_train = read_type_data(TJCM_num, TJCM_filename, offset=test_each)
            xn8_train, yn8_train = read_type_data(JCM_num, JCM_filename, offset=test_each)

            x_train = np.concatenate((xn7_train, xn8_train, x_train), axis=0)
            y_train = np.concatenate((yn7_train, yn8_train, y_train), axis=0)


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

