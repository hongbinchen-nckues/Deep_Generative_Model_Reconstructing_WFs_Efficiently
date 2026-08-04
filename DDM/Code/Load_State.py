import os
import numpy as np



def LoadWignerDataset(type_folder, types = [], num_of_data = -1):
    '''
    type_folder: str, dataset's folder
    type: list[str], what dataset you want in this folder, ex: ["cat"]
    num_of_data: None = all datas

    reture: dict[type]
    {
     type[0]:{"2d_data":list[float](256,256), 
              "1d_data":list[float](3,721)}, 
     type[1]: ...
    }
    '''
    k = {}
    if len(types) >= 0:
        for type in types:
            print(type_folder, type)

            main_data_files = os.listdir(os.path.join(type_folder, f"distribution_{type}", "main_data"))
            x_data_files = os.listdir(os.path.join(type_folder, f"distribution_{type}", "x_data"))
            y_data_files = os.listdir(os.path.join(type_folder, f"distribution_{type}", "y_data"))
            u_data_files = os.listdir(os.path.join(type_folder, f"distribution_{type}", "u_data"))


            omega = [main_data_files, x_data_files, y_data_files, u_data_files]

            safe_test = {}
            for data_files in omega:
                for it in data_files:
                    if ".npy" in it:
                        number = it.split("p", 1)[0]
                        if safe_test.get(number, 0) == 0:
                            safe_test[number] = 1
                        else:
                            safe_test[number] += 1

            main_data = []
            x_data = []
            y_data = []
            u_data = []

            it_list = list(safe_test.keys())
            it_list = sorted(it_list)
            if num_of_data == -1:
                num_of_data = len(it_list)
            
            get = 0
            for it in it_list:
                if get == num_of_data:
                    break
                if safe_test[it] == 4:
                    main_data.append(np.load(os.path.join(type_folder, f"distribution_{type}", "main_data", f"{it}p0.npy")))
                    x_data.append(np.load(os.path.join(type_folder, f"distribution_{type}", "x_data", f"{it}p1.npy")))
                    y_data.append(np.load(os.path.join(type_folder, f"distribution_{type}", "y_data", f"{it}p2.npy")))
                    u_data.append(np.load(os.path.join(type_folder, f"distribution_{type}", "u_data", f"{it}p3.npy")))
                get += 1

            main_data = np.stack(main_data)
            x_data = np.stack(x_data)
            y_data = np.stack(y_data)
            u_data = np.stack(u_data)
            if "withNCA" in type:
                data_list = np.stack([x_data, y_data, u_data[:,:-1]], axis=1)
                nc = u_data[:,-1]
                u = {"2d_data":main_data, "1d_data":np.stack(data_list, axis=0), "nonC":nc}
            else:
                data_list = np.stack([x_data, y_data, u_data], axis=1)
                u = {"2d_data":main_data, "1d_data":np.stack(data_list, axis=0)}
            k[type] = u
    return k

def DataLoad(side_dim, num_of_data, CircultType, Q_type = ["cat"], image_label = "01", point = 721):
    """
        side_dim -> loaded image size h and w (h=w)
        num_of_data -> int or None(all data)
        CircultType -> want type of dataset you want
        Q_type -> list
        image_label -> string (want number you want, e.g.: "01")
        num_of_data -> number of data you want, -1 = all of datas
        num_frequency_encoding -> 

        return train_data, train_cond, test_data, test_cond

        all data return device are cpu, so you need .to(device) if you want to run in diffenent device
        CircultType == "MI_MNIST": -> Return MNIST dataset
            side_dim should be 8 or 16
            Q_type is not required
            image_label is required if u want to change number
        CircultType == "MI_P3C": -> Return Simulated quasi dataset
            side_dim should be 22
            Q_type is not required
        CircultType == "MI_W3C": -> Return Wigner dataset
            side_dim should be 256
            choose Q_type data you want: include ["cat", "coherent", "harmonic", "squeezed"]
    """
    import json
    import torch
    import Code.Function as MF

    if "MI_MNIST" in CircultType:
        train_raw, test_raw = {}, {}
        with open(f"./Data/Train_Data_{side_dim}x{side_dim}.json", mode = "r", encoding="utf-8") as json_f:
            train_raw = json.load(json_f)
        with open(f"./Data/Test_Data_{side_dim}x{side_dim}.json", mode = "r", encoding="utf-8") as json_f:
            test_raw = json.load(json_f)

        if num_of_data == None:
            train_data = (torch.cat([torch.tensor(train_raw[i]) for i in image_label], dim=0) / 255)
            train_data = train_data.view(len(train_data), 1, side_dim,side_dim)
            train_cond = MF.MNISTToMarginalDistribution(train_data).view(len(train_data), 3*side_dim)
        else:
            num_of_data = (num_of_data // len(image_label)) * len(image_label)
            train_data = (torch.cat([torch.tensor(train_raw[i][0:num_of_data//len(image_label)]) for i in image_label], dim=0) / 255)
            train_data = train_data.view(len(train_data), 1, side_dim,side_dim)
            train_cond = MF.MNISTToMarginalDistribution(train_data).view(len(train_data), 3*side_dim)

        test_data = (torch.cat([torch.tensor(test_raw[i]) for i in image_label], dim=0) / 255)
        test_data = test_data.view(len(test_data), 1, side_dim,side_dim)
        test_cond = MF.MNISTToMarginalDistribution(test_data).view(len(test_data), 3*side_dim)

        train_data = (train_data * 2 - 1)
        test_data = (test_data * 2 - 1)

    elif "MNIST" in CircultType:
        # This will only return MNIST data along with its tags.
        train_raw, test_raw = {}, {}
        with open(f"./Data/Train_Data_{side_dim}x{side_dim}.json", mode = "r", encoding="utf-8") as json_f:
            train_raw = json.load(json_f)
        with open(f"./Data/Test_Data_{side_dim}x{side_dim}.json", mode = "r", encoding="utf-8") as json_f:
            test_raw = json.load(json_f)

        if num_of_data != None:
            num_of_data = (num_of_data // len(image_label)) * len(image_label)
            train_data = (torch.cat([torch.tensor(train_raw[it][0:num_of_data//len(image_label)]) for it in image_label], dim=0) / 255)
        else: 
            train_data = (torch.cat([torch.tensor(train_raw[it]) for it in image_label], dim=0) / 255)
        train_data = train_data.view(len(train_data), 1, side_dim,side_dim)
        train_cond = torch.cat([torch.full([len(train_raw[it])], fill_value = i) for i, it in enumerate(image_label)])

        test_data = (torch.cat([torch.tensor(test_raw[it]) for it in image_label], dim=0) / 255)
        test_data = test_data.view(len(test_data), 1, side_dim,side_dim)
        test_cond = torch.cat([torch.full([len(test_raw[it])], fill_value = i) for i, it in enumerate(image_label)])

        train_data = (train_data * 2 - 1)
        test_data = (test_data * 2 - 1)

    elif "MI_P3C" in CircultType:
        file_path = f"./Data/GenData/{side_dim}_{point}"
        if os.path.exists(file_path) == False:
            GeneratePData(file_path, side_dim, point)
        data = torch.tensor(np.load(os.path.join(file_path, "main_data.npy")))
        x = torch.tensor(np.load(os.path.join(file_path, "x_data.npy")))
        y = torch.tensor(np.load(os.path.join(file_path, "y_data.npy")))
        u = torch.tensor(np.load(os.path.join(file_path, "u_data.npy")))

        train_data = data.unsqueeze(1)
        train_cond = torch.stack([x,y,u], dim=1)

        spp = int(len(train_data) * 0.75)
        perm_in = torch.randperm(len(train_data))
        test_data = train_data[perm_in[spp:]]
        test_cond = train_cond[perm_in[spp:]]
        train_data = train_data[perm_in[0:spp]]
        train_cond = train_cond[perm_in[0:spp]]


        import matplotlib.pyplot as plt
        from Code.Function import rgb_cmap
        show_image = rgb_cmap(train_data[0:8, 0].numpy() ,-1.05, 1.05)
        fig, ax = plt.subplots(4, 8, figsize = (8*2, 4*2))
        for i in range(8):
            ax[0,i].imshow(np.clip(show_image[i], 0, 1))
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
        for j in range(3):
            for i in range(8):
                ax[1+j,i].plot([i for i in range(point)], train_cond[i][j].numpy(), c="black")
                ax[1+j,i].set_xticks([0,point-1])
        plt.tight_layout()


        # save figure
        plt.savefig(os.path.join(file_path, "Show.png"))
        plt.close()
        
    elif CircultType == "MI_W3C":
        type_folder = "npy_quantum"
        datas = LoadWignerDataset(type_folder, types = Q_type, num_of_data = num_of_data)

        train_data = []
        train_cond = []
        test_data = []
        test_cond = []

        train_nc = []
        test_nc = []
        if "AllCM" in Q_type:
            print("Load JCM and TJCM dataset")
            for i in range(4):
                cond = torch.tensor(np.load(os.path.join(type_folder, f"distribution_AllCM", f"x_train{i+1}.npy")), dtype = torch.float32)
                data = torch.tensor(np.load(os.path.join(type_folder, f"distribution_AllCM", f"y_train{i+1}.npy")), dtype = torch.float32).squeeze().unsqueeze(1)
                train_cond.append(cond.view(len(cond), 3, point))
                train_data.append(data)

            cond = torch.tensor(np.load(os.path.join(type_folder, f"distribution_AllCM", f"x_test.npy")), dtype = torch.float32)
            data = torch.tensor(np.load(os.path.join(type_folder, f"distribution_AllCM", f"y_test.npy")), dtype = torch.float32).squeeze().unsqueeze(1)
            test_cond.append(cond.view(len(cond), 3, point))
            test_data.append(data)
        else:
            for data in datas.keys():
                get_2d_data = torch.tensor(datas[data]['2d_data']).unsqueeze(1).to(dtype = torch.float32)
                get_1d_data = torch.tensor(datas[data]['1d_data']).to(dtype = torch.float32)
                
                print(get_2d_data.shape)
                print(get_1d_data.shape)
                if side_dim != 256:
                    from torch.nn import functional as F
                    get_2d_data = F.interpolate(get_2d_data, size=(side_dim,side_dim), mode='bilinear', align_corners=False)
                get_1d_data = torch.tensor(get_1d_data, dtype = torch.float32)

                torch.manual_seed(42)
                perm_in = torch.randperm(len(get_2d_data))
                perm_in = [i for i in range(len(get_2d_data))]
                num_test = 200
                test_data.append(get_2d_data[perm_in[0:num_test]])
                test_cond.append(get_1d_data[perm_in[0:num_test]])
                train_data.append(get_2d_data[perm_in[num_test:]])
                train_cond.append(get_1d_data[perm_in[num_test:]])
                
                if "withNCA" in Q_type[0]:
                    get_nonC_data = torch.tensor(datas[data]['nonC']).to(dtype = torch.float32)
                    test_nc.append(get_nonC_data[perm_in[0:num_test]])
                    train_nc.append(get_nonC_data[perm_in[num_test:]])

        train_data = torch.concat(train_data, dim=0)
        train_cond = torch.concat(train_cond, dim=0)
        test_data = torch.concat(test_data, dim=0)
        test_cond = torch.concat(test_cond, dim=0)
        if "withNCA" in Q_type[0]:
            test_nc = torch.concat(test_nc, dim=0)
            train_nc = torch.concat(train_nc, dim=0)
            print(f"train_data.shape:{train_data.shape}, train_cond:{train_cond.shape}")
            print(f"test_data.shape:{test_data.shape}, test_cond:{test_cond.shape}")
            return train_data, train_cond, test_data, test_cond, train_nc, test_nc


    elif CircultType == "MI_W3C_Old":
        type_folder = "npy_quantum"
        datas = LoadWignerDataset(type_folder, types = Q_type, num_of_data = num_of_data)
        train_data = []     # ground_true
        train_cond = []
        for data in datas.keys():
            get_data = train_data.append(datas[data]['2d_data']) 
            train_cond.append(datas[data]['1d_data'])


        train_data = torch.tensor(np.concat(train_data, axis=0), dtype = torch.float32)
        train_data = train_data.unsqueeze(1)
        if side_dim != 256:
            from torch.nn import functional as F
            train_data = F.interpolate(train_data, size=(side_dim,side_dim), mode='bilinear', align_corners=False)
        train_cond = torch.tensor(np.concat(train_cond, axis=0), dtype = torch.float32)

        spp = int(len(train_data) * 0.75)
        perm_in = torch.randperm(len(train_data))
        test_data = train_data[perm_in[spp:]]
        test_cond = train_cond[perm_in[spp:]]
        train_data = train_data[perm_in[0:spp]]
        train_cond = train_cond[perm_in[0:spp]]

    elif "MI_M3C" in CircultType:
        file_path = f"./Data/GenData/{side_dim}_{point}"
        if os.path.exists(file_path) == False:
            GeneratePData(file_path, side_dim, point)
        data = torch.tensor(np.load(os.path.join(file_path, "main_data.npy")))
        x = torch.tensor(np.load(os.path.join(file_path, "x_data.npy")), dtype=torch.float32)
        y = torch.tensor(np.load(os.path.join(file_path, "y_data.npy")), dtype=torch.float32)
        u = torch.tensor(np.load(os.path.join(file_path, "u_data.npy")), dtype=torch.float32)

        train_data = data.unsqueeze(1)
        train_cond = torch.stack([x,y,u], dim=1)

        spp = int(len(train_data) * 0.75)
        perm_in = torch.randperm(len(train_data))
        test_data = train_data[perm_in[spp:]]
        test_cond = train_cond[perm_in[spp:]]
        train_data = train_data[perm_in[0:spp]]
        train_cond = train_cond[perm_in[0:spp]]


        import matplotlib.pyplot as plt
        from Code.Function import rgb_cmap
        show_image = rgb_cmap(train_data[0:8, 0].numpy() ,-1.05, 1.05)
        fig, ax = plt.subplots(4, 8, figsize = (8*2, 4*2))
        for i in range(8):
            ax[0,i].imshow(np.clip(show_image[i], 0, 1))
            ax[0,i].set_xticks([])
            ax[0,i].set_yticks([])
        for j in range(3):
            for i in range(8):
                ax[1+j,i].plot([i for i in range(point)], train_cond[i][j].numpy(), c="black")
                ax[1+j,i].set_xticks([0,point-1])
        plt.tight_layout()


        # save figure
        plt.savefig(os.path.join(file_path, "Show.png"))
        plt.close()

    print(f"train_data.shape:{train_data.shape}, train_cond:{train_cond.shape}")
    print(f"test_data.shape:{test_data.shape}, test_cond:{test_cond.shape}")
    return train_data, train_cond, test_data, test_cond

def GeneratePData(folder_path, side_dim, point):
    from tqdm import tqdm
    if os.path.exists(folder_path) == False:
        os.mkdir(folder_path)
    main_data_file = os.path.join(folder_path, "main_data") # 10000 x side_dim x side_dim
    x_data_file = os.path.join(folder_path, "x_data")       # 10000 x point
    y_data_file = os.path.join(folder_path, "y_data")       # 10000 x point
    u_data_file = os.path.join(folder_path, "u_data")       # 10000 x point
    main_data_list = []
    x_data_list = []
    y_data_list = []
    u_data_list = []

    data = []
    def GetRandomPX1X13(grid, point):
        sigma_1 = np.random.uniform(0.15,0.3)     
        sigma_13 = np.random.uniform(0.15,0.3) 
        mu_1 = np.random.uniform(-0.4,0.4) 
        mu_13 = np.random.uniform(-0.4,0.4) 
        c = np.random.uniform(-0.8,0.8)  

        # print(f"sigma_1: {sigma_1}, sigma_13: {sigma_13}")
        # print(f"mu_1: {mu_1}, mu_13: {mu_13}, c: {c}")

        d_1 = (grid[:,:,0] - mu_1)**2 / (sigma_1**2)
        d_2 = c * (2 * (grid[:,:,0] - mu_1) * (grid[:,:,1] - mu_13)) / (sigma_1 * sigma_13)
        d_3 = (grid[:,:,1] - mu_13)**2 / (sigma_13**2)
        d = (-1 / (2*(1-c**2))) * (d_1 - d_2 + d_3)
        p = 1 / (2 * np.pi * sigma_1 * sigma_13 * np.sqrt(1-c**2)) * np.exp(d)

        k = np.linspace(-1.05, 1.05, point)
        x = 1 / (np.sqrt(2 * np.pi) * sigma_1) * np.exp(-(k - mu_1)**2 / (sigma_1**2) / 2)
        y = 1 / (np.sqrt(2 * np.pi) * sigma_13) * np.exp(-(k - mu_13)**2 / (sigma_13**2) / 2)

        s = sigma_1**2 + 2*c*sigma_1*sigma_13 + sigma_13**2
        u = 1 / (np.sqrt(np.pi * s)) * np.exp( - ((2*k - mu_1 - mu_13) / np.sqrt(2))**2 / s )
        return p, x, y, u

    np.random.seed(1234)
    for i in tqdm(range(10000)):
        k = np.linspace(-1.05, 1.05, side_dim)
        x,y = np.meshgrid(k,k)
        grid = np.stack([x,y],axis=-1)

        p1, x1, y1, u1 = GetRandomPX1X13(grid, point = point)
        p2, x2, y2, u2 = GetRandomPX1X13(grid, point = point)
        p3, x3, y3, u3 = GetRandomPX1X13(grid, point = point)

        A = np.random.uniform(-1.15, 1.15)
        main_data = (p1 + A*p2 - A*p3) / 10
        x_data = (x1 + A*x2 - A*x3) / 10
        y_data = (y1 + A*y2 - A*y3) / 10
        u_data = (u1 + A*u2 - A*u3) / 10
        main_data_list.append(main_data)
        x_data_list.append(x_data)
        y_data_list.append(y_data)
        u_data_list.append(u_data)

        show = False
        if show == True:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import sys
            sys.path.append('')
            import Code.Function as MF
            norm = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)

            print("main_data: ", main_data.min(), main_data.max())
            rgb_main_data = MF.rgb_cmap(main_data ,-1.05, 1.05)

            fig, ax = plt.subplots(1,4, figsize = (12, 3))
            ax[0].imshow(rgb_main_data)
            ax[1].plot([i for i in range(len(x_data))], x_data)
            ax[2].plot([i for i in range(len(y_data))], y_data)
            ax[3].plot([i for i in range(len(u_data))], u_data)
            plt.tight_layout()

            three_axis_pred = MF.Back2MDx3(main_data[None, ...], point=point)
            xx = 0.15
            ax[1].plot([i for i in range(len(x_data))], three_axis_pred[0][0] * xx)
            ax[2].plot([i for i in range(len(y_data))], three_axis_pred[0][1] * xx)
            ax[3].plot([i for i in range(len(u_data))], three_axis_pred[0][2] * xx)
            plt.show()

    main_data_list = np.stack(main_data_list, axis=0)
    x_data_list = np.stack(x_data_list, axis=0)
    y_data_list = np.stack(y_data_list, axis=0)
    u_data_list = np.stack(u_data_list, axis=0)

    print("main_data_list: ", main_data_list.min(), main_data_list.max())
    print("x_data_list: ", x_data_list.min(), x_data_list.max())
    print("y_data_list: ", y_data_list.min(), y_data_list.max())
    print("u_data_list: ", u_data_list.min(), u_data_list.max())

    np.save(main_data_file, main_data_list)
    np.save(x_data_file, x_data_list)
    np.save(y_data_file, y_data_list)
    np.save(u_data_file, u_data_list)