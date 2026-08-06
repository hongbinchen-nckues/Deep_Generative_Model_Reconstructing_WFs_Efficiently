


import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

def fourier_features_multi(x, num_freqs=6):
    '''
    diffusion / NeRF 常用的 (multi-frequency encoding)
    
    :param x: [B, 3, 721]
    :param num_freqs: Description
    :return: [B, 3*2*num_freqs, 721]
    '''
    
    B, C, L = x.shape
    freqs = 2 ** torch.arange(num_freqs, device=x.device)  # [num_freqs]
    
    # reshape for broadcasting
    freqs = freqs.view(1, 1, 1, num_freqs)  # [1,1,1,F]

    # x -> [B, 3, 721]
    x_freq = x.unsqueeze(-1) * freqs  # [B, 3, 721, F]
    
    sin_feat = torch.sin(x_freq) # [torch.sin(x), torch.sin(x*2), torch.sin(x*4), ...]
    cos_feat = torch.cos(x_freq) # [torch.cos(x), torch.cos(x*2), torch.cos(x*4), ...]
    
    # concat sin & cos
    out = torch.cat([sin_feat, cos_feat], dim=-1)  # [B, 3, 721, 2F]
    
    # reshape to channel dimension
    out = out.permute(0, 1, 3, 2).reshape(B, C * 2 * num_freqs, L)

    return out

def Back2MDx3(y_pred, vmin, vmax, step = 7, point = 721, channel = 1):
    import cv2
    from scipy import integrate, interpolate
    side_dim = y_pred.shape[1]

    num = side_dim
    we = (2 * step) / (num - 1)

    three_axis_pred = []
    for t in range(y_pred.shape[0]):
        z = y_pred[t]

        if z is None:
            raise ValueError(f"Error: Variable 'z' is None at index {t}.")
        z = np.asarray(z)
        if z.ndim not in (2, 3):
            raise ValueError(f"Error: 'z' has {z.ndim} dims at index {t}, but must be 2D or 3D.")
        if z.dtype not in (np.uint8, np.float32, np.float64):
            raise ValueError(f"Error: 'z' type incorrect. Current type is {z.dtype}, but only np.float32 and np.float64 are supported.")

        z = cv2.resize(z, (side_dim, side_dim))
        z = cv2.GaussianBlur(z, (9, 9), 1.5)

        x_s = []
        for i in range(z.shape[0]):
            x_z = integrate.simpson(z[:, i]) * we
            x_s.append(x_z)
        x_s = np.array(x_s)

        y_s = []
        for i in range(z.shape[1]):
            y_z = (integrate.simpson(z[i, :])) * we
            y_s.append(y_z)
        y_s = np.array(y_s)

        z_s = []
        n1 = [i for i in range(1, num + 1, 2)]
        n2 = [i for i in range(num - 1, -1, -2)]
        for k in n1:
            t_per1 = []
            for j in range(k):
                t_per1.append(np.flip(z, 0)[(num - 1) - (k - 1 - j), j])
            t_per1 = np.array(t_per1)
            tval = integrate.simpson(t_per1) * we * np.sqrt(2)
            z_s.append(tval)

        for k in n2:
            t_per2 = []
            for j in range(k):
                t_per2.append(np.flip(z, 0)[j, (num - 1) - (k - 1 - j)])
            t_per2 = np.array(t_per2)
            tval = integrate.simpson(t_per2) * we * np.sqrt(2)
            z_s.append(tval)

        z_s = np.array(z_s)


        mlin = np.linspace(-step, step, side_dim)
        mlin2 = np.linspace(-step, step, point)
        mfx = interpolate.interp1d(mlin, x_s, kind="linear")
        mfy = interpolate.interp1d(mlin, y_s, kind="linear")
        mfu = interpolate.interp1d(mlin, z_s, kind="linear")

        mx_new = mfx(mlin2)
        my_new = mfy(mlin2)
        mu_new = mfu(mlin2)

        m = np.stack((mx_new, my_new, mu_new), axis=0)
        three_axis_pred.append(m)

    three_axis_pred = np.stack(three_axis_pred, axis=0)
    return three_axis_pred

def rgb_cmap(z, vmin, vmax): 
    '''
    z shape -> [num, side_dim, side_dim] or [side_dim, side_dim]
    return R,G,B -> [side_dim, side_dim, 3] clip -> (0,1)
    '''

    z = np.clip(z, vmin, vmax)
    # normalization
    h = (z - vmin)/(vmax - vmin)
    zero = (0 - vmin)/(vmax - vmin)

    # color mapping
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    rgb = np.array(rgb)

    return rgb

def rgb_cmap3(vmin = -0.25, vmax = 0.25, num_colors = 256):
    h = np.linspace(0.2, 0.8, num_colors)
    zero = (0 - vmin) / (vmax - vmin)
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    rgb = np.clip(cv2.merge([r, g, b]), 0, 1)

    cmap = mcolors.ListedColormap(rgb, name='cmap', N=num_colors)

    return cmap

def CompareJDAndMDFig(ground_true, predict, ground_true_marginal, predict_marginal, file_path, vmin, vmax, label : list[int] = None, step = 7, point = 721):
    '''
    - ground_true(np.float32) [num, side_dim, side_dim]
    - ground_true_marginal(np.float32) [num, 3, point]
    - predict(np.float32) [num, side_dim, side_dim]
    - file_path(str)
    '''
    import matplotlib.pyplot as plt
    from scipy import interpolate
    side_dim = ground_true.shape[1]
    mlin = np.linspace(-step, step, side_dim)
    num = len(ground_true)


    # If shape wrong
    if ground_true_marginal.ndim == 2:
        ground_true_marginal = np.reshape(ground_true_marginal, (num, 3, point))

    # L2 Image Loss
    LossL2 = []
    side_dim = ground_true.shape[1]
    for i in range(len(ground_true)):
        LossL2.append(np.sqrt(np.pow(ground_true[i] - predict[i],2).sum() / (side_dim*side_dim)))

    # ground_true: [num, side_dim, side_dim] -> [num, side_dim, side_dim, 3]
    ground_true_c = rgb_cmap(ground_true, vmin, vmax)
    # predict: [num, side_dim, side_dim] -> [num, side_dim, side_dim, 3]
    predict_c = rgb_cmap(predict, vmin, vmax)


    # L1 Marginal Loss
    LossL1 = np.abs(ground_true_marginal - predict_marginal).sum(axis=2) / point
    
    # make plt
    fig, ax = plt.subplots(6, num, figsize = (num*3, 6*3))
    for i in range(num):
        ax[0,i].imshow(ground_true_c[i])
        if label is not None:
            if label[i] == 0:
                ax[0,i].set_title("0")
            elif label[i] == 1:
                ax[0,i].set_title("1")
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
    for i in range(num):
        ax[1,i].imshow(predict_c[i])
        ax[1,i].set_title(f"{LossL2[i]:.3E}")
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
    for i in range(num):
        error_map = np.abs(ground_true[i] - predict[i])
        real_max = np.max(error_map)
        ax[2,i].imshow(error_map, cmap="binary", vmin = 0, vmax = vmax * 0.3, interpolation="none")
        ax[2,i].text(0.03, 0.95, f"max={real_max:.1e}", fontsize=10, ha="left", va="top", color="black")
        ax[2,i].set_xticks([])
        ax[2,i].set_yticks([])
    for j in range(3):
        for i in range(num):
            ax[3+j,i].plot([i for i in range(point)], ground_true_marginal[i][j], c="black")
            ax[3+j,i].plot([i for i in range(point)], predict_marginal[i][j], c="red")
            ax[3+j,i].set_title(f"{LossL1[i,j]:.3E}")
            ax[3+j,i].set_xticks([0,point-1])

    plt.tight_layout()

    # save figure
    print(f"save at {file_path}")
    plt.savefig(file_path)

def _to_scalar_field(arr, vmin, vmax):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = rgb_cmap(arr, vmin, vmax)
    return arr.astype(np.float32)

def CompareOT(Y, y_pred, three_axis_gt, three_axis_pred, vmin, vmax, step = 7, point = 721):
    '''
        Y, y_pred, three_axis_gt, three_axis_pred -> numpy
    '''
    if os.path.exists("./DiffusionGenerateImage/OT") == False:
        os.mkdir("./DiffusionGenerateImage/OT")
    front_path = "./DiffusionGenerateImage/OT"
    save_root = "./DiffusionGenerateImage/OT"
    save_tag = "ALL_TEST"
    
    if len(three_axis_gt.shape) == 3:
        three_axis_gt = three_axis_gt.reshape(len(three_axis_gt), 3 * point)
    if len(three_axis_pred.shape) == 3:
        three_axis_pred = three_axis_pred.reshape(len(three_axis_pred), 3 * point)
    
    if Y.ndim == 2:
        Y = Y.reshape(len(Y), 256, 256)
    if y_pred.ndim == 2:
        y_pred = y_pred.reshape(len(Y), 256, 256)

    image_num = len(y_pred)//5
    for t in range(image_num):
        pic1 = t * 5
        pictrue_num = 5

        fig_dir = os.path.join(save_root, f"figure_{t+1}")
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        for i in range(pictrue_num):
            idx = pic1 + i

            # ---------- 1. Ground truth 2D Wigner matrix ----------
            gt_matrix = _to_scalar_field(Y[idx], vmin, vmax)
            gt_matrix = np.nan_to_num(gt_matrix, nan=0.0, posinf=0.05, neginf=-0.05)
            gt_matrix = np.clip(gt_matrix, -0.05, 0.05)

            # ---------- 2. Predicted 2D Wigner matrix ----------
            pred_matrix = _to_scalar_field(y_pred[idx], vmin, vmax)
            pred_matrix = np.nan_to_num(pred_matrix, nan=0.0, posinf=0.05, neginf=-0.05)
            pred_matrix = np.clip(pred_matrix, -0.05, 0.05)

            # ---------- 3. Absolute difference 2D Wigner matrix ----------
            abs_diff_matrix = np.abs(gt_matrix - pred_matrix)

            coord = np.linspace(-step, step, 256)
            X_grid, P_grid = np.meshgrid(coord, coord)

            x_flat = X_grid.ravel()
            p_flat = P_grid.ravel()
            gt_matrix = gt_matrix.ravel()

            # Ground truth → (x, p, WF)
            gt_triplet = np.column_stack((x_flat, p_flat, gt_matrix))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_ground_truth.csv"),
                gt_triplet,
                delimiter=",",
                fmt="%.8e",
                header="x,p,WF_value",
                comments=""
            )

            # Predicted → (x, p, WF)
            pred_triplet = np.column_stack((x_flat, p_flat, pred_matrix.ravel()))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_predicted.csv"),
                pred_triplet,
                delimiter=",",
                fmt="%.8e",
                header="x,p,WF_value",
                comments=""
            )

            # Absolute difference → (x, p, WF)
            abs_triplet = np.column_stack((x_flat, p_flat, abs_diff_matrix.ravel()))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_absolute_difference.csv"),
                abs_triplet,
                delimiter=",",
                fmt="%.8e",
                header="x,p,WF_value",
                comments=""
            )

            # ---------- 4. x marginal ----------
            # x_gt = X[idx][: int((X.shape[1]) / 3 + 1)]
            # x_pred = three_axis_pred[idx][: int((three_axis_pred.shape[1]) / 3 + 1)]
            x_gt = three_axis_gt[idx][0:point]
            x_pred = three_axis_pred[idx][0:point]
            x_pair = np.column_stack((x_gt, x_pred))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_x_marginal.csv"),
                x_pair,
                delimiter=",",
                fmt="%.8e",
                header="ground_truth,predicted",
                comments=""
            )

            # ---------- 5. p marginal ----------
            # p_gt = X[idx][int((X.shape[1]) / 3) : int((X.shape[1]) * 2 / 3 + 1)]
            # p_pred = three_axis_pred[idx][int((three_axis_pred.shape[1]) / 3) : int((three_axis_pred.shape[1]) * 2 / 3 + 1)]
            p_gt = three_axis_gt[idx][point:2*point]
            p_pred = three_axis_pred[idx][point:2*point]
            p_pair = np.column_stack((p_gt, p_pred))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_p_marginal.csv"),
                p_pair,
                delimiter=",",
                fmt="%.8e",
                header="ground_truth,predicted",
                comments=""
            )

            # ---------- 6. u marginal ----------
            # u_gt = X[idx][int((X.shape[1]) * 2 / 3) : -1]
            # u_pred = three_axis_pred[idx][int((three_axis_pred.shape[1]) * 2 / 3) : -1]
            u_gt = three_axis_gt[idx][2*point:3*point]
            u_pred = three_axis_pred[idx][2*point:3*point]
            u_pair = np.column_stack((u_gt, u_pred))
            np.savetxt(
                os.path.join(fig_dir, f"sample_{idx}_u_marginal.csv"),
                u_pair,
                delimiter=",",
                fmt="%.8e",
                header="ground_truth,predicted",
                comments=""
            )

    print(f"All matrices and marginals are saved in: {save_root}")

    for t in range(image_num):
        fig_dir = os.path.join(save_root, f"figure_{t+1}")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off()  

        pic1 = t * 5
        pictrue_num = 5

        pn_a1 = pictrue_num + 1
        # ---- GT ----
        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, 1)
            plt.text(0.5, 0.15, "Ground\ntruth\ndistribution", fontsize=8, ha="center")
            plt.axis("off")
            plt.subplot(6, pn_a1, i + 2)
        # =============================================================================
            cmap = rgb_cmap3()
            norm = mcolors.TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.11)


            gt_show = _to_scalar_field(Y[i + pic1], vmin, vmax)
            gt_show = np.nan_to_num(gt_show, nan=0.0, posinf=0.05, neginf=-0.05)
            gt_show = np.clip(gt_show, -0.05, 0.05)
            plt.imshow(np.flip(gt_show, 0), cmap=cmap, norm=norm, interpolation="none")
        # =============================================================================
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")

        # ---- Pred ----
        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, pn_a1 + 1)
            plt.text(0.5, 0.15, "Predicted\njoint\ndistribution", fontsize=8, ha="center")
            plt.axis("off")

            plt.subplot(6, pn_a1, i + pn_a1 + 2)
        # =============================================================================         
            gt = _to_scalar_field(Y[i + pic1], vmin, vmax)
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.05, neginf=-0.05)
            gt = np.clip(gt, -0.05, 0.05)

            pr_show = _to_scalar_field(y_pred[i + pic1], vmin, vmax)
            pr_show = np.nan_to_num(pr_show, nan=0.0, posinf=0.05, neginf=-0.05)
            pr_show = np.clip(pr_show, -0.05, 0.05)

            plt.imshow(np.flip(pr_show, 0), cmap=cmap, norm=norm, interpolation="none")

            diff = np.sqrt(np.sum((gt - pr_show)**2) / (256 * 256))
            ax = plt.gca()
            ax.text(
                0.97, 0.95,
                f"{diff:.1e}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="black"
            )

            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")

        # =============================================================================         
        # ---- Absolute difference ----
        last_im = None
        abs_axes = []


        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, 2 * pn_a1 + 1)
            plt.text(0.5, 0.15, "Absolute\ndifferences", fontsize=8, ha="center")
            plt.axis("off")

            plt.subplot(6, pn_a1, i + 2 * pn_a1 + 2)

            gt = _to_scalar_field(Y[i + pic1], vmin, vmax)
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.05, neginf=-0.05)
            gt = np.clip(gt, -0.05, 0.05)

            pr_show = _to_scalar_field(y_pred[i + pic1], vmin, vmax)
            pr_show = np.nan_to_num(pr_show, nan=0.0, posinf=0.05, neginf=-0.05)
            pr_show = np.clip(pr_show, -0.05, 0.05)

            error_map = np.abs(gt - pr_show)
            real_max = np.max(error_map)
            mean_diff = np.mean(error_map)

            im = plt.imshow(
                np.flip(error_map, 0),
                cmap="binary",
                vmin=0,
                vmax=vmax * 0.3,
                interpolation="none"
            )

            last_im = im
            ax = plt.gca()
            abs_axes.append(ax)

            ax.text(
                0.03, 0.95,
                f"max={real_max:.1e}",
                transform=ax.transAxes,
                fontsize=5,
                ha="left",
                va="top",
                color="black"
            )

            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")


        if last_im is not None:
            cbar = fig.colorbar(
                last_im,
                ax=abs_axes,
                fraction=0.025,
                pad=0.02
            )
            cbar.ax.tick_params(labelsize=6)
            # cbar.set_label("Abs. diff.", fontsize=7)
            pos = cbar.ax.get_position()
            cbar.ax.set_position([
                pos.x0,
                pos.y0 + 0.03,
                pos.width,
                pos.height
            ])

        # ---- x/p/u marginals ----
        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, 3 * pn_a1 + 1)

            plt.text(
                0.5, 0.15, "x-marginal\ndistributions\ncomparison", fontsize=8, ha="center"
            )
            plt.axis("off")

            plt.subplot(6, pn_a1, i + 3 * pn_a1 + 2)
            cmap2 = ["black", "red"]

            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1][0:point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1][0:point],
                cmap2[1],
                linewidth=0.5,
            )

            diffx = np.mean(
                np.abs(
                    three_axis_gt[i + pic1][0:point]
                    - three_axis_pred[i + pic1][0:point]
                )
            )

            plt.text(
                point,
                np.max(three_axis_pred[i + pic1][0:point]),
                f"{diffx:.1e}",
                fontsize=6,
                ha="right",
                va="top",
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, 4 * pn_a1 + 1)
            plt.text(
                0.5, 0.15, "p-marginal\ndistributions\ncomparison", fontsize=8, ha="center"
            )
            plt.axis("off")

            plt.subplot(6, pn_a1, i + 4 * pn_a1 + 2)
            cmap2 = ["black", "red"]

            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1][point:2*point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1][point:2*point],
                cmap2[1],
                linewidth=0.5,
            )

            diffy = np.mean(
                np.abs(
                    three_axis_gt[i + pic1][point:2*point]
                    - three_axis_pred[i + pic1][point:2*point]
                )
            )

            plt.text(
                point,
                np.max(
                    three_axis_pred[i + pic1][
                        int((three_axis_pred.shape[1]) / 3) : int((three_axis_pred.shape[1]) * 2 / 3 + 1)
                    ]
                ),
                f"{diffy:.1e}",
                fontsize=6,
                ha="right",
                va="top",
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        for i in range(pictrue_num):
            plt.subplot(6, pn_a1, 5 * pn_a1 + 1)
            plt.text(
                0.5, 0.15, "u-marginal\ndistributions\ncomparison", fontsize=8, ha="center"
            )
            plt.axis("off")

            plt.subplot(6, pn_a1, i + 5 * pn_a1 + 2)
            cmap2 = ["black", "red"]

            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1][2*point:3*point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1][2*point:3*point],
                cmap2[1],
                linewidth=0.5,
            )


            diffu = np.mean(
                np.abs(
                    three_axis_gt[i + pic1][2*point:3*point]
                    - three_axis_pred[i + pic1][2*point:3*point]
                )
            )

            plt.text(
                point,
                np.max(three_axis_pred[i + pic1][int((three_axis_pred.shape[1]) * 2 / 3) : -1]),
                f"{diffu:.1e}",
                fontsize=6,
                ha="right",
                va="top",
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        # fig.tight_layout()

        prediction_filename = "test_visualization_Wigner"
        path_prediction = os.path.join(front_path, prediction_filename)
        if not os.path.exists(path_prediction):
            os.mkdir(path_prediction)

        fig.savefig(
            os.path.join(path_prediction, f"test_comparison_{save_tag}_{t}.png"),
            dpi=2000,
        )

        plt.savefig(f"{fig_dir}/result.png")

def MNISTToMarginalDistribution(images, show_image = False):
    images = images.squeeze()
    batch_size = len(images)
    dim = images.shape[1]
    output = []
    for b in range(batch_size):
        # x_1 part
        x_1 = torch.sum(images[b], dim = 0) / dim

        # x_13 part
        x_13 = torch.sum(images[b], dim = 1) / dim

        # u part
        u_all = torch.zeros([dim*2+1])
        u = torch.zeros([dim])
        i = 0
        for k in range(-(dim-1), dim):
            s = 0
            i_min = max(0, k)
            i_max = min(dim-1, dim-1 + k)
            if i_min <= i_max:
                j_s = np.arange(i_min - k, i_max - k + 1)
                i_s = np.arange(i_min, i_max + 1)
                s = images[b, i_s, j_s].sum()
            u_all[i+1] = s
            i += 1
        for i in range(dim):
            u[i] = u_all[i*2]/2 + u_all[i*2+1] + u_all[i*2+2]/2
        
        u = u / (2*dim)
        
        '''
        if show_image == True:
            
            fig, ax = plt.subplots(1,4, figsize=(8,2))
            ax[0].imshow(torch.stack([images[b],images[b],images[b]],dim=-1).numpy())
            ax[0].set_title(f"Image ({b})")
            ax[1].plot([i for i in range(dim)], x_1.numpy())
            ax[1].set_title("x_1")
            ax[2].plot([i for i in range(dim)], x_13.numpy())
            ax[2].set_title("x_13")
            ax[3].plot([i for i in range(dim)], u.numpy())
            ax[3].set_title("u")

            plt.tight_layout()
            plt.savefig(os.path.join("QGAN", "Image", "MNISTToMarginal", f"{image_label}.png"))
            plt.close()
        '''

        stk = torch.stack([x_1, x_13, u], dim=0)
        output.append(stk)
    return torch.stack(output, dim=0) #[batch_size, 3, dim, dim]