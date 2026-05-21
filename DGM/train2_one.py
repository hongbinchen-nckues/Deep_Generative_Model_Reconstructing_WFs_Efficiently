import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
import cv2

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scipy import integrate, interpolate

# from model import ResNet_base_decoder, ResNet_base_decoder2, ResNet_base_decoder3, ResNet_base_decoder4
from TF import  ResNetBaseDecoder, ResNetBaseDecoder2, ResNetBaseDecoder3, ResNetBaseDecoder4, ResNetBaseDecoder5, ResNetBaseDecoder6

from ResNet import ResNet152Decoder1
from ResNeXt_2 import ResNeXt50Decoder_32x4d, ResNeXt101Decoder_32x4d, ResNeXt152Decoder_32x4d
from ResNeXt_3 import ResNeXt50Decoder_32x4d_fc, ResNeXt101Decoder_32x4d_fc, ResNeXt152Decoder_32x4d_fc
from DenseNet import DenseNetDecoder169
from cmap import rgb_cmap_inverse2, rgb_cmap_inverse3, rgb_cmap, rgb_cmap_inverse, rgb_cmap2, rgb_cmap4, rgb_cmap3


# -----------------------
# 小工具：把影像統一成 (H,W) 的標量場（僅用於視覺化與誤差計算）
# -----------------------
def _to_scalar_field(arr, vmin, vmax):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        arr = rgb_cmap_inverse2(arr, vmin, vmax)
    return arr.astype(np.float32)


# =============================
# integrating2：完全比照第一版（TensorFlow）流程
# =============================
def integrating2(y_pred, vmin, vmax, step, point, channel):
    """
    與第一版 (TF) 完全等價的積分流程：
    - 固定 num=256
    - resize -> (256,256)
    - GaussianBlur(9,9,1.5)
    - x/p 軸用 Simpson 積分並乘 we
    - u 軸沿對角線分兩段聚合，乘 we*sqrt(2)
    - 最後用 linear 插值到 'point' 長度，串成 (x|p|u) 三段
    - 若 channel==1：不做色盤反解；否則對 (H,W,3) 做 rgb_cmap_inverse2
    """
    num = 256
    we = (2 * step) / (num - 1)

    three_axis_pred = []
    for t in range(y_pred.shape[0]):
        # 與第一版一致的通道處理
        if channel == 1:
            z = y_pred[t]
        else:
            z = rgb_cmap_inverse2(y_pred[t], vmin, vmax)

        if z is None:
            raise ValueError(f"Error: Variable 'z' is None at index {t}.")
        z = np.asarray(z)
        if z.ndim not in (2, 3):
            raise ValueError(f"Error: 'z' has {z.ndim} dims at index {t}, but must be 2D or 3D.")
        if z.dtype not in (np.uint8, np.float32, np.float64):
            z = z.astype(np.float32)

        # 與第一版一致的 resize + blur
        z = cv2.resize(z, (256, 256))
        z = cv2.GaussianBlur(z, (9, 9), 1.5)

        # 若仍是 3 通道，轉回標量
        if z.ndim == 3 and z.shape[-1] == 3:
            z = rgb_cmap_inverse2(z, vmin, vmax)

        # x 軸（沿 columns）
        x_s = []
        for i in range(z.shape[0]):
            x_z = integrate.simpson(z[:, i]) * we
            x_s.append(x_z)
        x_s = np.array(x_s)

        # y 軸（沿 rows）
        y_s = []
        for i in range(z.shape[1]):
            y_z = integrate.simpson(z[i, :]) * we
            y_s.append(y_z)
        y_s = np.array(y_s)

        # u 軸（對角線）
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

        # 與第一版一致的插值座標
        mlin = np.linspace(-step, step, 256)
        mlin2 = np.linspace(-step, step, point)
        mfx = interpolate.interp1d(mlin, x_s, kind="linear")
        mfy = interpolate.interp1d(mlin, y_s, kind="linear")
        mfu = interpolate.interp1d(mlin, z_s, kind="linear")

        mx_new = mfx(mlin2)
        my_new = mfy(mlin2)
        mu_new = mfu(mlin2)

        m = np.concatenate((mx_new, my_new, mu_new), axis=0)
        three_axis_pred.append(m)

    three_axis_pred = np.asarray(three_axis_pred)
    three_axis_pred = three_axis_pred.reshape(three_axis_pred.shape[0], three_axis_pred.shape[1], 1)
    return three_axis_pred

# =============================
# 資料檢查 / 過濾工具
# =============================
def _has_bad_values(arr):
    arr = np.asarray(arr)
    return np.isnan(arr).any() or np.isinf(arr).any()

def _all_zero(arr, tol=1e-15):
    arr = np.asarray(arr)
    return np.all(np.abs(arr) < tol)

def _too_large(arr, max_abs=1e6):
    arr = np.asarray(arr)
    return np.max(np.abs(arr)) > max_abs

def filter_bad_samples(x_data, y_data, max_abs=1e6, verbose=True):
    """
    自動過濾掉：
    1. x 或 y 含 NaN / Inf
    2. x 或 y 幾乎全零
    3. x 或 y 數值過大
    """
    good_idx = []
    bad_idx = []

    for i in range(len(x_data)):
        x = x_data[i]
        y = y_data[i]

        bad_reason = None

        if _has_bad_values(x):
            bad_reason = "X has NaN/Inf"
        elif _has_bad_values(y):
            bad_reason = "Y has NaN/Inf"
        elif _all_zero(x):
            bad_reason = "X all zero"
        elif _all_zero(y):
            bad_reason = "Y all zero"
        elif _too_large(x, max_abs=max_abs):
            bad_reason = f"X abs > {max_abs}"
        elif _too_large(y, max_abs=max_abs):
            bad_reason = f"Y abs > {max_abs}"

        if bad_reason is None:
            good_idx.append(i)
        else:
            bad_idx.append((i, bad_reason))
            if verbose:
                print(f"⚠️ remove bad sample {i}: {bad_reason}")

    x_good = x_data[good_idx]
    y_good = y_data[good_idx]

    print(f"Original samples : {len(x_data)}")
    print(f"Filtered samples : {len(x_good)}")
    print(f"Removed samples  : {len(bad_idx)}")

    return x_good, y_good, good_idx, bad_idx


# =============================
# 選模型（維持 channel==1 / 3 的行為）
# =============================
def _pick_model(input_shape, channel):
    # TF.py 的模型吃 flat 向量
    if isinstance(input_shape, tuple):
        input_dim = int(np.prod(input_shape))
    else:
        input_dim = int(input_shape)

    if channel == 1:
        return DenseNetDecoder169(input_dim)
    else:
        return DenseNetDecoder264(input_dim)


def training2(
    dataset_filename,
    model_name,
    epochs=250,
    batch_size=32,
    data_patch=4,
    channel=1,
    resume=True,
    save_every=1,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set save path
    osd = os.getcwd()
    front_path = os.path.abspath(os.path.join(osd, os.path.pardir))
    path_dataset = os.path.join(front_path, dataset_filename)

    x_train, y_train = [], []
    for i in range(data_patch):
        x_train += list(
            np.load(
                os.path.join(path_dataset, f"x_train{i+1}.npy"),
                allow_pickle=True,
            )
        )
        y_train += list(
            np.load(
                os.path.join(path_dataset, f"y_train{i+1}.npy"),
                allow_pickle=True,
            )
        )

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("x_train.shape is", x_train.shape)
    print("y_train.shape is", y_train.shape)

    # =============================
    # 訓練前先過濾壞樣本
    # =============================
    x_train, y_train, good_idx, bad_idx = filter_bad_samples(
        x_train, y_train, max_abs=1e6, verbose=True
    )

    print("after filtering:")
    print("x_train.shape is", x_train.shape)
    print("y_train.shape is", y_train.shape)

    if len(x_train) == 0:
        raise ValueError("All training samples were removed after filtering!")

    input_shape = x_train[0].shape
    checkpoint_path = f"{model_name}_checkpoint.pth"
    best_model_path = f"{model_name}_best.pth"

    # ================================================================================
    # ① 建立原始模型並搬 GPU
    model = _pick_model(input_shape, channel).to(device)

    # ② compile 前先載入 model weights（未 compile 的 key 不含 _orig_mod. 前綴，完全匹配）
    start_epoch = 1
    best_val_loss = float("inf")
    history_tr, history_va = [], []
    _ckpt = None  # 暫存 checkpoint，供 compile 後還原 optimizer/scheduler/scaler

    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        _ckpt = torch.load(checkpoint_path, map_location=device)

        clean_state_dict = {}
        for k, v in _ckpt["model_state_dict"].items():
            clean_state_dict[k.replace("_orig_mod.", "")] = v

        model.load_state_dict(clean_state_dict)  # compile 前 load，key 完全匹配

    # ③ torch≥2.0 加速：compile 必須在 optimizer 建立前，否則 optimizer
    #   綁的參數對象在 compile 後換掉，gradient 更新會出問題
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode='reduce-overhead')
    # ================================================================================
    flatten_input = len(input_shape) != 1
    print(model)

    # --- 模擬 Keras: validation_split=0.2 ---
    N = len(x_train)
    idx = np.arange(N)
    # Keras: shuffle=True 會在 split 前洗牌一次（這裡不固定 seed）
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(np.floor(N * (1 - 0.2)))  # 前 80% 當訓練，後 20% 當驗證
    tr_idx = idx[:split]
    va_idx = idx[split:]

    # tensors on CPU first
    x_tr = torch.from_numpy(x_train[tr_idx]).float()
    y_tr = torch.from_numpy(y_train[tr_idx]).float()
    x_va = torch.from_numpy(x_train[va_idx]).float()
    y_va = torch.from_numpy(y_train[va_idx]).float()

    # convert Y to NCHW
    def to_nchw(t):
        if t.ndim == 3:
            return t.unsqueeze(1)
        if t.ndim == 4 and t.shape[-1] in (1, 3):
            return t.permute(0, 3, 1, 2)
        return t

    y_tr = to_nchw(y_tr)
    y_va = to_nchw(y_va)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, foreach=False)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.4, patience=8, min_lr=1e-6
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ④ compile 與 optimizer 建立後，還原訓練狀態
    if _ckpt is not None:
        optimizer.load_state_dict(_ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in _ckpt:
            scheduler.load_state_dict(_ckpt["scheduler_state_dict"])

        if "scaler_state_dict" in _ckpt and _ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(_ckpt["scaler_state_dict"])

        start_epoch = _ckpt["epoch"] + 1
        best_val_loss = _ckpt.get("best_val_loss", float("inf"))
        history_tr = _ckpt.get("history_tr", [])
        history_va = _ckpt.get("history_va", [])

        print(f"✅ Resume from epoch {start_epoch}")
    else:
        print("🆕 Start training from scratch")


    for ep in tqdm(range(start_epoch, epochs + 1), desc="Training", unit="epoch"):
        model.train()
        perm = torch.randperm(x_tr.size(0))
        epoch_loss = 0.0

        for s in tqdm(
            range(0, x_tr.size(0), batch_size),
            desc=f"Epoch {ep}/{epochs} [Train]",
            unit="batch",
            leave=False
        ):
            sel = perm[s : s + batch_size]
            xb = x_tr[sel]
            yb = y_tr[sel]
            if torch.isnan(xb).any() or torch.isinf(xb).any():
                print(f"⚠️ TRAIN batch {s // batch_size}: xb has NaN/Inf, skip")
                continue

            if torch.isnan(yb).any() or torch.isinf(yb).any():
                print(f"⚠️ TRAIN batch {s // batch_size}: yb has NaN/Inf, skip")
                continue

            if flatten_input:
                xb = xb.view(xb.size(0), -1)

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    if pred.dim() == 4 and yb.dim() == 4 and pred.shape[-2:] != yb.shape[-2:]:
                        pred = torch.nn.functional.interpolate(
                            pred, size=yb.shape[-2:], mode="bilinear", align_corners=False
                        )
                    if pred.dim() == 4 and yb.dim() == 4 and pred.shape[1] != yb.shape[1]:
                        if yb.shape[1] == 1 and pred.shape[1] > 1:
                            pred = pred.mean(dim=1, keepdim=True)
                        elif yb.shape[1] == 3 and pred.shape[1] == 1:
                            pred = pred.repeat(1, 3, 1, 1)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                if pred.dim() == 4 and yb.dim() == 4 and pred.shape[-2:] != yb.shape[-2:]:
                    pred = torch.nn.functional.interpolate(
                        pred, size=yb.shape[-2:], mode="bilinear", align_corners=False
                    )
                if pred.dim() == 4 and yb.dim() == 4 and pred.shape[1] != yb.shape[1]:
                    if yb.shape[1] == 1 and pred.shape[1] > 1:
                        pred = pred.mean(dim=1, keepdim=True)
                    elif yb.shape[1] == 3 and pred.shape[1] == 1:
                        pred = pred.repeat(1, 3, 1, 1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

            del xb, yb, pred, loss

        train_loss = epoch_loss / x_tr.size(0)

        
        # validation（batched）
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        bad_val_batches = 0

        with torch.no_grad():
            for s in tqdm(
                range(0, x_va.size(0), batch_size),
                desc=f"Epoch {ep}/{epochs} [Val]",
                unit="batch",
                leave=False
            ):
                xb = x_va[s : s + batch_size]
                yb = y_va[s : s + batch_size]

                if flatten_input:
                    xb = xb.view(xb.size(0), -1)

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # ===== 先檢查 validation input =====
                if torch.isnan(xb).any() or torch.isinf(xb).any():
                    print(f"⚠️ VAL batch {s // batch_size}: xb has NaN/Inf")
                    bad_val_batches += 1
                    continue

                if torch.isnan(yb).any() or torch.isinf(yb).any():
                    print(f"⚠️ VAL batch {s // batch_size}: yb has NaN/Inf")
                    bad_val_batches += 1
                    continue

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred_va = model(xb)

                        if pred_va.dim() == 4 and yb.dim() == 4 and pred_va.shape[-2:] != yb.shape[-2:]:
                            pred_va = torch.nn.functional.interpolate(
                                pred_va, size=yb.shape[-2:], mode="bilinear", align_corners=False
                            )

                        if pred_va.dim() == 4 and yb.dim() == 4 and pred_va.shape[1] != yb.shape[1]:
                            if yb.shape[1] == 1 and pred_va.shape[1] > 1:
                                pred_va = pred_va.mean(dim=1, keepdim=True)
                            elif yb.shape[1] == 3 and pred_va.shape[1] == 1:
                                pred_va = pred_va.repeat(1, 3, 1, 1)

                        loss_va = criterion(pred_va, yb)
                else:
                    pred_va = model(xb)

                    if pred_va.dim() == 4 and yb.dim() == 4 and pred_va.shape[-2:] != yb.shape[-2:]:
                        pred_va = torch.nn.functional.interpolate(
                            pred_va, size=yb.shape[-2:], mode="bilinear", align_corners=False
                        )

                    if pred_va.dim() == 4 and yb.dim() == 4 and pred_va.shape[1] != yb.shape[1]:
                        if yb.shape[1] == 1 and pred_va.shape[1] > 1:
                            pred_va = pred_va.mean(dim=1, keepdim=True)
                        elif yb.shape[1] == 3 and pred_va.shape[1] == 1:
                            pred_va = pred_va.repeat(1, 3, 1, 1)

                    loss_va = criterion(pred_va, yb)

                # ===== 再檢查 validation output / loss =====
                if torch.isnan(pred_va).any() or torch.isinf(pred_va).any():
                    print(f"⚠️ VAL batch {s // batch_size}: pred_va has NaN/Inf")
                    print("pred range:", float(torch.min(pred_va)), float(torch.max(pred_va)))
                    bad_val_batches += 1
                    continue

                if torch.isnan(loss_va) or torch.isinf(loss_va):
                    print(f"⚠️ VAL batch {s // batch_size}: loss_va is NaN/Inf")
                    print("xb range:", float(torch.min(xb)), float(torch.max(xb)))
                    print("yb range:", float(torch.min(yb)), float(torch.max(yb)))
                    print("pred range:", float(torch.min(pred_va)), float(torch.max(pred_va)))
                    bad_val_batches += 1
                    continue

                val_loss_sum += loss_va.item() * xb.size(0)
                val_count += xb.size(0)

                del xb, yb, pred_va, loss_va
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if bad_val_batches > 0:
            print(f"⚠️ validation skipped bad batches = {bad_val_batches}")

        if val_count == 0:
            val_loss = float("nan")
            print("⚠️ All validation batches were skipped. val_loss = nan")
        else:
            val_loss = val_loss_sum / val_count

        if not np.isnan(val_loss):
            scheduler.step(val_loss)
        else:
            print("⚠️ Skip scheduler.step() because val_loss is nan")

        history_tr.append(train_loss)
        history_va.append(val_loss)
        print(f"[{ep:03d}/{epochs}] loss={train_loss:.8f}  val_loss={val_loss:.8f}")


        # =============================
        # 儲存最佳模型
        # =============================
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "input_shape": input_shape,
                    "channel": channel,
                    "flatten_input": flatten_input,
                    "best_val_loss": best_val_loss,
                    "history_tr": history_tr,
                    "history_va": history_va,
                },
                best_model_path,
            )
            print(f"🌟 Best model saved: {best_model_path}")

        # =============================
        # 儲存最新 checkpoint
        # =============================
        if ep % save_every == 0:
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "input_shape": input_shape,
                    "channel": channel,
                    "flatten_input": flatten_input,
                    "best_val_loss": best_val_loss,
                    "history_tr": history_tr,
                    "history_va": history_va,
                },
                checkpoint_path,
            )
            print(f"💾 Checkpoint saved: {checkpoint_path}")



        # 存檔（沿用你的命名；建議可改 .pt/.pth）
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "channel": channel,
            "flatten_input": flatten_input,
        },
        f"{model_name}.py",
    )
    print(f"Model checkpoint saved: {model_name}.py")

    plt.figure()
    plt.plot(history_tr, label="Train")
    plt.plot(history_va, label="Validation")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig("model_loss.png", dpi=300, bbox_inches="tight")
    plt.show()


# =============================
# 推論：讀/建模都用 .py（state_dict）
# =============================
def _load_model_for_infer(model_name, input_shape, channel, device):
    ckpt = torch.load(f"{model_name}.py", map_location=device)

    # 去掉 _orig_mod. 前缀
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    saved_channel = ckpt.get("channel", channel)
    raw_shape = ckpt.get("input_shape", input_shape)
    saved_input_shape = (
        tuple(raw_shape) if isinstance(raw_shape, (list, tuple, np.ndarray)) else (int(raw_shape),)
    )
    flatten_input = bool(ckpt.get("flatten_input", (len(saved_input_shape) != 1)))

    model = _pick_model(saved_input_shape, saved_channel).to(device)
    model.load_state_dict(new_state_dict)  # 使用清洗后的 state_dict
    model.eval()
    return model, flatten_input


def prediction2(model_name, prediction_npy_name, x_test, y_test):
    # set save path
    prediction_filename = "prediction"
    front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    path_prediction = os.path.join(front_path, prediction_filename)
    if not os.path.exists(path_prediction):
        os.mkdir(path_prediction)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_shape = x_test[0].shape

    # 依 y_test 猜 channel（沒提供就當 1）
    channel_infer = 1
    if isinstance(y_test, np.ndarray) and y_test.ndim == 4 and y_test.shape[-1] == 3:
        channel_infer = 3

    model, flatten_input = _load_model_for_infer(
        model_name, input_shape, channel_infer, device
    )

    x = torch.from_numpy(x_test).float().to(device)
    if flatten_input:
        x = x.view(x.size(0), -1)

    with torch.no_grad():
        y_pred_t = model(x)

    y_pred = y_pred_t.detach().cpu().numpy()
    if y_pred.ndim == 4 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, ...]
    elif y_pred.ndim == 4 and y_pred.shape[1] == 3:
        y_pred = np.transpose(y_pred, (0, 2, 3, 1))

    np.save(os.path.join(path_prediction, prediction_npy_name), y_pred)


def test_comparison2(
    train_filename,
    model_name,
    image_num=5,
    vmin=-0.01,
    vmax=0.045,
    point=2241,  # 與第一版一致
    step=7,      # 與第一版一致
    channel=1,   # 與第一版常用一致（RGB colormap）
    a=2,         # 0=TJCM test, 1=JCM test, 2=all test, 3=train1
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    front_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    path_combined_dataset = os.path.join(front_path, train_filename)

    # X_test = np.load(os.path.join(path_combined_dataset, "x_test.npy"))
    # Y_test = np.load(os.path.join(path_combined_dataset, "y_test.npy"))

    x_test_path = os.path.join(path_combined_dataset, "x_test.npy")
    y_test_path = os.path.join(path_combined_dataset, "y_test.npy")

# =========================================================================
    # if os.path.exists(x_test_path) and os.path.exists(y_test_path):
    #     print("Load test dataset: x_test.npy / y_test.npy")
    #     X_test = np.load(x_test_path)
    #     Y_test = np.load(y_test_path)

    # else:
    #     print("x_test.npy not found. Skip test dataset and load training data instead.")
    #     X_test = np.load(os.path.join(path_combined_dataset, "x_train1.npy"))
    #     Y_test = np.load(os.path.join(path_combined_dataset, "y_train1.npy"))

    # X_train = np.load(os.path.join(path_combined_dataset, "x_train1.npy"))
    # Y_train = np.load(os.path.join(path_combined_dataset, "y_train1.npy"))
# ===========================================================================
    x_test_path = os.path.join(path_combined_dataset, "x_test.npy")
    y_test_path = os.path.join(path_combined_dataset, "y_test.npy")
    test_label_path = os.path.join(path_combined_dataset, "test_label.npy")

    # image_num=5 時，後面畫圖最多會用到約 5*6 筆
    max_needed = max(image_num * 6, 1)

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        print("Load test dataset: x_test.npy / y_test.npy")
        X_test_all = np.load(x_test_path, mmap_mode="r")
        Y_test_all = np.load(y_test_path, mmap_mode="r")
        using_real_test = True
    else:
        print("x_test.npy not found. Use small subset from x_train1.npy instead.")
        X_test_all = np.load(os.path.join(path_combined_dataset, "x_train1.npy"), mmap_mode="r")
        Y_test_all = np.load(os.path.join(path_combined_dataset, "y_train1.npy"), mmap_mode="r")
        using_real_test = False

    if os.path.exists(test_label_path):
        test_label = np.load(test_label_path)
        print("Load test_label.npy")
    else:
        test_label = None
        print("test_label.npy not found. Skip label-based selection.")

# ===========================================================================


    # 讀 test label：0 = TJCM, 1 = JCM
    # test_label = np.load(os.path.join(path_combined_dataset, "test_label.npy"))

    test_label_path = os.path.join(path_combined_dataset, "test_label.npy")

    if os.path.exists(test_label_path):
        test_label = np.load(test_label_path)
        print("Load test_label.npy")
    else:
        test_label = None
        print("test_label.npy not found. Skip label-based selection.")


# ==============================================================================
    # # 依照 a 選資料
    # if a == 0:
    #     selected_idx = np.where(test_label == 0)[0]
    #     X = X_test[selected_idx]
    #     Y = Y_test[selected_idx]
    #     save_tag = "TJCM"
    # elif a == 1:
    #     selected_idx = np.where(test_label == 1)[0]
    #     X = X_test[selected_idx]
    #     Y = Y_test[selected_idx]
    #     save_tag = "JCM"
    # elif a == 2:
    #     X = X_test
    #     Y = Y_test
    #     save_tag = "ALL_TEST"
    # else:
    #     X = X_train
    #     Y = Y_train
    #     save_tag = "TRAIN"
# ==================================================================================
    if a == 0 and test_label is not None:
        selected_idx = np.where(test_label == 0)[0]
        save_tag = "TJCM"

    elif a == 1 and test_label is not None:
        selected_idx = np.where(test_label == 1)[0]
        save_tag = "JCM"

    elif a == 2:
        selected_idx = np.arange(len(X_test_all))
        save_tag = "ALL_TEST" if using_real_test else "TRAIN_SUBSET"

    else:
        selected_idx = np.arange(len(X_test_all))
        save_tag = "TRAIN_SUBSET"

    # 關鍵：只取畫圖真正需要的資料量
    selected_idx = selected_idx[:min(max_needed, len(selected_idx))]

    X = np.asarray(X_test_all[selected_idx], dtype=np.float32)
    Y = np.asarray(Y_test_all[selected_idx], dtype=np.float32)
# ==================================================================================

    print("selected dataset:", save_tag)
    print("X.shape =", X.shape)
    print("Y.shape =", Y.shape)

    input_shape = X[0].shape
    model, flatten_input = _load_model_for_infer(model_name, input_shape, channel, device)

    # ==================================================================
    # with torch.no_grad():
    #     x_in = torch.from_numpy(X).float().to(device)
    #     if flatten_input:
    #         x_in = x_in.view(x_in.size(0), -1)
    #     y_pred_t = model(x_in)

    # y_pred = y_pred_t.detach().cpu().numpy().astype(np.float32)
    # ==================================================================
    infer_batch_size = 1
    y_pred_list = []

    model.eval()

    with torch.no_grad():
        for s in range(0, len(X), infer_batch_size):
            x_batch = torch.from_numpy(X[s:s+infer_batch_size]).float()

            if flatten_input:
                x_batch = x_batch.view(x_batch.size(0), -1)

            x_batch = x_batch.to(device, non_blocking=True)

            y_pred_t = model(x_batch)
            y_pred_np = y_pred_t.detach().cpu().numpy().astype(np.float32)
            y_pred_list.append(y_pred_np)

            del x_batch, y_pred_t

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_pred = np.concatenate(y_pred_list, axis=0)

    # ==================================================================


    infer_batch_size = 1
    y_pred_list = []

    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), infer_batch_size):
            x_batch = torch.from_numpy(X[s:s + infer_batch_size]).float()

            if flatten_input:
                x_batch = x_batch.view(x_batch.size(0), -1)

            x_batch = x_batch.to(device, non_blocking=True)

            y_pred_t = model(x_batch)
            y_pred_np = y_pred_t.detach().cpu().numpy().astype(np.float32)
            y_pred_list.append(y_pred_np)

            del x_batch, y_pred_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    y_pred = np.concatenate(y_pred_list, axis=0)
    # ==================================================================

    # 輸出形狀修正
    if y_pred.ndim == 4 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0, ...]
    elif y_pred.ndim == 4 and y_pred.shape[1] == 3:
        y_pred = np.transpose(y_pred, (0, 2, 3, 1))

    # Ground truth marginal
    three_axis_gt = integrating2(
        Y, vmin, vmax, step=step, point=point, channel=channel
    )[:, :, 0]

    # Predicted marginal
    three_axis_pred = integrating2(
        y_pred, vmin, vmax, step=step, point=point, channel=channel
    )[:, :, 0]

    print("y_pred.shape is", y_pred.shape)
    print("three_axis_pred.shape is", three_axis_pred.shape)
    print("x.shape is", X.shape)
    print("y.shape is", Y.shape)

    print("input image_num =", image_num)

# =============================================================================
    # 每張大圖實際會用到的最大 index = 6*t + 4
    # 若資料長度為 len(X)，安全的大圖數量是 floor((len(X)+1)/6)
    max_image_num = (len(X) + 1) // 6
    image_num = min(image_num, max_image_num)

    print("safe image_num =", image_num)
# =============================================================================

    # =========================================================
    # 儲存最後展示圖對應的五種資料到 data_martix
    # =========================================================
    save_root = os.path.join(front_path, "data_martix")
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for t in range(image_num):
        pic1 = t * 6 + 1
        pic2 = t * 6 + 6
        pictrue_num = pic2 - pic1 + 1

        fig_dir = os.path.join(save_root, f"{save_tag}_figure_{t+1}")
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        for i in range(pictrue_num - 1):
            idx = i + pic1 - 1

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

            # 建立對應的 x, p 座標
            coord = np.linspace(-step, step, 256)
            X_grid, P_grid = np.meshgrid(coord, coord)

            x_flat = X_grid.ravel()
            p_flat = P_grid.ravel()

            # Ground truth → (x, p, WF)
            gt_triplet = np.column_stack((x_flat, p_flat, gt_matrix.ravel()))
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

            # ---------- 3. x marginal ----------
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

            # ---------- 4. p marginal ----------
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
            # ---------- 5. u marginal ----------
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
        fig, ax = plt.subplots()
        ax.set_axis_off()   # ← 關掉外層主軸（不影響子圖）


        pic1 = t * 6 + 1
        pic2 = t * 6 + 6
        print(f"t={t}, pic1={pic1}, pic2={pic2}, len(Y)={len(Y)}")
        pictrue_num = pic2 - pic1 + 1

        # ---- GT ----
        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, 1)
            plt.text(0.5, 0.15, "Ground\ntruth\ndistribution", fontsize=8, ha="center")
            plt.axis("off")
            plt.subplot(6, pictrue_num, i + 2)
# =============================================================================
            cmap = rgb_cmap3()
            # 與第一版一致的色階範圍
            norm = mcolors.TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.11)


            gt_show = _to_scalar_field(Y[i + pic1 - 1], vmin, vmax)
            gt_show = np.nan_to_num(gt_show, nan=0.0, posinf=0.05, neginf=-0.05)
            gt_show = np.clip(gt_show, -0.05, 0.05)
            plt.imshow(np.flip(gt_show, 0), cmap=cmap, norm=norm, interpolation="none")
# =============================================================================
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")

# =============================================================================
        # # # ---- Pred ----
        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, pictrue_num + 1)
            plt.text(0.5, 0.15, "Predicted\njoint\ndistribution", fontsize=8, ha="center")
            plt.axis("off")

            plt.subplot(6, pictrue_num, i + pictrue_num + 2)

             # L2（sum）差異
            gt = _to_scalar_field(Y[i + pic1 - 1], vmin, vmax)
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.05, neginf=-0.05)
            gt = np.clip(gt, -0.05, 0.05)

            pr_show = _to_scalar_field(y_pred[i + pic1 - 1], vmin, vmax)
            pr_show = np.nan_to_num(pr_show, nan=0.0, posinf=0.05, neginf=-0.05)
            pr_show = np.clip(pr_show, -0.05, 0.05)

            plt.imshow(np.flip(pr_show, 0), cmap=cmap, norm=norm, interpolation="none")

            diff = np.sqrt(np.sum((gt - pr_show)**2) / (256 * 256))
            plt.text(245, 10, f"{diff:.1e}", fontsize=6, ha="right", va="top")

            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")

# =============================================================================         

                # ---- Absolute difference ----
        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, 2 * pictrue_num + 1)
            plt.text(0.5, 0.15, "Absolute\ndifferences", fontsize=8, ha="center")
            plt.axis("off")

            plt.subplot(6, pictrue_num, i + 2 * pictrue_num + 2)

            # 與原本 Pred 區塊一致：先轉成 scalar field，再做 nan / inf / clip
            gt = _to_scalar_field(Y[i + pic1 - 1], vmin, vmax)
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.05, neginf=-0.05)
            gt = np.clip(gt, -0.05, 0.05)

            pr_show = _to_scalar_field(y_pred[i + pic1 - 1], vmin, vmax)
            pr_show = np.nan_to_num(pr_show, nan=0.0, posinf=0.05, neginf=-0.05)
            pr_show = np.clip(pr_show, -0.05, 0.05)

            # absolute difference
            error_map = np.abs(gt - pr_show)

            # 避免 LogNorm 遇到 0 出錯
            positive_error = error_map[error_map > 0]
            if positive_error.size > 0:
                vmin_log = max(np.min(positive_error), 1e-3)
                vmax_log = max(np.max(error_map) * 1.2, vmin_log * 10)
                # plt.imshow(
                #     np.flip(error_map, 0),
                #     cmap="binary",
                #     norm=mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log),
                #     interpolation="none",
                # )

                im = plt.imshow(
                    np.flip(error_map, 0),
                    cmap="binary",
                    vmin=0,
                    vmax=vmax*0.3,
                    interpolation="none"
                )
            else:
                im = plt.imshow(
                    np.flip(error_map, 0),
                    cmap="binary",
                    interpolation="none",
                )



            total_diff = np.sum(error_map)
            plt.text(245, 10, f"{total_diff:.2e}", fontsize=6, ha="right", va="top", color="white")

            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            plt.axis("auto")
            
            # if i == pictrue_num - 2:
            #     pos = ax.get_position()
            #     cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.006, pos.height])
            #     fig.colorbar(im, cax=cax)
# ###--------------------------------------------------------------------------------------------------------------------------------------------------------------
            # === 三通道→標量；尺度對齊到 (256,256) 僅作檢查用 ===
            gt_scalar = _to_scalar_field(Y[i + pic1 - 1], vmin, vmax)   # (H,W) float32
            pr_scalar = _to_scalar_field(y_pred[i + pic1 - 1], vmin, vmax)

            # 與第一版一致：integrating2 內是 256×256 做積分；
            # 如果你想在這裡檢查/列印範圍，就把兩者都 resize 成 256×256 再比
            gt_chk = cv2.resize(gt_scalar, (256, 256))
            pr_chk = cv2.resize(pr_scalar, (256, 256))

            assert gt_chk.shape == pr_chk.shape == (256, 256), "Shapes do not match after resize!"
            print("GT range (after 256x256):", float(np.min(gt_chk)), float(np.max(gt_chk)))
            print("Pred range (after 256x256):", float(np.min(pr_chk)), float(np.max(pr_chk)))



# ###--------------------------------------------------------------------------------------------------------------------------------------------------------------

        # ---- x/p/u marginals ----
        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, 3 * pictrue_num + 1)
            plt.text(
                0.5, 0.15, "x-marginal\ndistributions\ncomparison", fontsize=8, ha="center"
            )
            plt.axis("off")

            plt.subplot(6, pictrue_num, i + 3 * pictrue_num + 2)
            cmap2 = ["black", "red"]
            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1 - 1][0:point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1 - 1][0:point],
                cmap2[1],
                linewidth=0.5,
            )

            diffx = np.mean(
                np.abs(
                    three_axis_gt[i + pic1 - 1][0:point]
                    - three_axis_pred[i + pic1 - 1][0:point]
                )
            )

            ax = plt.gca()
            ax.text(
                0.97, 0.95,
                f"{diffx:.1e}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="black"
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, 4 * pictrue_num + 1)
            plt.text(
                0.5, 0.15, "p-marginal\ndistributions\ncomparison", fontsize=8, ha="center"
            )
            plt.axis("off")

            plt.subplot(6, pictrue_num, i + 4 * pictrue_num + 2)
            cmap2 = ["black", "red"]
            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1 - 1][point:2*point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1 - 1][point:2*point],
                cmap2[1],
                linewidth=0.5,
            )

            diffy = np.mean(
                np.abs(
                    three_axis_gt[i + pic1 - 1][point:2*point]
                    - three_axis_pred[i + pic1 - 1][point:2*point]
                )
            )

            ax = plt.gca()
            ax.text(
                0.97, 0.95,
                f"{diffy:.1e}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="black"
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        for i in range(pictrue_num - 1):
            plt.subplot(6, pictrue_num, 5 * pictrue_num + 1)
            plt.text( 0.5, 0.15, "u-marginal\ndistributions\ncomparison", fontsize=8, ha="center" )
            plt.axis("off")

            plt.subplot(6, pictrue_num, i + 5 * pictrue_num + 2)
            cmap2 = ["black", "red"]
            plt.plot(
                np.arange(point),
                three_axis_gt[i + pic1 - 1][2*point:3*point],
                cmap2[0],
                linewidth=0.5,
            )
            plt.plot(
                np.arange(point),
                three_axis_pred[i + pic1 - 1][2*point:3*point],
                cmap2[1],
                linewidth=0.5,
            )


            diffu = np.mean(
                np.abs(
                    three_axis_gt[i + pic1 - 1][2*point:3*point]
                    - three_axis_pred[i + pic1 - 1][2*point:3*point]
                )
            )

            ax = plt.gca()
            ax.text(
                0.97, 0.95,
                f"{diffu:.1e}",
                transform=ax.transAxes,
                fontsize=6,
                ha="right",
                va="top",
                color="black"
            )
            plt.yticks(fontsize=7)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            plt.axis("auto")

        fig.tight_layout()

        prediction_filename = "test_visualization_Wigner"
        path_prediction = os.path.join(front_path, prediction_filename)
        if not os.path.exists(path_prediction):
            os.mkdir(path_prediction)

        fig.savefig(
            os.path.join(path_prediction, f"test_comparison_{save_tag}_{t}.png"),
            dpi=2000,
        )

        # 100 筆差`值輸出
        diff_values = {"diff": [], "diffx": [], "diffy": [], "diffu": []}
        N_eval = min(100, len(Y))

        for i in range(N_eval):
            # --- 2D diff：與畫圖完全一致 ---
            gt = _to_scalar_field(Y[i], vmin, vmax)
            gt = np.nan_to_num(gt, nan=0.0, posinf=0.05, neginf=-0.05)
            gt = np.clip(gt, -0.05, 0.05)

            pr_show = _to_scalar_field(y_pred[i], vmin, vmax)
            pr_show = np.nan_to_num(pr_show, nan=0.0, posinf=0.05, neginf=-0.05)
            pr_show = np.clip(pr_show, -0.05, 0.05)

            diff = np.sqrt(np.sum((gt - pr_show) ** 2) / gt.size)

            # --- marginal diff：與畫圖完全一致 ---
            diffx = np.mean(np.abs(
                three_axis_gt[i][0:point] - three_axis_pred[i][0:point]
            ))
            diffy = np.mean(np.abs(
                three_axis_gt[i][point:2*point] - three_axis_pred[i][point:2*point]
            ))
            diffu = np.mean(np.abs(
                three_axis_gt[i][2*point:3*point] - three_axis_pred[i][2*point:3*point]
            ))

            diff_values["diff"].append(diff)
            diff_values["diffx"].append(diffx)
            diff_values["diffy"].append(diffy)
            diff_values["diffu"].append(diffu)


        diff_csv_filename = os.path.join(front_path, "diff_values_wigner_one.csv")
        pd.DataFrame(diff_values).to_csv(diff_csv_filename, index=False)
        print(f"Diff values saved to {diff_csv_filename}")