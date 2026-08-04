
import sys
sys.path.append('')

import math
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import json
import time

import Code.Function as MF
from Code.Load_State import DataLoad
from Code.ClassicalModel import VAE_ForDiffusionModel, VAELoss
from Code.ClassicalModel import ResidualBlock as ResNet_Block, LayerNorm2d


import torch
import torch.nn as nn                      
from torch.nn import functional as F        
from torchvision import datasets, transforms  
import torch.utils.data as dset                 
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary

    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 1. 自注意力：圖像自己看自己
        self.attn1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # 2. 交叉注意力：圖像看文本提示 (context)
        self.attn2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # 3. 前饋網路
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context):
        # x: (Batch, Channels, H, W) -> 需要轉為 (Batch, Sequence, Dim)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1) # (B, H*W, C)
        
        # Self-Attention
        res = x_flat

        x_flat, _ = self.attn1(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + res)
        
        # Cross-Attention (與文本 Embedding 互動)
        res = x_flat
        x_flat, _ = self.attn2(x_flat, context, context)
        x_flat = self.norm2(x_flat + res)
        
        # Feed Forward
        x_flat = self.norm3(x_flat + self.ff(x_flat))
        
        # 轉回 (Batch, Channels, H, W)
        return x_flat.permute(0, 2, 1).view(b, c, h, w)
    
class TransformerStack(nn.Module):
    def __init__(self, dim, num_layers=2, heads=8):
        super().__init__()
        # 建立 10 個獨立的 TransformerBlock 實例
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads) 
            for _ in range(num_layers)
        ])

    def forward(self, x, context):
        # 依序通過這 10 層 (殘差連接通常在 Block 內部處理了)
        for layer in self.layers:
            x = layer(x, context)
        return x

class UNet_model(nn.Module):
    def __init__(self, side_dim, channel_list = [3, 9, 18, 36, 72], diff_step = 1000, point = 721, num_frequency_encoding = -1, CircultType = ""):
        super(UNet_model, self).__init__()
        self.channel_list = channel_list
        self.side_dim = side_dim
        self.CircultType = CircultType
        self.num_frequency_encoding = num_frequency_encoding

        # embedding
        # time embedding
        self.t_emb = nn.Sequential(nn.Linear(channel_list[-1], channel_list[-1]*2))

        # marginal embedding
        self.l_emb_list = nn.ModuleList() 
        self.l_emb_list.append(nn.Sequential(nn.Linear(point, 2048), nn.SiLU(),
                                             nn.Linear(2048, 2048), nn.SiLU(),
                                             nn.Linear(2048, 512), nn.SiLU()))
        
        for i in range(3, len(channel_list)):
            if self.num_frequency_encoding != -1:
                self.l_emb_list.append(nn.Linear(self.num_frequency_encoding * 2 * 3, channel_list[i]))
            else:
                self.l_emb_list.append(nn.Linear(3, channel_list[i]))

        # DDPM superparameter
        self.diff_step = diff_step
        self.beta_s = torch.linspace(0.0001, 0.02, diff_step).to(device)
        self.alpha_s = 1 - self.beta_s
        self.alpha_bar_t = torch.cumprod(self.alpha_s, dim = 0)
        self.sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t)
        self.sqrt_alpha_bar_1_t = torch.sqrt(1.0 - self.alpha_bar_t)

        # --- Unet ---
        self.encoder = nn.ModuleList()
        self.encoder_t = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.decoder_t = nn.ModuleList()

        # channel change: 1 -> 64
        self.first = ResNet_Block(channel_list[0], channel_list[1], emb_channels=channel_list[-1]*2, num_groups=1)

        # channel change: 64 -> 96 -> 128 -> 196
        for i in range(1, len(channel_list)-1):
            self.encoder.append(ResNet_Block(channel_list[i], channel_list[i+1], emb_channels=channel_list[-1]*2))
            if i != 1:
                self.encoder_t.append(TransformerStack(dim=channel_list[i+1], num_layers = 2, heads = 8))
                self.encoder_t.append(TransformerStack(dim=channel_list[i+1], num_layers = 2, heads = 8))
            self.encoder.append(ResNet_Block(channel_list[i+1], channel_list[i+1], emb_channels=channel_list[-1]*2))
            self.encoder.append(nn.Conv2d(channel_list[i+1], channel_list[i+1], kernel_size= 2, stride=2))

        # channel change: 196 -> 196*2 -> 196
        self.center_t = TransformerStack(dim = channel_list[-1], num_layers = 5, heads = 8)
        self.center1 = ResNet_Block(channel_list[-1], channel_list[-1]*2, emb_channels=channel_list[-1]*2)
        self.center2 = ResNet_Block(channel_list[-1]*2, channel_list[-1], emb_channels=channel_list[-1]*2)

        # channel change: 196+(196) -> 128+(128) -> 18+(18) -> 9+(9) -> 3
        for i in range(1, len(channel_list)-1):
            self.decoder.append(ResNet_Block(channel_list[-i]*2, channel_list[-i], emb_channels=channel_list[-1]*2))
            if i <= 2:
                self.decoder_t.append(TransformerStack(dim=channel_list[-i], num_layers = 2, heads = 8))
                self.decoder_t.append(TransformerStack(dim=channel_list[-i], num_layers = 2, heads = 8))
            self.decoder.append(ResNet_Block(channel_list[-i]*2, channel_list[-i], emb_channels=channel_list[-1]*2))
            self.decoder.append(nn.Conv2d(channel_list[-i], channel_list[-(i+1)], kernel_size=1))

        self.final = nn.Sequential(nn.Conv2d(channel_list[1], channel_list[1], kernel_size=5, padding=2),
                                   LayerNorm2d(channel_list[1]),
                                   nn.ReLU(),
                                   nn.Conv2d(channel_list[1], channel_list[1], kernel_size=3, padding=1),
                                   LayerNorm2d(channel_list[1]),
                                   nn.ReLU(),
                                   nn.Conv2d(channel_list[1], channel_list[0], kernel_size=3, padding=1))
        
        self.noise_sample = torch.randn((1, self.side_dim, self.side_dim)).to(device)

    def forward(self, x, t, l):
        # x -> [batch_size, channel_dim, h, w]
        # t -> [batch_size, 1] 時間維度，數值在 (0~1) 之間，代表目前圖片的 noise 程度
        # l -> [batch_size, 3, point] 圖片標籤
        t_emb = self.sinusoidal_embedding(t, self.channel_list[-1])
        t_emb = self.t_emb(t_emb)

        # Model 內部
        # l.shape: [batch_size, 3, point]
        if self.num_frequency_encoding != -1:   # 如果 self.num_frequency_encoding 選擇不是 -1 的話
            # l.shape: [batch_size, 3, point] -> [batch_size, 3*2*self.num_frequency_encoding, point]
            l = MF.fourier_features_multi(l, num_freqs = self.num_frequency_encoding)
        # 線性層
        l = self.l_emb_list[0](l)

        l = l.permute(0, 2, 1)

        l_to = []
        for i in range(2):
            l_to.append(self.l_emb_list[1+i](l))    # convolution 128 196

        # Unet 部分
        s = []
        # Down
        # x -> [batch_size, channel_dim_1, h_1, w_1]
        # t -> [batch_size, 1]
        # l -> [batch_size, 12, emb_dim] 
        x = self.first(x, t_emb)
        for i in range(len(self.encoder)//3):
            x = self.encoder[i*3](x, t_emb)  # convolution 
            s.append(x)

            if i != 0:
                x = self.encoder_t[(i-1)*2](x, l_to[i-1])   

            x = self.encoder[i*3+1](x, t_emb)  # convolution 
            s.append(x)

            if i != 0:
                x = self.encoder_t[(i-1)*2+1](x, l_to[i-1])   

            x = self.encoder[i*3+2](x)  # pooling

        # Center
        # x -> [batch_size, channel_dim_n, h_n, w_n]
        # t_emb -> [batch_size, channel_dim_n, 1, 1]
        x = self.center_t(x, l_to[-1])
        x = self.center1(x, t_emb) + t_emb[..., None, None]             # 加入時間維度(圖片加入noise的程度)
        x = self.center2(x, t_emb)

        # Up
        for i in range(len(self.decoder)//3):
            x = F.interpolate(x, scale_factor=2, mode="nearest")

            x = torch.concat((x, s[-(i*2+1)]), dim = 1) # s: -1, -3, -5
            x = self.decoder[i*3](x, t_emb)          # convolution 

            if i <= 1:
                x = self.decoder_t[i*2](x, l_to[1-i])

            x = torch.concat((x, s[-(i+1)*2]), dim = 1) # s: -2, -4, -6
            x = self.decoder[i*3+1](x, t_emb)

            if i <= 1:
                x = self.decoder_t[i*2+1](x, l_to[1-i])
            x = self.decoder[i*3+2](x)

        return self.final(x)
    
    def sinusoidal_embedding(self, t, dim):
        half = dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * -(np.log(10000) / half))
        emb = t * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def generatr_noise(self, num, fixed_noise = False):
        if fixed_noise == True:
            noise = [self.noise_sample for i in range(num)]
            noise = torch.stack(noise, dim=0)
        else:
            noise = torch.randn((num, 1, self.side_dim, self.side_dim)).to(device)
        return noise
    
    def forward_diffusion(self, x0, t, noise, noise_fixed = None):
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_t[t][:, None, None, None]
        sqrt_alpha_bar_1_t = self.sqrt_alpha_bar_1_t[t][:, None, None, None]

        return sqrt_alpha_bar_t * x0 + sqrt_alpha_bar_1_t * noise

# -----------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------

def generate_image_DDPM(model : UNet_model, num, l = [[0]], x_t = None):
    h = model.side_dim
    w = model.side_dim
    image_all = np.zeros((h*2, w*10, 3), dtype=np.float32)
    with torch.no_grad():
        if x_t is None:
            x_t = model.generatr_noise(num, fixed_noise = True)
            init_noise = x_t

        for i in tqdm(range(model.diff_step)):
            t = model.diff_step - i - 1                                                            
            t_input = torch.tensor([t for _ in range(num)]).view(-1, 1).to(torch.float32).to(device) / diff_step
            l_input = l.to(torch.float32).to(device)
            predited_noise = model(x_t, t_input, l_input)
            # 有頂線的是 alpha_t，沒有頂線的是 alpha_s
            x_t = (1 / torch.sqrt(model.alpha_s[t])) * (x_t - ((1 - model.alpha_s[t])/ model.sqrt_alpha_bar_1_t[t]) * predited_noise)

            if t != 0:
                x_t += torch.sqrt(model.beta_s[t]) * torch.randn_like(x_t) * eta
            #print(x_t.shape)
            #image_all[h*(i//10):h*(i//10+1),w*(i%10):w*(i%10+1)] = torch.clip((x_t+1)/2, 0, 1)[0,0,:,:,None].cpu().numpy()

        #image_all = cv2.cvtColor(image_all*255, cv2.COLOR_RGB2BGR)
        #w_scale = 1600/(w*10)
        #image_all = cv2.resize(image_all, (int(w * 10 * w_scale),int(2 * w * w_scale)), interpolation=cv2.INTER_NEAREST)
        #cv2.imwrite(os.path.join("./DiffusionGenerateImage", "Step_test", f"MNIST_DDPM_{add_name}.png"), image_all)

        return x_t.squeeze(), init_noise

#　TODO: step 問題
def show_image(model, decoder_model, data, d3_dis, file_path, epoch_now, show_num = None, label = None):
    if os.path.exists(image_folder) == False:
        os.mkdir(image_folder)

    h = side_dim
    w = side_dim
    if "MI_P3C" in model.CircultType:
        vmin, vmax = -1.05, 1.05
    elif "MI_W3C" in model.CircultType:
        vmin, vmax = -0.45, 0.45
    else:
        vmin, vmax = data.min(), data.max()
    vmin, vmax = -0.15, 0.15

    
    # --- DDPM Part ---
    if show_num is not None:
        selected = [len(d3_dis)//show_num*i for i in range(show_num)]
        cond = d3_dis[selected].to(device)
        GT = data[selected].view(len(selected), side_dim, side_dim)
    
        predict, init_noise = generate_image_DDPM(model, num = show_num, l = cond)
        if decoder_model is not None:
            with torch.no_grad():
                predict = decoder_model.decoder(predict.unsqueeze(1) * range_scale).squeeze(1).cpu().numpy()
    else:
        cond = d3_dis.to(device)
        GT = data.view(len(data), side_dim, side_dim)
        predict = []
        for i in tqdm(range(15)):
            predict_out, init_noise = generate_image_DDPM(model, num = 10, l = cond[i*10:(i+1)*10])
            if decoder_model is not None:
                with torch.no_grad():
                    predict_out = decoder_model.decoder(predict_out.unsqueeze(1) * range_scale).squeeze(1).cpu().numpy()
            predict = predict + [predict_out]
   
        predict = np.concat(predict, axis=0)
    cond = cond.cpu().numpy()
    GT = GT.cpu().numpy()

    # Make sure that the dim of predict is [batch_size, side_sim, side_sim]
    if len(predict.shape) == 2:
        predict = predict.reshape(len(selected), side_dim, side_dim)
    predict = np.clip(predict, -1, 1)


    # --- Result(image show) ---
    if show_num is None:
        show_num = 10
    image_all = np.zeros((h*3, w*show_num, 3), dtype=np.float32)

    # Show Result - GT
    for i in range(show_num):
        if "MI_MNIST" in CircultType:
            to = (GT[i].reshape(side_dim, side_dim)+1)/2
            image_all[0:h, w*i:w*(i+1)] = np.stack([to, to, to], axis=2)
        else:
            image_all[0:h, w*i:w*(i+1)] = MF.rgb_cmap(GT[i], vmin, vmax)

    # Show Result - Predict   
    for i in range(show_num):
        if "MI_MNIST" in CircultType:
            to = (predict[i]+1)/2
            image_all[h:h*2, w*i:w*(i+1)] = np.stack([to, to, to], axis=2)
        else:
            image_all[h:h*2, w*i:w*(i+1)] = MF.rgb_cmap(np.clip(predict[i], vmin, vmax), vmin, vmax)

    print(f"GT   : min: {GT.min()}, max: {GT.max()}, mean: {GT.mean()}, var: {GT.var()}")
    print(f"predi: min: {predict.min()}, max: {predict.max()}, mean: {predict.mean()}, var: {predict.var()}")

    # Show Result - Noise
    for i in range(show_num):
        to = ((init_noise[i]+1)/2).squeeze().cpu().numpy()
        to = np.clip(to, 0, 1)
        if to.shape[0] != side_dim:
            to = cv2.resize(to, (side_dim, side_dim), interpolation=cv2.INTER_NEAREST)
        image_all[h*2:h*3, w*i:w*(i+1)] = np.stack([to, to, to], axis=2)
        

    image_all = np.clip(image_all,0,1)
    image_all = cv2.cvtColor(image_all*255, cv2.COLOR_RGB2BGR)

    w_scale = 1600/(w*show_num)
    image_all = cv2.resize(image_all, (int(w * show_num * w_scale),int(h*3 * w_scale)), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(os.path.join(file_path, f"MNIST_DDPM_{epoch_now}.png"), image_all)


    # 顯示整體的比較表 GT -> Predict -> x_marginal -> y_marginal -> u_marginal
    # predict_marginal = MF.Back2MDx3(predict, vmin, vmax, point=point) * 0.642   # -> [3, point]
    save_path = os.path.join(file_path, f"MNIST_DDPM_plt_{epoch_now}.png")
    GT_marginal = cond
    predict_marginal = MF.Back2MDx3(predict, vmin, vmax, point=points)
    MF.CompareJDAndMDFig(GT[0:show_num], predict[0:show_num], GT_marginal[0:show_num], predict_marginal[0:show_num], 
                         save_path, vmin, vmax, point=points, label=label)

    MF.CompareOT(GT, predict, cond, predict_marginal, vmin, vmax, point=points)

def VAE_show_image(model, data, d3_dis, file_path, epoch_now):
    h = side_dim
    w = side_dim
    #vmin, vmax = -1.05, 1.05    # For generate data
    vmin, vmax, step = -0.45, 0.45, 4.5    # For ["harmonic","squeezed","coherent","cat"]
    #vmin, vmax = -0.2, 0.2       # For TJCM

    l = [len(d3_dis)//10*i for i in range(10)]
    cond = d3_dis[l].view(len(l),-1).cpu().numpy()
    GT = data[l].to(device)

    # DDPM
    image_all = np.zeros((h*2, w*10, 3), dtype=np.float32)
    with torch.no_grad():
        predict, latent, _, _ = model(GT, V = False)
    latent = F.interpolate(latent, scale_factor=4, mode="nearest").cpu().numpy()

    GT = GT.view(len(l), side_dim, side_dim).cpu().numpy()
    predict = predict.view(len(l), side_dim, side_dim).cpu().numpy()
    for i in range(10):
        image_all[0:h, w*i:w*(i+1)] = MF.rgb_cmap(GT[i], vmin, vmax)

    for i in range(10):
        to = np.clip(latent[i,0] + 0.5, 0, 1)
        image_all[h:h*2, w*i:w*(i+1)] = np.stack([to,to,to], axis=-1) / range_scale

    GT_marginal = cond
    predict_marginal = MF.Back2MDx3(predict, vmin, vmax, step = step, point=points)
    MF.CompareJDAndMDFig(GT, predict, GT_marginal, predict_marginal, 
                         os.path.join(file_path, f"VAE_DDPM_plt_{epoch_now}.png"), vmin, vmax, point=points)
    
    image_all = np.clip(image_all,0,1)
    image_all = cv2.cvtColor(image_all*255, cv2.COLOR_RGB2BGR)

    w_scale = 1600/(w*10)
    image_all = cv2.resize(image_all, (int(w*10 * w_scale),int(h*2 * w_scale)), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(file_path, f"MNIST_DDPM_{epoch_now}.png"), image_all)


# -----------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------

# 訓練模型
def Training(model : UNet_model, data, optimizer, scheduler, diff_step, device, now_steps = 0):
    total_loss = 0      # for optimizer
    total_num = 0       # for optimizer
    loss_function = nn.MSELoss().to(device)     # define loss function
    now_steps = now_steps
    while 1:
        for j, loader in enumerate(tqdm(data.trainLoader)):          
            x = loader[0].to(device).to(dtype=torch.float32)  
            condition = loader[1].to(device) 
            
            batch_size = len(x)

            t = torch.randint(low=0, high=diff_step, size=[batch_size]).to(device)
            # data.shape -> [batch_size, channels, h, w]
            # noise -> [batch_size, channels, h, w]
            noise = model.generatr_noise(len(x))
            #noise_fixed = model.generatr_noise(len(x), fixed_noise = True)
            x_t = model.forward_diffusion(x, t, noise)

            # Forward propagation
            noise_predit = model(x_t, t.view(-1, 1).to(torch.float32) / diff_step, condition)
            loss = loss_function(noise_predit, noise)
    
            # Backward propagation
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()       
            scheduler.step()
            total_loss += loss.detach().cpu()
            total_num += 1


            if now_steps % 10000 == 0:
                show_image(UNet, VAE, data.test_data, data.test_3_dis, image_folder, now_steps, show_num = 10)
                SaveModel(UNet, optimizer, scheduler, save_path, now_steps)

            now_steps += 1
            if now_steps > total_steps:
                return
            
        print(f"step: {now_steps}, loss: {total_loss/total_num}")    
        print()
    
# 訓練 VAE
def TrainingVAE(model, data, optimizer, scheduler):
    total_loss = 0
    loss_function = VAELoss(kl_weight=0)
    for j, loader in enumerate(tqdm(data)):  
        #with torch.amp.autocast("cuda:0", enabled=torch.cuda.is_available()):
        x = loader[0].to(device)

        # Forward propagation
        recon, latent, mu, logvar = model(x, V = False)
        loss = loss_function(recon, x, mu, logvar) + 0.01 * (1-latent.var()) + 0.01 * abs(latent.mean())

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu()
        if j % 100 == 0:
           print(f"VAELoss {j} iteration: {loss.detach().cpu() / len(x)}")

        if (j+1) % (len(data)//3) == 0:
            total_loss = total_loss / len(data)
            scheduler.step(total_loss)
            print(total_loss/len(data), latent.min().item(), latent.max().item(), latent.mean().item(), latent.var().item())

# save model
def SaveModel(UNet, optimizer, scheduler, save_path, now_steps = 0):
    print("save file at:",{save_path})
    torch.save({"model": UNet.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "noise": UNet.noise_sample,
            "now_steps": now_steps}, save_path)

# load model
def LoadModel(UNet, optimizer, scheduler, save_path):
    print(f"Model will save on {save_path}")
    if os.path.exists(save_path): 
        print("Find Exist Model, Loading")
        checkpoint = torch.load(save_path, weights_only=False)
        UNet.load_state_dict(checkpoint["model"]) 
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        now_steps = checkpoint['now_steps']
    else:
        now_steps = 0
    return now_steps

# -----------------------------------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------------------------------

class DataClass():
    def __init__(self, train_data, train_3_dis, test_data, test_3_dis, batch_size):
        self.batch_size = batch_size

        len_train = len(train_data)
        indices = torch.randperm(len_train)
        ct = int(len_train*0.6)
        self.train_data = train_data[indices[0:ct]]
        self.train_3_dis = train_3_dis[indices[0:ct]]

        self.val_data = train_data[indices[ct:]]
        self.val_3_dis = train_3_dis[indices[ct:]]

        self.test_data = test_data
        self.test_3_dis = test_3_dis
        print(f"Train shape: {self.train_data.shape}")
        print(f"Valid shape: {self.val_data.shape}")
        print(f"Test  shape: {self.test_data.shape}")
    
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None

    def Make_TrainLoader(self, VAE = None):
        if VAE is None:
            self.trainLoader = dset.DataLoader(dset.TensorDataset(self.train_data, self.train_3_dis), batch_size=batch_size, shuffle=True)
        else:
            train_data_s = self.DataToLatentSpace(VAE, self.train_data)
            self.trainLoader = dset.DataLoader(dset.TensorDataset(train_data_s.to(device), self.train_3_dis.to(device)), batch_size=batch_size)

    def DataToLatentSpace(self, VAE, data):
        data_save = []
        data_len = len(data)
        bc = 64
        with torch.no_grad():
            for i in tqdm(range(int(np.ceil(data_len/bc)))):
                _, latent, _, _ = VAE(data[i*bc:min((i+1)*bc, data_len)].to(device), V = False)
                data_save.append(latent)
        return torch.cat(data_save, dim=0) / range_scale


# --- Training Setup --- 
side_dim = 256
VAE_epoch = 20
points = 2241
batch_size = 8

num_of_data = 5000

ModelType = "Marginal"               # "Marginal", "Joint"     #-----------------
time_emb = False
CircultType = "MI_W3C"              # MI_W3C, MI_P3C, MI_MNIST_HEA, MI_MNIST_ALT
GenerateLayers = 38                                             #-----------------
Learning_rate = 0.00005


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Diffusion Setup --- 
diff_step = 1000
eta = 0           # add eta*noise at each step
range_scale = 1
num_frequency_encoding = 6
warmup_steps = 10000
total_steps = 500000
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


# --- Data load info ---
image_label = "01"
basic_folder = "QGAN"
Q_type = ["TJCM"]   #  ["harmonic","squeezed","coherent","cat"]


# --- Save dirt ---
add_name = f"{side_dim}x{side_dim}_{CircultType}_{GenerateLayers}"
if time_emb == True:
    add_name += f"_T_emb"
if "3C" in CircultType:
    add_name += f"_{points}"
if "MNIST" in CircultType:
    add_name += f"_{image_label}"
if "W3C" in CircultType:
    add_name += f"_{'+'.join(Q_type)}"
if num_of_data != None:
    add_name += f"_NofD{num_of_data}"
add_tage = "test"
add_name += add_tage                    # ------------------------ add tage -------------------------
save_path =  f"./{basic_folder}/DiffusionModel/{add_name}.pt"
image_folder = f"./{basic_folder}/DiffusionGenerateImage/{add_name}"
if os.path.exists(image_folder) == False:
    os.mkdir(image_folder)

use_amp = torch.cuda.is_available() 
scaler = torch.amp.GradScaler("cuda:0", enabled=use_amp) # 自動混合精度 (Automatic Mixed Precision, AMP)

UseVAE = True
TrainMode = True
if __name__ == "__main__":
    if TrainMode:
        # --- Loading Data ---
        # train_data, train_3_dis, test_data, test_3_dis
        data = DataClass(*DataLoad(side_dim, num_of_data, CircultType, Q_type, image_label=image_label, point=points), batch_size=batch_size)

    # --- Check if VAE Model Exists, if not, training VAE Model ---
    if UseVAE:
        VAE = VAE_ForDiffusionModel(in_channels=1, latent_channels=1, initial_channels=64).to(device)
        VAE_save_path =  f"./{basic_folder}/DiffusionModel/VAE_{add_name}.pt"
        VAE_image_folder = f"./{basic_folder}/DiffusionGenerateImage/VAE_{add_name}"
        if os.path.exists(VAE_image_folder) == False:
            os.mkdir(VAE_image_folder)

        summary(VAE)
        print(VAE_save_path)
        if os.path.exists(VAE_save_path) == True:
            print("Find VAE model, Loading...")
            checkpoint = torch.load(VAE_save_path, weights_only=False)
            VAE.load_state_dict(checkpoint["model"])
            # VAE_show_image(VAE, data.test_data, data.test_3_dis, VAE_image_folder, VAE_epoch)
        else:
            print("Trining VAE model...")
            optimizer_VAE = optim.AdamW(VAE.parameters(), lr=0.0005)
            scheduler_VAE = ReduceLROnPlateau(optimizer_VAE, mode="min", factor=0.5, patience=3, min_lr=1e-7)
            trainLoader = dset.DataLoader(dset.TensorDataset(data.train_data, data.train_3_dis), batch_size=batch_size, shuffle=True)
            for i in range(VAE_epoch):
                print(f"Train epoch {i}")
                if i % 5 == 0:
                    VAE_show_image(VAE, data.test_data, data.test_3_dis, VAE_image_folder, i)
                    torch.save({"model":VAE.state_dict(),}, VAE_save_path.replace(".pt",f"_{i}.pt"))
                TrainingVAE(VAE, trainLoader, optimizer_VAE, scheduler_VAE)
                torch.save({"model":VAE.state_dict(),}, VAE_save_path)
    else:
        VAE = None

    # --- Diffusion Part ---
    # diffusion model 
    UNet = UNet_model(64, channel_list = [1, 64, 96, 128, 192], diff_step = diff_step, point=points, num_frequency_encoding=num_frequency_encoding, CircultType=CircultType).to(device)
    summary(UNet)

    # diffusion model - optimizer
    optimizer = optim.AdamW(UNet.parameters(), lr=Learning_rate, weight_decay=0.001)

    # diffusion model - scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # load diffusion model
    now_steps = LoadModel(UNet, optimizer, scheduler, save_path)


    # --- Training and Testing part ---
    if TrainMode:
        data.Make_TrainLoader(VAE)

        # 訓練模型
        print(data.train_data.min(), data.train_data.max(), data.train_data.mean(), data.train_data.var())
        print(f"start at: {now_steps}")

        Training(UNet, data, optimizer, scheduler, diff_step, device, now_steps)
        SaveModel(UNet, optimizer, scheduler, save_path, total_steps)

        # Show Final Result
        show_image(UNet, VAE, data.test_data, data.test_3_dis, image_folder, total_steps, 10)

    # --- Test Result ---
    if TrainMode == False:
        test_data = torch.tensor(np.load(os.path.join("npy_quantum", f"distribution_AllCM", f"y_test.npy"))).squeeze().unsqueeze(1)
        test_3_dis = torch.tensor(np.load(os.path.join("npy_quantum", f"distribution_AllCM", f"x_test.npy")))
        test_3_dis = test_3_dis.view(len(test_3_dis), 3, points)

        CM_label = np.load(os.path.join("npy_quantum", f"distribution_AllCM", f"test_label.npy"))
        show_image(UNet, VAE, test_data, test_3_dis, image_folder, total_steps, label = CM_label)

