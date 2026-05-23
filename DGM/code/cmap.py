import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt



def rgb_cmap(z, vmin, vmax): 
    z = np.clip(z, vmin, vmax)
    # normalization
    h = (z - vmin)/(vmax - vmin)
    zero = (0 - vmin)/(vmax - vmin)

    # color mapping
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    
    rgb = np.clip(cv2.merge([r, g, b]), -1, 1)
    rgb = np.array(rgb)
    
    return rgb

def rgb_cmap_inverse(rgb, vmin, vmax):
    e = pow(10, -4)
    rgb = np.clip(rgb, -1+e, 1-e)
    zero = (0 - vmin)/(vmax - vmin)
    r, g, b = cv2.split(rgb)
    
    # color mapping inverse
    g1_r = np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    g1_l = -np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    
    rk_r = (1.148/(np.exp(-25*(g1_r - (zero - 0.12))) + 1)/(np.exp(5*(g1_r - (zero + 0.45))) + 1))*2 - 1
    rk_l = (1.148/(np.exp(-25*(g1_l - (zero - 0.12))) + 1)/(np.exp(5*(g1_l - (zero + 0.45))) + 1))*2 - 1
    
    right = abs(rk_r - r).flatten()
    left = abs(rk_l - r).flatten()
    
    z = []      
    for i in range(np.prod(r.shape)):
    
        if right[i] > left[i] :
            z.append(g1_l.flatten()[i])
        elif right[i] <= left[i] :
            z.append(g1_r.flatten()[i])
            
    z = np.array(z).reshape((256, 256))  
    
    # normalization inverse
    denormal = z*(vmax - vmin) + vmin   
    
    return denormal

def rgb_cmap2(z, vmin, vmax):   
    z = np.clip(z, vmin, vmax)
    h = (z - vmin)/(vmax - vmin)
    zero = (0 - vmin)/(vmax - vmin)
    # h=h*255
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    rgb = np.clip(cv2.merge([r, g, b]), -1, 1)
    rgb = np.array(rgb)
    
    return rgb

def rgb_cmap_inverse2(rgb, vmin, vmax):
    e = pow(10, -4)
    
    # 確保 OpenCV 可以處理的資料格式 (float32)
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)  # ⭐ 轉換為 float32
        
    rgb = np.clip(rgb, -1+e, 1-e)
    zero = (0 - vmin)/(vmax - vmin)
    r, g, b = cv2.split(rgb)
    
    # ✅ 確保 g 不會有 NaN 或 Inf
    g = np.clip(g, 1e-6, 1-e)  # 避免 log(0) 和 sqrt(負數)
    
    g1_r = np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    g1_l = -np.sqrt(-np.log((g+1)/2)*(2*(0.14)**2)) + zero
    
    rk_r = (1.148/(np.exp(-25*(g1_r - (zero - 0.12))) + 1)/(np.exp(5*(g1_r - (zero + 0.45))) + 1))*2 - 1
    rk_l = (1.148/(np.exp(-25*(g1_l - (zero - 0.12))) + 1)/(np.exp(5*(g1_l - (zero + 0.45))) + 1))*2 - 1
    
    right = abs(rk_r - r).flatten()
    left = abs(rk_l - r).flatten()
    
    z = []
    for i in range(np.prod(r.shape)):
    
        if right[i] > left[i] :
            z.append(g1_l.flatten()[i])
        elif right[i] <= left[i] :
            z.append(g1_r.flatten()[i])
            
    z = np.array(z).reshape((256, 256))     
    denormal = z*(vmax - vmin) + vmin   
    
    return denormal

def rgb_cmap3(vmin = -0.25, vmax = 0.25, num_colors = 256):
    h = np.linspace(0.2, 0.8, num_colors)
    zero = (0 - vmin) / (vmax - vmin)
    r = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    b = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    rgb = np.clip(cv2.merge([r, g, b]), 0, 1)

    cmap = mcolors.ListedColormap(rgb, name='cmap', N=num_colors)

    return cmap

def rgb_cmap_inverse3(rgb, vmin, vmax):
    e = 1e-4
    rgb = np.clip(rgb, -1 + e, 1 - e)  # Ensure values are within the valid range
    zero = (0 - vmin) / (vmax - vmin)
    
    # Split the RGB channels
    r, g, b = cv2.split(rgb)
    
    # Apply transformations to compute g1_r and g1_l
    g_clipped = np.clip((g + 1) / 2, 1e-10, 1)  # Clip to avoid log errors
    g1_r = np.sqrt(-np.log(g_clipped) * (2 * (0.14) ** 2)) + zero
    g1_l = -np.sqrt(-np.log(g_clipped) * (2 * (0.14) ** 2)) + zero
    
    # Compute rk_r and rk_l
    rk_r = (1.148 / (np.exp(-25 * (g1_r - (zero - 0.12))) + 1) / (np.exp(5 * (g1_r - (zero + 0.45))) + 1)) * 2 - 1
    rk_l = (1.148 / (np.exp(-25 * (g1_l - (zero - 0.12))) + 1) / (np.exp(5 * (g1_l - (zero + 0.45))) + 1)) * 2 - 1
    
    # Vectorized comparison to determine z
    right = abs(rk_r - r)
    left = abs(rk_l - r)
    z = np.where(right > left, g1_l, g1_r)  # Use NumPy's `where` for element-wise selection
    
    # Rescale z to denormalize
    denormal = z * (vmax - vmin) + vmin
    
    return denormal

def rgb_cmap4(vmin = -0.25, vmax = 0.25, num_colors = 256):
    h = np.linspace(0.2, 0.8, num_colors)
    zero = (0 - vmin) / (vmax - vmin)
    b = 1.148/(np.exp(-25*(h - (zero - 0.12))) + 1)/(np.exp(5*(h - (zero + 0.45))) + 1)*2 - 1
    g = np.exp(-(h - zero)**2/(2*(0.14**2)))*2 - 1
    r = 1.148/(np.exp(25*(h - (zero + 0.12))) + 1)/(np.exp(-5*(h - (zero - 0.45))) + 1)*2 - 1
    rgb = np.clip(cv2.merge([r, g, b]), 0, 1)

    cmap = mcolors.ListedColormap(rgb, name='cmap', N=num_colors)

    return cmap

