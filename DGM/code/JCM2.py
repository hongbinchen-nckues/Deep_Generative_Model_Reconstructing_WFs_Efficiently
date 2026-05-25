import numpy as np
import random
import os
from tqdm import trange
import cv2
from scipy.special import laguerre, hermite, eval_hermite, factorial, eval_laguerre,eval_genlaguerre
from scipy.integrate import  simpson
from scipy.special import genlaguerre




def build_data(dataset_filename, types, num, resolution, length):

    # set save path of dataset
    osd = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))

    if types == 'Entanglement':
        data_type_filename = "distribution_Entanglement"
    if types == 'TJCM':
        data_type_filename = "new_distribution_TJCM"
    if types == 'JCM':
        data_type_filename = "distribution_JCM"

    main_distribution_npy_filename = 'main_data'
    x_distribution_filename = 'x_data'
    y_distribution_filename = 'y_data'
    u_distribution_filename = 'u_data'

    path = os.path.join(osd, dataset_filename, data_type_filename)
    
    path_main_data_npy = os.path.join(osd, dataset_filename, data_type_filename, main_distribution_npy_filename)
    path_x_data = os.path.join(osd, dataset_filename, data_type_filename, x_distribution_filename)
    path_y_data = os.path.join(osd, dataset_filename, data_type_filename, y_distribution_filename)
    path_u_data = os.path.join(osd, dataset_filename, data_type_filename, u_distribution_filename)

    print("Data save in {path}".format(path=path))

    path_list = [os.path.join(osd, dataset_filename), path,
                 path_main_data_npy, path_x_data, path_y_data, path_u_data]

    for path in path_list:
        if os.path.exists(path) == False:
            os.mkdir(path)
    kx = np.arange(-length, length + resolution, resolution)
    ky = np.arange(-length, length + resolution, resolution)
    ku = np.arange(-length * np.sqrt(2), (length + resolution) * np.sqrt(2), resolution * np.sqrt(2))
    kv = np.arange(-length * np.sqrt(2), (length + resolution) * np.sqrt(2), resolution * np.sqrt(2))
    U, V = np.meshgrid(ku, kv)
    x, y = np.meshgrid(kx, ky)

    # # Starting number setting
    start_index = 57000
    for e in trange(num):
        index = start_index + e  # True archive index
          
    # ============================================================================================
    
        if types == 'Entanglement':
            # ----------------- 參數設定 -----------------
            q = random.randint(0,4)          # 光子數差 (n1 - n2)
            zeta = random.uniform(1, 5.0)      # PCS 強度參數
            theta = np.random.uniform(0, np.pi/2)     # 原子初始角度
            theta2 = 0.0
            g = 1
            N_max = 20      

            # 隨機時間 T
            T_val = random.randint(0, 315) / 100.0  

            # ----------------- PCS 相關係數 -----------------
            def calc_Nq(q, zeta, cutoff):
                """計算歸一化常數 Nq (Eq. 3)"""
                val = 0.0
                for n in range(cutoff):
                    val += (zeta**(2*n)) / (factorial(n) * factorial(n + q))
                return 1.0 / np.sqrt(val)

            def calc_Cn(n, q, zeta, Nq):
                """計算 PCS 係數 Cn (Eq. 6 下方)"""
                denom = np.sqrt(factorial(n) * factorial(n + q))
                return Nq * (zeta**n) / denom

            # ----------------- 動力學係數 X1~X4 -----------------
            def get_X_coeffs(T, n, q, theta):
                """計算動力學係數 X1, X2, X3, X4 (對稱情形 g=1)"""
                # alpha_n (Eq. 22, g=1)
                alpha_n = 4*np.sqrt((n + 1) * (n + 2) * (n + q + 1) * (n + q + 2))
                
                # eta (Rabi frequency, Eq. 25)
                eta = np.sqrt(2 * (n + 2) * (n + q + 2) + 2 * (n + 1) * (n + q + 1))
                
                if eta == 0:
                    return 0.0, 0.0, 0.0, 0.0

                # X1 (Eq. 23)
                c1 = 2 * np.cos(T * eta) / (eta**2)
                c2 = 2 / (eta**2)
                term1 = (n + 1) * (n + q + 1) * np.cos(theta) + 0.25 * alpha_n * np.sin(theta)
                term2 = (n + 2) * (n + q + 2) * np.cos(theta) - 0.25 * alpha_n * np.sin(theta)
                X1 = c1 * term1 + c2 * term2

                # X2, X3 (Symmetric, Eq. 24)
                c_sin = -np.sin(T * eta) / eta
                term_x2 = (
                    np.sqrt((n + 1) * (n + q + 1)) * np.cos(theta)
                    + np.sqrt((n + 2) * (n + q + 2)) * np.sin(theta)
                )
                X2 = c_sin * term_x2
                X3 = X2

                # X4 (Eq. 24 below)
                term1_x4 = (n + 2) * (n + q + 2) * np.sin(theta) + 0.25 * alpha_n * np.cos(theta)
                term2_x4 = (n + 1) * (n + q + 1) * np.sin(theta) - 0.25 * alpha_n * np.cos(theta)
                X4 = c1 * term1_x4 + c2 * term2_x4

                return X1, X2, X3, X4

            # ----------------- Wigner function 主體 A(x,y) -----------------
            def distribution_atom(x, y, theta, T, zeta, q, N_max):
                """
                輸入：x, y = meshgrid(kx, ky)
                輸出：A(x, y)，shape 跟 x/y 一樣
                """
                # |alpha|^2 = x^2 + y^2
                Alpha_sq = x**2 + y**2

                # 先算出 P(k) 陣列
                P_k = np.zeros(N_max + 5)
                Nq_val = calc_Nq(q, zeta, N_max)

                for n in range(N_max):
                    Cn_val = calc_Cn(n, q, zeta, Nq_val)
                    X1, X2, X3, X4 = get_X_coeffs(T, n, q, theta)
                    prob = np.abs(Cn_val)**2

                    # k = n (來自 |e1, e2>)
                    if n < len(P_k):
                        P_k[n] += prob * np.abs(X1)**2
                    
                    # k = n + 1 (來自 |e1, g2>, |g1, e2>)
                    if n + 1 < len(P_k):
                        P_k[n+1] += prob * (np.abs(X2)**2 + np.abs(X3)**2)
                        
                    # k = n + 2 (來自 |g1, g2>)
                    if n + 2 < len(P_k):
                        P_k[n+2] += prob * np.abs(X4)**2

                # 初始化 Wigner Matrix
                A = np.zeros_like(Alpha_sq, dtype=np.float64)

                # 根據公式累加: W = (2/pi) * exp(-2|alpha|^2) * sum [ P(k) * (-1)^k * Lk(4|alpha|^2) ]
                prefactor = (2.0 / np.pi) * np.exp(-2.0 * Alpha_sq)

                for k_idx in range(len(P_k)):
                    if P_k[k_idx] > 1e-10:  # 只計算有顯著機率的項以加速
                        Lk_val = genlaguerre(k_idx, 0)(4.0 * Alpha_sq)
                        sign = 1.0 if (k_idx % 2 == 0) else -1.0
                        A += P_k[k_idx] * sign * Lk_val

                return A * prefactor

            # =====================================================
            # 1. 計算 Wigner 分佈矩陣 A(x,y)
            #    這裡用的是你固定好的 meshgrid：x, y = np.meshgrid(kx, ky)
            # =====================================================
            A_matrix = distribution_atom(x, y, theta, T_val, zeta, q, N_max)

            # main_data：存 256×256 的版本
            A_main = cv2.resize(A_matrix, (256, 256))

            # =====================================================
            # 2. 計算邊緣分佈：P(kx), P(ky)
            #    🔴 這裡積分的座標，使用 1D 的 ky, kx，而不是 meshgrid
            # =====================================================
            # A1：對 y 積分，得到 P(kx)，積分變數是 y → 對應 1D 陣列 ky
            # axis=0 對應第一個維度（rows），長度 = len(ky)
            A1 = simpson(A_matrix, ky, axis=0)

            # A2：對 x 積分，得到 P(ky)，積分變數是 x → 對應 1D 陣列 kx
            # axis=1 對應第二個維度（cols），長度 = len(kx)
            A2 = simpson(A_matrix, kx, axis=1)

            # =====================================================
            # 3. (u,v) 變換後，再對 v 積分得到 P(ku)
            #    這裡用的是你固定好的 U, V, ku, kv
            # =====================================================
            X_uv = (U + V) / np.sqrt(2)
            Y_uv = (U - V) / np.sqrt(2)

            # 在 (u, v) 網格上計算 Wigner 分佈
            Z_uv = distribution_atom(X_uv, Y_uv, theta, T_val, zeta, q, N_max)

            # 對 v 軸積分得到 P(u)
            # Z_uv 的第一個維度對應 kv，長度 = len(kv)
            A3 = simpson(Z_uv, kv, axis=0)

            # =====================================================
            # 4. 儲存結果
            # =====================================================
            np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), A_main)
            np.save(os.path.join(path_x_data,       '{}p1.npy'.format(index)), A1)
            np.save(os.path.join(path_y_data,       '{}p2.npy'.format(index)), A2)
            np.save(os.path.join(path_u_data,       '{}p3.npy'.format(index)), A3)

    
    # ============================================================================================

       
        if types == 'TJCM':
            mag_alpha = random.randint(0, 150) / 100   # 大小
            phi_alpha = np.random.uniform(0.0, 2 * np.pi)  # 相位
            alpha = mag_alpha * np.exp(1j * phi_alpha)     # 複數形式

            # n_bar = np.abs(alpha)**2
            # T_tilde = (2*np.pi*np.sqrt(n_bar+1.5))      # 時間參數
            # T_v = random.randint(85, 100)/100
            T = np.random.uniform(5, 2176)/100

            epsilon = np.random.uniform(0,2)
            m_choices = [0, 1, 2]
            m = np.random.choice(m_choices)
            k_choices = [1, 2, 3]
            k = np.random.choice(k_choices)

    # ============================================================================================
            # only test for paper's data
            # alpha = 3.0
            # n_bar = np.abs(alpha)**2
            # T_tilde = (2*np.pi*np.sqrt(n_bar+1.5))      # 時間參數
            # T = T_tilde/2
            # epsilon = 0
            # m = 0
            # k = 1



            N_cutoff = 22
            
            # ---------- lambda_epsilon (公式 2) ----------
            def lambda_eps(alpha, epsilon, m):
                abs_alpha_sq = np.abs(alpha) ** 2
                Lm = eval_genlaguerre(m, 0, 4 * abs_alpha_sq)
                norm_squared = 1 + epsilon**2 + 2 * epsilon * np.exp(-2 * abs_alpha_sq) * Lm

                if (not np.isfinite(norm_squared)) or norm_squared <= 1e-14:
                    return None
                return 1 / np.sqrt(norm_squared)
            
            # ---------- D(alpha) matrix element ----------
 
            def D_element(n, m, alpha):
                coeff = 0.0
                abs_alpha_sq = np.abs(alpha) ** 2

                if n >= m:
                    coeff = np.sqrt(factorial(m) / factorial(n)) * alpha ** (n - m)
                    Lag = eval_genlaguerre(m, n - m, abs_alpha_sq)
                else:
                    coeff = np.sqrt(factorial(n) / factorial(m)) * (-np.conj(alpha)) ** (m - n)
                    Lag = eval_genlaguerre(n, m - n, abs_alpha_sq)

                return np.exp(-0.5 * abs_alpha_sq) * coeff * Lag

            # ---------- Generate C(n, m) ----------
            def generate_Cnm(alpha, epsilon, m, N_cutoff):
                C_nm = np.zeros((N_cutoff,), dtype=np.complex128)
                lam = lambda_eps(alpha, epsilon, m)
                if lam is None:
                    return None

                for n in range(N_cutoff):
                    C1 = D_element(n, m, alpha)
                    C2 = D_element(n, m, -alpha)
                    C_nm[n] = lam * (C1 + epsilon * C2)
                return C_nm

            C_nm = generate_Cnm(alpha, epsilon, m, N_cutoff)
            if C_nm is None:
                continue

            # ---------- 時間演化項定義 ----------
            def zeta_n(n, k):
                return np.sqrt(
                    2.0 * factorial(n + k) / factorial(n)
                    + 2.0 * factorial(n + 2 * k) / factorial(n + k)
                )

            def X1(T, n, k):
                fn = factorial(n)
                fnk = factorial(n + k)
                fn2k = factorial(n + 2 * k)

                z = zeta_n(n, k)
                denom = fnk**2 + fn * fn2k

                return (fn * fnk / denom) * (
                    (fnk / fn) * np.cos(T * z)
                    + (fn2k / fnk)
                )

            def X2(T, n, k):
                fn = factorial(n)
                fnk = factorial(n + k)

                z = zeta_n(n, k)
                return -1j * np.sqrt(fnk / fn) * np.sin(T * z) / z

            def X3(T, n, k):
                return X2(T, n, k)

            def X4(T, n, k):
                fn = factorial(n)
                fnk = factorial(n + k)
                fn2k = factorial(n + 2 * k)

                z = zeta_n(n, k)
                denom = fnk**2 + fn * fn2k

                return (
                    fnk * np.sqrt(fn * fn2k) / denom
                ) * (np.cos(T * z) - 1.0)

            X1_arr = np.array([X1(T, i, k) for i in range(N_cutoff)])
            X2_arr = np.array([X2(T, i, k) for i in range(N_cutoff)])
            X3_arr = np.array([X3(T, i, k) for i in range(N_cutoff)])
            X4_arr = np.array([X4(T, i, k) for i in range(N_cutoff)])         
                        
            
            # ---------- Wigner function 主體 ----------          
            def Wigner_function(x, y, N_cutoff, C_nm, X1_arr, X2_arr, X3_arr, X4_arr):

                result = np.zeros_like(x, dtype=np.float64)
                chi = x + 1j * y
                abs_chi2 = np.abs(chi) ** 2

                for n in range(N_cutoff):
                    X1_n = X1_arr[n]
                    X2_n = X2_arr[n]
                    X3_n = X3_arr[n]
                    X4_n = X4_arr[n]

                    for np_ in range(N_cutoff):
                        # keep upper triangle including diagonal: n <= n'
                        Delta = int(np_ - n)
                        if Delta < 0:
                            continue

                        chi_diff = chi ** Delta

                        # unified prefactor for n <= n'
                        coeff = (
                            C_nm[n] * np.conj(C_nm[np_])
                            * ((-1) ** n)
                            * (2 ** (Delta / 2.0))
                            * chi_diff
                            * np.exp(-abs_chi2) / np.pi
                        )

                        try:
                            X1_np = X1_arr[np_]
                            X2_np = X2_arr[np_]
                            X3_np = X3_arr[np_]
                            X4_np = X4_arr[np_]

                            # term 1
                            term1 = (
                                X1_n * np.conj(X1_np)
                                * np.sqrt(factorial(n) / factorial(np_))
                                * eval_genlaguerre(n, Delta, 2 * abs_chi2)
                            )

                            # term 2
                            term2 = (
                                ((-1) ** k)
                                * X2_n * np.conj(X2_np)
                                * np.sqrt(factorial(n + k) / factorial(np_ + k))
                                * eval_genlaguerre(n + k, Delta, 2 * abs_chi2)
                            )

                            # term 3
                            term3 = (
                                X3_n * np.conj(X3_np)
                                * np.sqrt(factorial(n + k) / factorial(np_ + k))
                                * eval_genlaguerre(n + k, Delta, 2 * abs_chi2)
                            )

                            # term 4
                            term4 = (
                                X4_n * np.conj(X4_np)
                                * np.sqrt(factorial(n + 2 * k) / factorial(np_ + 2 * k))
                                * eval_genlaguerre(n + 2 * k, Delta, 2 * abs_chi2)
                            )

                            val = coeff * (term1 + term2 + term3 + term4)

                            if n == np_:
                                result += np.real(val)
                            else:
                                result += 2.0 * np.real(val)

                        except Exception:
                            continue

                return np.asarray(result, dtype=np.float64)
                

            G_raw = Wigner_function(x, y, N_cutoff, C_nm, X1_arr, X2_arr, X3_arr, X4_arr)
            G = cv2.resize(G_raw, (256, 256))

           

            
            # ---------- Wigner function Marginal ----------
            def Distribution_TJCM(kx, ky, T, m, k, N_cutoff, C_nm):
                result = np.zeros_like(kx, dtype=np.float64)
                chi = kx + 1j * ky
                abs_chi2 = np.abs(chi) ** 2

                for n in range(N_cutoff):
                    X1_n = X1(T, n, k)
                    X2_n = X2(T, n, k)
                    X3_n = X3(T, n, k)
                    X4_n = X4(T, n, k)

                    for np_ in range(N_cutoff):
                        Delta = int(np_ - n)
                        if Delta < 0:
                            continue

                        chi_diff = chi ** Delta

                        coeff = (
                            C_nm[n] * np.conj(C_nm[np_])
                            * ((-1) ** n)
                            * (2 ** (Delta / 2.0))
                            * chi_diff
                            * np.exp(-abs_chi2) / np.pi
                        )

                        try:
                            X1_np = X1(T, np_, k)
                            X2_np = X2(T, np_, k)
                            X3_np = X3(T, np_, k)
                            X4_np = X4(T, np_, k)

                            term1 = (
                                X1_n * np.conj(X1_np)
                                * np.sqrt(factorial(n) / factorial(np_))
                                * eval_genlaguerre(n, Delta, 2 * abs_chi2)
                            )

                            term2 = (
                                ((-1) ** k)
                                * X2_n * np.conj(X2_np)
                                * np.sqrt(factorial(n + k) / factorial(np_ + k))
                                * eval_genlaguerre(n + k, Delta, 2 * abs_chi2)
                            )

                            term3 = (
                                X3_n * np.conj(X3_np)
                                * np.sqrt(factorial(n + k) / factorial(np_ + k))
                                * eval_genlaguerre(n + k, Delta, 2 * abs_chi2)
                            )

                            term4 = (
                                X4_n * np.conj(X4_np)
                                * np.sqrt(factorial(n + 2 * k) / factorial(np_ + 2 * k))
                                * eval_genlaguerre(n + 2 * k, Delta, 2 * abs_chi2)
                            )

                            val = coeff * (term1 + term2 + term3 + term4)

                            if n == np_:
                                result += np.real(val)
                            else:
                                result += 2.0 * np.real(val)

                        except Exception:
                            continue

                return np.asarray(result, dtype=np.float64)
                        
            B_matrix = Distribution_TJCM(x, y, T, m, k, N_cutoff, C_nm)

            # P(kx)：沿 ky 軸積分，axis=0
            G1 = simpson(B_matrix, y, axis=0)

            # P(ky)：沿 kx 軸積分，axis=1
            G2 = simpson(B_matrix, x, axis=1)

            # P(ku)：變數轉換後自己做
            Xuv = (U + V) / np.sqrt(2)
            Yuv = (U - V) / np.sqrt(2)

            Zuv = Wigner_function(Xuv, Yuv, N_cutoff, C_nm, X1_arr, X2_arr, X3_arr, X4_arr)
            # 再對 kv 做積分，得 P(ku)
            G3 = simpson(Zuv, kv, axis=0)

            # 儲存
            np.save(os.path.join(path_main_data_npy, '{}p0_raw.npy'.format(index)), G_raw)
            np.savetxt(os.path.join(path_main_data_npy, '{}p0_raw.csv'.format(index)), G_raw, delimiter=",")
            np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), G)
            np.savetxt(os.path.join(path_main_data_npy, '{}p0.csv'.format(index)), G, delimiter=",")
            np.save(os.path.join(path_x_data, '{}p1.npy'.format(index)), G1)
            np.save(os.path.join(path_y_data, '{}p2.npy'.format(index)), G2)
            np.save(os.path.join(path_u_data, '{}p3.npy'.format(index)), G3)
    # ============================================================================================
        # if types == 'TJCM':

        # #     T_v = random.randint(10,100)/10
        # #     T = T_tilde / T_v
            
        #     alpha = 3
        #     n_bar = np.abs(alpha)**2
        #     T_tilde = (2*np.pi*np.sqrt(n_bar+1.5))      # 時間參數
        #     T_v = 4
        #     T = T_tilde / T_v
        #     k = 1
        #     N_cutoff = 24
            
        #     n = np.arange(N_cutoff)
        #     parity = (-1.0)**n
        #     fact = factorial(n, exact=False)

        #     # coherent state coefficients C_n
        #     C = np.exp(-0.5 * alpha**2) * (alpha**n) / np.sqrt(fact)

        #     # =========================================================
        #     # 只跟 T 有關的解析時間演化（每次 T 變就重算）
        #     # =========================================================
        #     def compute_X_arrays(T):
        #         zeta = np.sqrt(4.0 * n + 6.0)
        #         cosT = np.cos(T * zeta)
        #         sinT = np.sin(T * zeta)

        #         X1 = ((n + 1.0) * cosT + (n + 2.0)) / (2.0 * n + 3.0)
        #         X2 = (-1j) * np.sqrt(n + 1.0) * sinT / zeta
        #         X4 = np.sqrt((n + 1.0) * (n + 2.0)) / (2.0 * n + 3.0) * (cosT - 1.0)
        #         return X1, X2, X4

        #     # =========================================================
        #     # Wigner function（簡化版）
        #     # =========================================================
        #     def Wigner_TJCM(x, y, T, X1, X2, X4):
        #         z = x + 1j*y
        #         abs_z2 = x*x + y*y
        #         expf = np.exp(-abs_z2) / np.pi

        #         G = np.zeros_like(x, dtype=np.float64)

        #         for n1 in range(N_cutoff):
        #             for n2 in range(n1 + 1):
        #                 delta = n1 - n2

        #                 coeff = (
        #                     C[n1] * C[n2] * parity[n2] *
        #                     (2.0**(0.5 * delta)) *
        #                     (z**delta) * expf
        #                 )

        #                 term1 = (
        #                     X1[n1] * X1[n2] *
        #                     np.sqrt(factorial(n2) / factorial(n1)) *
        #                     eval_genlaguerre(n2, delta, 2.0 * abs_z2)
        #                 )

        #                 # k=1, X3=X2 → term2 = -2 |X2||X2'|
        #                 term2 = (
        #                     -2.0 * np.abs(X2[n1]) * np.abs(X2[n2]) *
        #                     np.sqrt(factorial(n2 + 1) / factorial(n1 + 1)) *
        #                     eval_genlaguerre(n2 + 1, delta, 2.0 * abs_z2)
        #                 )

        #                 term3 = (
        #                     X4[n1] * X4[n2] *
        #                     np.sqrt(factorial(n2 + 2) / factorial(n1 + 2)) *
        #                     eval_genlaguerre(n2 + 2, delta, 2.0 * abs_z2)
        #                 )

        #                 val = coeff * (term1 + term2 + term3)

        #                 if n1 == n2:
        #                     G += val.real
        #                 else:
        #                     G += 2.0 * val.real

        #         return G

        #     # =========================================================
        #     # Distribution（kx, ky）
        #     # =========================================================
        #     def Distribution_TJCM(kx, ky, T, X1, X2, X4):
        #         z = kx + 1j*ky
        #         abs_z2 = kx*kx + ky*ky
        #         expf = np.exp(-abs_z2) / np.pi

        #         B = np.zeros_like(kx, dtype=np.float64)

        #         for n1 in range(N_cutoff):
        #             for n2 in range(n1 + 1):
        #                 delta = n1 - n2

        #                 coeff = (
        #                     C[n1] * C[n2] * parity[n2] *
        #                     (2.0**(0.5 * delta)) *
        #                     (z**delta) * expf
        #                 )

        #                 term1 = (
        #                     X1[n1] * X1[n2] *
        #                     np.sqrt(factorial(n2) / factorial(n1)) *
        #                     eval_genlaguerre(n2, delta, 2.0 * abs_z2)
        #                 )

        #                 term2 = (
        #                     -2.0 * np.abs(X2[n1] * X2[n2]) *
        #                     np.sqrt(factorial(n2 + 1) / factorial(n1 + 1)) *
        #                     eval_genlaguerre(n2 + 1, delta, 2.0 * abs_z2)
        #                 )

        #                 term3 = (
        #                     X4[n1] * X4[n2] *
        #                     np.sqrt(factorial(n2 + 2) / factorial(n1 + 2)) *
        #                     eval_genlaguerre(n2 + 2, delta, 2.0 * abs_z2)
        #                 )

        #                 val = coeff * (term1 + term2 + term3)

        #                 if n1 == n2:
        #                     B += val.real
        #                 else:
        #                     B += 2.0 * val.real

        #         return B

        #     # =========================================================
        #     # 時間設定（唯一會變）
        #     # =========================================================

        #     X1, X2, X4 = compute_X_arrays(T)

        #     # =========================================================
        #     # 主分布
        #     # =========================================================
        #     G = Wigner_TJCM(x, y, T, X1, X2, X4)
        #     G = cv2.resize(G, (256, 256))

        #     # =========================================================
        #     # Marginals
        #     # =========================================================
        #     B = Distribution_TJCM(x, y, T, X1, X2, X4)

        #     G1 = simpson(B, y, axis=0)   # P(kx)
        #     G2 = simpson(B, x, axis=1)   # P(ky)

        #     # u-space
        #     Xuv = (U + V) / np.sqrt(2)
        #     Yuv = (U - V) / np.sqrt(2)

        #     Zuv = Wigner_TJCM(Xuv, Yuv, T, X1, X2, X4)
        #     G3 = simpson(Zuv, kv, axis=0)

        #     # =========================================================
        #     # 儲存
        #     # =========================================================

        #     np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), G)
        #     np.save(os.path.join(path_x_data, '{}p1.npy'.format(index)), G1)
        #     np.save(os.path.join(path_y_data, '{}p2.npy'.format(index)), G2)
        #     np.save(os.path.join(path_u_data, '{}p3.npy'.format(index)), G3)

    # ============================================================================================
        if types == 'JCM':
            # mag_alpha = random.randint(0, 150)/100          # 0 ~ 1.5 之間
            # phi_alpha = np.random.uniform(0.0, 2*np.pi)      # 0 ~ 2π 隨機相位
            # alpha = mag_alpha * np.exp(1j * phi_alpha)       # 複數 α，|α| < 3
            # n_choices = [0, 1, 2]
            # n = np.random.choice(n_choices)
            # epsilon = np.random.uniform(0,2)
            # k_choices = [1, 2, 3]
            # k = np.random.choice(k_choices)
            # cutoff = 20
            # T = np.random.uniform(13,4260)/100
    # ============================================================================================
    # Test extra data
            alpha = 1.6
            n = 1
            epsilon = 1.0
            k = 1
            cutoff = 20
            T = 3.4
    # ============================================================================================

            # ========= λ_epsilon ==========
            def lambda_eps(alpha, n, epsilon):
                    beta2 = np.abs(alpha)**2
                    Ln = eval_genlaguerre(n, 0, 4 * beta2)
                    abs_eps = np.abs(epsilon)
                    phi = 0.0 if abs_eps == 0 else np.angle(epsilon)
                    norm = 1 + abs_eps**2 + 2*abs_eps*np.exp(-2*beta2)*Ln*np.cos(phi)

                    # 避免 sqrt(0) 或 sqrt(負數)
                    if norm <= 1e-12 or not np.isfinite(norm):
                        return None  # 用 None 表示「這組參數不可用」
                    return 1 / np.sqrt(norm)
                
            lambda_e = lambda_eps(alpha, n, epsilon)
            if lambda_e is None:
                continue  # 重抽一次 e（不要存檔）

            # ========= SDSN 展開係數 ==========
            def C_m_r0(m, alpha, n, epsilon, lambda_e, fact_n=None, fact_m=None):
                    if fact_n is None: fact_n = factorial(n)
                    if fact_m is None: fact_m = factorial(m)
                    
                    def D_mn(m, n, alpha, fact_n, fact_m):
                        if m >= n:
                            prefactor = np.sqrt(fact_n / fact_m)
                            L = eval_genlaguerre(n, m - n, np.abs(alpha)**2)
                            phase = alpha**(m - n)
                        else:
                            prefactor = np.sqrt(fact_m / fact_n)
                            L = eval_genlaguerre(m, n - m, np.abs(alpha)**2)
                            phase = (-np.conj(alpha))**(n - m)
                        return prefactor * np.exp(-np.abs(alpha)**2/2) * L * phase

                    term_plus = D_mn(m, n, alpha, fact_n, fact_m)
                    term_minus = D_mn(m, n, -alpha, fact_n, fact_m)
                    return lambda_e * (term_plus + epsilon * term_minus)

            # ========= 計算 Wigner Function ==========
            def Wigner_function_JCM(x, y, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0):
                    # 這裡初始化 J (修正 Code 2 漏掉的部分)
                    J = np.zeros_like(x, dtype=np.float64) 
                    fact = [factorial(i) for i in range(cutoff + k + 5)]
                    
                    # 關鍵優化：chi 計算放在迴圈外
                    chi = x + 1j * y
                    abs_chi2 = np.abs(chi)**2

                    for m in range(cutoff):
                        # 預計算 Cm 與時間演化項
                        Cm = C_m_r0(m, alpha, n, epsilon, lambda_e, fact[n], fact[m])
                        h_mk = np.sqrt(fact[m + k] / fact[m])
                        cos_m = np.cos(T * h_mk)
                        sin_m = np.sin(T * h_mk)

                        for mp in range(m + 1):
                            Cmp = C_m_r0(mp, alpha, n, epsilon, lambda_e, fact[n], fact[mp])
                            
                            # 忽略極小項加速
                            if abs(Cm * Cmp) < 1e-15: 
                                continue

                            delta = int(m - mp)
                            prefactor = ((-1)**mp) * (chi**delta) * (2**(delta / 2))
                            
                            sqrt_ratio_cos = np.sqrt(fact[mp] / fact[m])
                            sqrt_ratio_sin = np.sqrt(fact[mp + k] / fact[m + k])
                            
                            L1 = eval_genlaguerre(mp, delta, 2 * abs_chi2)
                            L2 = eval_genlaguerre(mp + k, delta, 2 * abs_chi2)
                            
                            h_mpk = np.sqrt(fact[mp + k] / fact[mp])
                            cos_mp = np.cos(T * h_mpk)
                            sin_mp = np.sin(T * h_mpk)
                            
                            cos_part = sqrt_ratio_cos * cos_m * cos_mp
                            sin_part = (-1)**k * sqrt_ratio_sin * sin_m * sin_mp
                            
                            term = prefactor * Cm * np.conj(Cmp) * (cos_part * L1 + sin_part * L2)

                            # 非對角項修正 (Area=1 的關鍵)
                            if m == mp:
                                J += np.real(term)
                            else:
                                J += 2 * np.real(term)

                    J *= np.exp(-abs_chi2) / np.pi
                    return J
                        
            J = Wigner_function_JCM(x, y, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0)
            J = cv2.resize(J, (256, 256))
            J = np.asarray(J, dtype=np.float64)
            if not np.isfinite(J).all():
                continue

            def Distribution_JCM(kx, ky, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0):
                    # 這裡初始化 J (修正 Code 2 漏掉的部分)
                    J = np.zeros_like(kx, dtype=np.float64)
                    fact = [factorial(i) for i in range(cutoff + k + 5)]
                    
                    # 關鍵優化：chi 計算放在迴圈外
                    chi = kx + 1j * ky
                    abs_chi2 = np.abs(chi)**2

                    for m in range(cutoff):
                        # 預計算 Cm 與時間演化項
                        Cm = C_m_r0(m, alpha, n, epsilon, lambda_e, fact[n], fact[m])
                        h_mk = np.sqrt(fact[m + k] / fact[m])
                        cos_m = np.cos(T * h_mk)
                        sin_m = np.sin(T * h_mk)

                        for mp in range(m + 1):
                            Cmp = C_m_r0(mp, alpha, n, epsilon, lambda_e, fact[n], fact[mp])
                            
                            # 忽略極小項加速
                            if abs(Cm * Cmp) < 1e-15: 
                                continue

                            delta = int(m - mp)
                            prefactor = ((-1)**mp) * (chi**delta) * (2**(delta / 2))
                            
                            sqrt_ratio_cos = np.sqrt(fact[mp] / fact[m])
                            sqrt_ratio_sin = np.sqrt(fact[mp + k] / fact[m + k])
                            
                            L1 = eval_genlaguerre(mp, delta, 2 * abs_chi2)
                            L2 = eval_genlaguerre(mp + k, delta, 2 * abs_chi2)
                            
                            h_mpk = np.sqrt(fact[mp + k] / fact[mp])
                            cos_mp = np.cos(T * h_mpk)
                            sin_mp = np.sin(T * h_mpk)
                            
                            cos_part = sqrt_ratio_cos * cos_m * cos_mp
                            sin_part = (-1)**k * sqrt_ratio_sin * sin_m * sin_mp
                            
                            term = prefactor * Cm * np.conj(Cmp) * (cos_part * L1 + sin_part * L2)

                            # 非對角項修正 (Area=1 的關鍵)
                            if m == mp:
                                J += np.real(term)
                            else:
                                J += 2 * np.real(term)

                    J *= np.exp(-abs_chi2) / np.pi
                    return J

            J_matrix = Distribution_JCM(x, y, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0)
            # ---------- 數值安全檢查：避免 NaN/Inf 污染資料 ----------
            J_matrix = np.asarray(J_matrix, dtype=np.float64)
            if (not np.isfinite(J_matrix).all()):
                continue  # 這筆資料不存，重抽參數
            
            J1 = simpson(J_matrix, x=ky, axis=0)
            J2 = simpson(J_matrix, x=kx, axis=1)
            if (not np.isfinite(J1).all()) or (not np.isfinite(J2).all()):
                continue
            # P(ku)：變數轉換後自己做
            Xuv = (U + V) / np.sqrt(2)
            Yuv = (U - V) / np.sqrt(2)
            Zuv = Wigner_function_JCM(Xuv, Yuv, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0)
            Zuv = np.asarray(Zuv, dtype=np.float64)
            if not np.isfinite(Zuv).all():
                continue

            # 再對 kv 做積分，得 P(ku)
            J3 = simpson(Zuv, x=kv, axis=0)
            if not np.isfinite(J3).all():
                continue

            # 儲存
            np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), J)
            np.save(os.path.join(path_x_data, '{}p1.npy'.format(index)), J1)
            np.save(os.path.join(path_y_data, '{}p2.npy'.format(index)), J2)
            np.save(os.path.join(path_u_data, '{}p3.npy'.format(index)), J3)
