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


    for e in trange(num):        
   
    # ============================================================================================

       
        if types == 'TJCM':
            mag_alpha = random.randint(0, 150) / 100  
            phi_alpha = np.random.uniform(0.0, 2 * np.pi)  
            alpha = mag_alpha * np.exp(1j * phi_alpha)    

            T = np.random.uniform(5, 2176)/100

            epsilon = np.random.uniform(0,2)
            m_choices = [0, 1, 2]
            m = np.random.choice(m_choices)
            k_choices = [1, 2, 3]
            k = np.random.choice(k_choices)

            N_cutoff = 22
            

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

            G1 = simpson(B_matrix, y, axis=0)

            G2 = simpson(B_matrix, x, axis=1)

            Xuv = (U + V) / np.sqrt(2)
            Yuv = (U - V) / np.sqrt(2)

            Zuv = Wigner_function(Xuv, Yuv, N_cutoff, C_nm, X1_arr, X2_arr, X3_arr, X4_arr)
            G3 = simpson(Zuv, kv, axis=0)

            np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), G)
            np.save(os.path.join(path_x_data, '{}p1.npy'.format(index)), G1)
            np.save(os.path.join(path_y_data, '{}p2.npy'.format(index)), G2)
            np.save(os.path.join(path_u_data, '{}p3.npy'.format(index)), G3)
    # ============================================================================================
        if types == 'JCM':
            mag_alpha = random.randint(0, 150)/100          
            phi_alpha = np.random.uniform(0.0, 2*np.pi)      
            alpha = mag_alpha * np.exp(1j * phi_alpha)       
            n_choices = [0, 1, 2]
            n = np.random.choice(n_choices)
            epsilon = np.random.uniform(0,2)
            k_choices = [1, 2, 3]
            k = np.random.choice(k_choices)
            cutoff = 20
            T = np.random.uniform(13,4260)/100

            # ========= λ_epsilon ==========
            def lambda_eps(alpha, n, epsilon):
                    beta2 = np.abs(alpha)**2
                    Ln = eval_genlaguerre(n, 0, 4 * beta2)
                    abs_eps = np.abs(epsilon)
                    phi = 0.0 if abs_eps == 0 else np.angle(epsilon)
                    norm = 1 + abs_eps**2 + 2*abs_eps*np.exp(-2*beta2)*Ln*np.cos(phi)

                    if norm <= 1e-12 or not np.isfinite(norm):
                        return None 
                    return 1 / np.sqrt(norm)
                
            lambda_e = lambda_eps(alpha, n, epsilon)
            if lambda_e is None:
                continue  

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

                    J = np.zeros_like(x, dtype=np.float64) 
                    fact = [factorial(i) for i in range(cutoff + k + 5)]
                    

                    chi = x + 1j * y
                    abs_chi2 = np.abs(chi)**2

                    for m in range(cutoff):
                        Cm = C_m_r0(m, alpha, n, epsilon, lambda_e, fact[n], fact[m])
                        h_mk = np.sqrt(fact[m + k] / fact[m])
                        cos_m = np.cos(T * h_mk)
                        sin_m = np.sin(T * h_mk)

                        for mp in range(m + 1):
                            Cmp = C_m_r0(mp, alpha, n, epsilon, lambda_e, fact[n], fact[mp])
                            
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
                    J = np.zeros_like(kx, dtype=np.float64)
                    fact = [factorial(i) for i in range(cutoff + k + 5)]
                    
                    chi = kx + 1j * ky
                    abs_chi2 = np.abs(chi)**2

                    for m in range(cutoff):
                        Cm = C_m_r0(m, alpha, n, epsilon, lambda_e, fact[n], fact[m])
                        h_mk = np.sqrt(fact[m + k] / fact[m])
                        cos_m = np.cos(T * h_mk)
                        sin_m = np.sin(T * h_mk)

                        for mp in range(m + 1):
                            Cmp = C_m_r0(mp, alpha, n, epsilon, lambda_e, fact[n], fact[mp])
                            
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

                            if m == mp:
                                J += np.real(term)
                            else:
                                J += 2 * np.real(term)

                    J *= np.exp(-abs_chi2) / np.pi
                    return J

            J_matrix = Distribution_JCM(x, y, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0)
            J_matrix = np.asarray(J_matrix, dtype=np.float64)
            if (not np.isfinite(J_matrix).all()):
                continue  
            
            J1 = simpson(J_matrix, x=ky, axis=0)
            J2 = simpson(J_matrix, x=kx, axis=1)
            if (not np.isfinite(J1).all()) or (not np.isfinite(J2).all()):
                continue
            Xuv = (U + V) / np.sqrt(2)
            Yuv = (U - V) / np.sqrt(2)
            Zuv = Wigner_function_JCM(Xuv, Yuv, alpha, epsilon, n, T, k, cutoff, lambda_e, C_m_r0)
            Zuv = np.asarray(Zuv, dtype=np.float64)
            if not np.isfinite(Zuv).all():
                continue

            J3 = simpson(Zuv, x=kv, axis=0)
            if not np.isfinite(J3).all():
                continue

            np.save(os.path.join(path_main_data_npy, '{}p0.npy'.format(index)), J)
            np.save(os.path.join(path_x_data, '{}p1.npy'.format(index)), J1)
            np.save(os.path.join(path_y_data, '{}p2.npy'.format(index)), J2)
            np.save(os.path.join(path_u_data, '{}p3.npy'.format(index)), J3)
