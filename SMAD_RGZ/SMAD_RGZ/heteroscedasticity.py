import func as f
import sympy as sp
import numpy as np
import math
import scipy.stats as st
from functools import *
###################################
def heterosced(x1, x2, x3, x4, Y):
    N = 87
    matr_X = f.create_X_matr(x1, x2, x3, x4)
    est_tetta = f.parameter_estimation_tetta(matr_X, Y)
    sigm = sigma(x1, x2, x3, x4)
    Ess_2, hi, e_t_2 = test_Breusch_Pagan(x1, x2, x3, x4, sigm, Y, est_tetta, N)
    rss, F = test_Goldfeld_Quandt(x1, x2, x3, x4, Y, N)

########################################
def sigma(x1, x2, x3, x4):
    sigm = [math.sqrt(el1 ** 2 + el2 ** 2 + el3 ** 2 + el4 ** 2) for el1, el2, el3, el4 in zip(x1, x2, x3, x4)]
    return sigm
def residual(Y, x1, x2, x3, x4, est_tetta, N):
    e_t = []
    e_t_2 = []
    nu = []
    for i in range(N):
        f1 = []
        f1.append(x1[i] ** 2)
        f1.append(x2[i])
        f1.append(x3[i])
        f1.append(x4[i])
        f1.append(x1[i] * x2[i])
        f1.append(x1[i] * x3[i])
        f1.append(x2[i] * x3[i])
        f1 = np.array(f1, dtype=float)
        nu.append(np.matmul(f1.T, est_tetta))
    e_t_2 = list(map(lambda x,y: (x - y) ** 2, Y, nu))
    est_sigm = math.sqrt(np.sum(e_t_2) / N)
    return  est_sigm, e_t_2
#########################
def Z_t(sigm, N):
    for i in range(N):
        z_t = [[1., el ** 2] for el in sigm]
    return np.array(z_t, dtype=float)
###############################
def regres_construction(e_t_2, est_sigm, z_t, N):
    c_t = list(map(lambda x: x / (est_sigm ** 2), e_t_2))
    est_alpha = f.parameter_estimation_tetta(z_t, c_t)
    est_c_t = np.matmul(est_alpha.T, z_t.T)
    M_c_t = np.sum(c_t) / N
    ESS = list(map(lambda x: (x - M_c_t) ** 2, est_c_t))
    Ess = np.sum(ESS) / 2.0
    hi = st.chi2.ppf(1 - 0.05, 4)
    a = False
    if(Ess / 2 > hi):
        a = True
    else:
        a = False
    return Ess / 2, hi
################################################
def test_Breusch_Pagan(x1, x2, x3, x4, sigm, Y, est_tetta, N):
    est_sigm, e_t_2 = residual(Y, x1, x2, x3, x4, est_tetta, N)
    z_t = Z_t(sigm, N)
    ESS_2, hi = regres_construction(e_t_2, est_sigm, z_t, N)
    return ESS_2, hi, e_t_2
################################################
def get_RSS(new_arr, k, n_c):
    x1_c1 = np.array([new_arr[i][0]  for i in range(k, n_c)])
    x2_c1 = np.array([new_arr[i][1]  for i in range(k, n_c)])
    x3_c1 = np.array([new_arr[i][2]  for i in range(k, n_c)])
    x4_c1 = np.array([new_arr[i][3]  for i in range(k, n_c)])
    y_c1 = np.array([new_arr[i][4]  for i in range(k, n_c)])
    matrX_c1 = f.create_X_matr(x1_c1, x2_c1, x3_c1, x4_c1)
    est_tetta_c1 = f.parameter_estimation_tetta(matrX_c1, y_c1)
    XTet_1 = np.matmul(matrX_c1, est_tetta_c1)
    difY_XTet_1 = y_c1 - XTet_1
    RSS_1 = np.matmul(difY_XTet_1.T, difY_XTet_1)
    return RSS_1
#######################################
def test_Goldfeld_Quandt(x1, x2,x3, x4, Y, N):
    arr = [[el1, el2, el3, el4, el5] for el1, el2, el3, el4, el5 in zip(x1, x2, x3, x4, Y)]
    new_arr = sorted(arr, key=lambda x: x[0] ** 2 + x[1] ** 2 + x[2] **2 + x[3] ** 2 )

    n_c1 = int(N / 3)    
    n_c3 = N - n_c1

    x1_c1 = np.array([new_arr[i][0]  for i in range(n_c1)])
    x2_c1 = np.array([new_arr[i][1]  for i in range(n_c1)])
    x3_c1 = np.array([new_arr[i][2]  for i in range(n_c1)])
    x4_c1 = np.array([new_arr[i][3]  for i in range(n_c1)])
    y_c1 = np.array([new_arr[i][4]  for i in range(n_c1)])

    x1_c3 = np.array([new_arr[i][0] for i in range(n_c3, N)])
    x2_c3 = np.array([new_arr[i][1] for i in range(n_c3, N)])
    x3_c3 = np.array([new_arr[i][2] for i in range(n_c3, N)])
    x4_c3 = np.array([new_arr[i][3] for i in range(n_c3, N)])
    y_c3 = np.array([new_arr[i][4] for i in range(n_c3, N)])

    matrX_c1 = f.create_X_matr(x1_c1, x2_c1, x3_c1, x4_c1)
    est_tetta_c1 = f.parameter_estimation_tetta(matrX_c1, y_c1)
    matrX_c3 = f.create_X_matr(x1_c3, x2_c3, x3_c3, x4_c3)
    est_tetta_c3 = f.parameter_estimation_tetta(matrX_c3, y_c3)

    XTet_1 = np.matmul(matrX_c1, est_tetta_c1)
    difY_XTet_1 = y_c1 - XTet_1
    RSS_1 = np.matmul(difY_XTet_1.T, difY_XTet_1)

    XTet_3 = np.matmul(matrX_c3, est_tetta_c3)
    difY_XTet_3 = y_c3 - XTet_3
    RSS_2 = np.matmul(difY_XTet_3.T, difY_XTet_3)


    RSS1 = get_RSS(new_arr, 0, n_c1)
    RSS2 = get_RSS(new_arr, n_c3, N)
    rss = RSS_2 / RSS_1

    F = st.f.ppf(1 - 0.05, 25, 25)
    a =  False

    if(rss > F):
        a = True
    else:
        a = False
    return rss, F
