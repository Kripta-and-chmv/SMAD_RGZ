import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st
import autocorr as a
    #########################
def create_X_matr(x1, x2, x3, x4):
        #X = [[el1 ** 2, el2, el3, el4,  el1 * el2, el1 * el3, el2 * el3] for el1, el2, el3, el4 in zip(x1, x2, x3, x4)]
        X = [[el1 ** 2, el3, el1 * el2, el1 * el3,  el2 * el3] for el1, el2, el3, el4 in zip(x1, x2, x3, x4)]
        return np.array(X, dtype=float)
####################################
def parameter_estimation_theta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_theta = np.matmul(XtX_1_Xt, Y)
    return est_theta
######################################
def Graph(x, y):
    plt.ylim(-5, 5)
    plt.xlim(-2, 2)
    p1 = plt.plot(x, y, 'ro')
    plt.show()
  ############################00####
def freq_intervals_for_response(x1, x2, x3, x4, Y):
    N = 87
    matr_X = create_X_matr(x1, x2, x3, x4)
    est_theta = a.Cochrane_Oreutt_procedure(matr_X,  Y, N)
    est_e = Y - np.matmul(matr_X, est_theta)
    est_sigm = np.matmul(est_e.T, est_e) / (N - 5)
    nu = []
    freq_intervals = [[], []]
    for i in range(N):
        ff = np.array([x1[i] ** 2, x3[i], x1[i] * x2[i], x1[i] * x3[i], x2[i] * x3[i]])
        nu.append(np.matmul(ff.T, est_theta))
    y_max = max(nu)
    f1 = np.array([x1[60] ** 2, x3[60], x1[60] * x2[60], x1[60] * x3[60], x2[60] * x3[60]])
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    tmp = (np.matmul(np.matmul(f1.T, XtX_1), f1))
    delta = 1.989 * est_sigm * math.sqrt(tmp)
    freq_intervals[0].append(y_max - delta)
    freq_intervals[1].append(y_max + delta)
    return est_theta
###########################################
def Check_adequacy(x1, x2, x3, x4, y, N):
   matr_X = create_X_matr(x1, x2, x3, x4)
   series = []
   seria = []
   Nc = 82
   m = 4
   arr = [[el1, el2, el3, el4, el5] for el1, el2, el3, el4, el5 in zip(x1, x2, x3, x4, y)]
   for i in range(Nc - 1):
        series.append(arr[i])
   for i in range(81, 87):
        seria.append(arr[i])
   series.append(seria)
   f_e = N - Nc
   f_lf = Nc - m
   y_avg = np.array([np.average(i) for i in series])
   est_theta = a.Cochrane_Oreutt_procedure(matr_X,  y, N)
   disp_LF = Calc_disp_LF(series, y_avg, matr_X, est_theta, f_lf)
   disp_E = Calc_disp_e(series, y_avg, y, est_theta, f_e)
   return disp_LF/disp_E, st.f.ppf(0.95,f_lf,f_e) 
        

def Calc_disp_LF(series, y_avg, X, tetta, f_lf):
    Xtetta = np.matmul(X, tetta)
    SS_first = np.matmul((y_avg[:-1] - Xtetta[:-6]).T, (y_avg[:-1] - Xtetta[:-6]))
    SS_last =  np.matmul((np.array([y_avg[81]]*6) - Xtetta[81]), (np.array([y_avg[81]]*6) - Xtetta[81]).T)
    SS1 = SS_first + SS_last
    return SS1 / f_lf
###########################################################
def Calc_disp_e(series, y_avg, y, tetta, f_e):     
     SS_first = np.matmul((y[:-6] - y_avg[:-1]).T, (y[:-6] - y_avg[:-1]))
     SS_last =  np.matmul((y[81] - np.array([y_avg[81]]*6)).T, (y[81] - np.array([y_avg[81]]*6)))
     SSE = SS_first + SS_last
     return SSE / f_e
#######################################################
def graf_tail(x1, x2, x3, x4, Y):
    N = 87
    matr_X = create_X_matr(x1, x2, x3, x4)
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XXtX_1 = np.matmul(matr_X, XtX_1)
    XtX_1_Xt = np.matmul(XXtX_1, matr_X.T)
    est_theta = a.Cochrane_Oreutt_procedure(matr_X,  Y, N)
    est_e = Y - np.matmul(matr_X, est_theta)
    diag = np.diag(XtX_1_Xt)
    tmp = [math.sqrt(np.matmul(est_e.T, est_e)/(82*(1 - el ))) for el in diag]
    di = [el / el1 for el, el1 in zip(est_e, tmp)]
    est_sigm = np.matmul(est_e.T, est_e)/(N - 5)
    ###########
    x = np.linspace(1, 87, 87)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.xlabel('x1x2')
    plt.ylabel('y-y^')
    z1 = [el1*el2 for el1, el2 in zip(x1, x2)]
    #z2 = [el1*el2 for el1, el2 in zip(x1, x3)]
    #z3 = [el1*el2 for el1, el2 in zip(x1, x4)]
    #z4 = [el1*el2 for el1, el2 in zip(x2, x3)]
    #z5 = [el1*el2 for el1, el2 in zip(x2, x4)]
    #z6 = [el1*el2 for el1, el2 in zip(x3, x4)]
    #Graph(x, est_e)
    p1 = plt.plot(z1, di, 'ro')
    plt.show()
    return est_theta
