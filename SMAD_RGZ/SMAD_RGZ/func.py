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
