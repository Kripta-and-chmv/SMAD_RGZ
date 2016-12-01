import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st
import func as f

    #########################################
def det_inf_matr(matr_X):
        XtX = np.matmul(matr_X.T, matr_X)
        det_XtX = np.linalg.det(XtX)
        return det_XtX, XtX
    ##########################################
def eigen_vals(XtX):
        eig = np.linalg.eig(XtX)
        max_eig = max(eig[0])
        min_eig = min(eig[0])
        return min_eig, max_eig
    #####################################
def measure_cond_matr_Neumann_Goldstein(min_eig, max_eig):
        return max_eig / min_eig
    #######################################
def pair_conjugation(matr_X):
        matr_Xt = matr_X.T
        a = np.array([el for el in matr_Xt], dtype = float)
        r = []
        r_1 = []
        k1 = 5
        k=4
        for i in range(k):
            r.append([])
            for j in range(k):
                if(i != j):
                    r[i].append(np.sum(a[i] * a[j]) / ((np.linalg.norm(a[i]) * np.linalg.norm(a[j]))))
                    r_1.append(np.sum(a[i] * a[j]) / ((np.linalg.norm(a[i]) * np.linalg.norm(a[j]))))
                else:
                    r[i].append(1.)
        max_r = np.max(r_1)
        return max_r, r
    ###################################
def conjugation(r):
        r = np.array(r)
        R = np.linalg.inv(r)
        R_ii = np.diag(R)
        R_i_2 = [1. - 1. / el for el in R_ii]
        max_R_i = np.max(R_i_2)
        return R_i_2, max_R_i
    #################################
def check_mult(x1, x2, x3, x4):
    #matr_X = f.create_X_matr(x1, x2, x3, x4)
    matr_X = np.array([[el1 , el2, el3, el4] for el1, el2, el3, el4 in zip(x1, x2, x3, x4)])
    det_XtX, XtX = det_inf_matr(matr_X)
    min_eigvals, max_eigvals = eigen_vals(XtX)
    cond_NG = measure_cond_matr_Neumann_Goldstein(min_eigvals, max_eigvals)
    max_r, r = pair_conjugation(matr_X)
    R, R_max = conjugation(r)
    return det_XtX, min_eigvals, max_eigvals, cond_NG, max_r, R