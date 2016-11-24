import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st
    #########################
def create_X_matr(x1, x2, x3, x4):
        X = [[el1 ** 2, el2, el3, el4,  el1 * el2, el1 * el3, el2 * el3] for el1, el2, el3, el4 in zip(x1, x2, x3, x4)]
        return np.array(X, dtype=float)
####################################
def parameter_estimation_tetta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_tetta = np.matmul(XtX_1_Xt, Y)
    return est_tetta
######################################
def Graph(x, y):
    p1 = plt.plot(x, y, 'ro')
    plt.show()