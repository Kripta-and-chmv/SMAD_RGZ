import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st
from functools import *
import func as f
import autocorr as a
def check_adequacy_of_the_model(x1, x2, x3, x4, Y):
    N = 87
    matr_X = f.create_X_matr(x1, x2, x3, x4)
    est_theta = a.Cochrane_Oreutt_procedure(matr_X,  Y, N)
    est_e = Y - np.matmul(matr_X, est_theta)
    est_sigm = np.matmul(est_e.T, est_e)/(N - 5)
    #matr_X = f.create_X_matr(x1, x2, x3, x4)
    #est_theta = f.parameter_estimation_theta(matr_X, Y)
    
    Ft = 1.5705
    f = st.f.ppf(1 - 0.05, 83, np.isinf)
    F = est_sigm ** 2 / sigm ** 2
    return Ft >= F, F, Ft


