import func as f
import heteroscedasticity as h
import sympy as sp
import numpy as np
import math
import scipy.stats as st
from functools import *
def auto_corr(x1, x2, x3, x4, Y):
    N = 87
    matr_X = f.create_X_matr(x1, x2, x3, x4)
    est_theta = f.parameter_estimation_theta(matr_X, Y)
    DW, fl = test_Durbin_Watson(matr_X, est_theta, Y, N)
    est_t = Cochrane_Oreutt_procedure(matr_X,  Y, N)
    ##############################################3
def test_Durbin_Watson(matr_X, est_theta, Y, N):
    est_e = Y - np.matmul(matr_X, est_theta)
    DW2 = np.sum(est_e ** 2)
    tmp = list(map(lambda x,y: (x - y) ** 2, est_e[:-1], est_e[1:]))
    DW1 = np.sum(tmp)
    DW = DW1 / DW2
    D_l = 1.56
    D_u = 1.75
    fl = 0
    if(DW > 0 and DW < D_l):
        fl = 1
    if(DW > D_l and DW < D_u):
        fl = 0.5
    if(DW > D_u and DW < 4 - D_u):
        fl = 0
    if(DW > 4 - D_u and DW < 4 - D_l):
        fl = 0.5
    if(DW > 4 - D_l and DW < 4):
        fl = -1
    return DW, fl
################################
def parameter_estimation_OMNK(est_p, matr_X, Y):
    a = np.array([1.])
    b = np.array([1 + est_p ** 2] * 85)
    estp = np.hstack((np.hstack((a,b)), a))
    ep = np.array([-est_p] * 86)
    V = np.diag(estp) + np.diag(ep, k = 1) + np.diag(ep, k = -1)
    V_1 = np.linalg.inv(V)
    X1 = np.matmul(matr_X.T, V_1)
    X2 = np.matmul(X1, matr_X)
    X3 = np.linalg.inv(X2)
    X4 = np.matmul(X3, matr_X.T)
    X5 = np.matmul(X4, V_1)
    est_theta = np.matmul(X5, Y)
    return est_theta
##########################
def search_est_p(matr_X, Y, est_e):
    d = np.sum(est_e ** 2)
    tmp = list(map(lambda x,y: x * y, est_e[:-1], est_e[1:]))
    d_0 = np.sum(tmp)
    est_p1 = d_0 / d
    est_theta1 = parameter_estimation_OMNK(est_p1, matr_X, Y)
    est_e1 = Y - np.matmul(matr_X, est_theta1) 
    return est_p1, est_theta1, est_e1
############################################
def Cochrane_Oreutt_procedure(matr_X,  Y, N):
    est_theta = f.parameter_estimation_theta(matr_X, Y)
    est_e = Y - np.matmul(matr_X, est_theta)
    est_p1, est_theta1, est_e1 = search_est_p(matr_X, Y, est_e)
    est_p2, est_theta2, est_e2 = search_est_p(matr_X, Y, est_e1)
    est_p3, est_theta3, est_e3 = search_est_p(matr_X, Y, est_e2)
    est_p4, est_theta4, est_e4 = search_est_p(matr_X, Y, est_e3)
    est_p5, est_theta5, est_e5 = search_est_p(matr_X, Y, est_e4)
    return est_theta5