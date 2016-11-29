import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st
from functools import *
import func as f
#################################
def select_best_regress_model(x1, x2, x3, x4, Y):
    m = 7
    N = 87
    matr_X = f.create_X_matr(x1, x2, x3, x4)
    C, R, E, AEV = model_base(Y, matr_X, N, m)
####################################
def elimination_algorithm(matr_X, y, RSS, N, m, p):
        matr_x = [np.delete(matr_X, np.s_[i], 1) for i in range(p)]
        est_theta = [f.parameter_estimation_tetta(el, y) for el in matr_x]
        est_y = [np.matmul(el1, el2) for el1, el2 in zip(matr_x, est_theta)]
        RSS_1 = [np.matmul((y - el).T, (y - el)) for el in est_y]
        F = [(N - m) * (el - RSS) / el for el in RSS_1] 
        return F
    ####################
def quality_criterion(sigm_2, matr_X, y, p, Ys, N):
        est_theta_1 = f.parameter_estimation_tetta(matr_X, y)
        est_y_1 = np.matmul(matr_X, est_theta_1)
        y_est_y_1 = y - est_y_1
        RSS1 = np.matmul((y - est_y_1).T, (y - est_y_1))
        C_p1 = RSS1 / sigm_2 + 2 * p - N
        est_y_mean = np.mean(est_y_1)
        est_Y_s1 = np.sum(np.array([(el - est_y_mean)** 2 for el in est_y_1]))
        R = est_Y_s1 / Ys
        E = (RSS1 / (N * (N - p))) * (1 + N + ((p * (N + 1)) / (N - p - 2)))
        AEV = (p * RSS1) / (N *(N - p))
        return C_p1, R, E, AEV, est_theta_1, est_y_1, y_est_y_1

##################################
def model_base(y, matr_X, N, m):
    p = m
    est_theta = f.parameter_estimation_tetta(matr_X, y)
    RSS = np.matmul((y - np.matmul(matr_X, est_theta)).T, (y - np.matmul(matr_X, est_theta)))
    sigm_2 = RSS / (N - m)
    y_mean = np.mean(y)
    Ys = np.sum(np.array([(el - y_mean) ** 2 for el in y]))
    Cp, R, E, AEV, est_theta, est_y, y_est_y = quality_criterion(sigm_2, matr_X, y, p, Ys, N)
    #p1 = plt.plot(y_est_y, 'r')
    #plt.show()
    f1 = elimination_algorithm(matr_X, y, RSS, N, m, p)
    ######????????? ????????? ?????????
    matr_X1 = np.delete(matr_X, np.s_[1], 1)
    p = m - 1
    Cp1, R1, E1, AEV1, est_theta_1, est_y_1, y_est_y_1 = quality_criterion(sigm_2, matr_X1, y, p, Ys, N)
    #p2 = plt.plot(y_est_y_1, 'r')
    #plt.show()
    f2 = elimination_algorithm(matr_X1, y, RSS, N, m, p)
    ###########################
    matr_X2 = np.delete(matr_X1, np.s_[4], 1)
    p = m - 2
    Cp2, R2, E2, AEV2, est_theta_2, est_y_2, y_est_y_2 = quality_criterion(sigm_2, matr_X2, y, p, Ys, N)
    #p3= plt.plot(y_est_y_2, 'r')
    #plt.show()
    f3 = elimination_algorithm(matr_X2, y, RSS, N, m, p)
    ###############################
    matr_X3 = np.delete(matr_X2, np.s_[2], 1)
    p = m - 3
    Cp3, R3, E3, AEV3, est_theta_3, est_y_3, y_est_y_3 = quality_criterion(sigm_2, matr_X3, y, p, Ys, N)
    #p4 =plt.plot(y_est_y_3, 'r')
    #plt.show()
    f4 = elimination_algorithm(matr_X3, y, RSS, N, m, p)
    ###############
    matr_X4 = np.delete(matr_X3, np.s_[2], 1)
    p = m - 4
    Cp4, R4, E4, AEV4, est_theta_4, est_y_4, y_est_y_4 = quality_criterion(sigm_2, matr_X4, y, p, Ys, N)
    #p5=plt.plot(y_est_y_4, 'r')
    #plt.show()
    f5 = elimination_algorithm(matr_X4, y, RSS, N, m, p)
    #############
    matr_X5 = np.delete(matr_X4, np.s_[2], 1)
    p = m - 5
    Cp5, R5, E5, AEV5, est_theta_5, est_y_5, y_est_y_5 = quality_criterion(sigm_2, matr_X5, y, p, Ys, N)
    #p6=plt.plot(y_est_y_5, 'r')
    #plt.show()
    f6 = elimination_algorithm(matr_X5, y, RSS, N, m, p)
   ###########################################
    matr_X6 = np.delete(matr_X5, np.s_[1], 1)
    p = m - 6
    Cp6, R6, E6, AEV6, est_theta_6, est_y_6, y_est_y_6 = quality_criterion(sigm_2, matr_X6, y, p, Ys, N)
    #p6=plt.plot(y_est_y_5, 'r')
    #plt.show()
    f7 = elimination_algorithm(matr_X6, y, RSS, N, m, p)
    C = [Cp, Cp1, Cp2, Cp3, Cp4, Cp5, Cp6]
    R_ = [R, R1, R2, R3, R4, R5, R6]
    E_ = [E, E1, E2, E3, E4, E5, E6]
    AEV_ = [AEV, AEV1, AEV2, AEV3, AEV4, AEV5, AEV6]
    return Cp, R, E, AEV