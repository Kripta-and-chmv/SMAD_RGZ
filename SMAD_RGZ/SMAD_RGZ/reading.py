import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

class ReadData:
    def get_data(self, fname):
        str_file = []
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        y = []
        with open(fname, 'r') as f:
            for line in f:
                str_file.append(line)
        for i in range(len(str_file)):
            s = str_file[i].expandtabs(1).rstrip()
            x1_el, x2_el, x3_el, x4_el, y_el = s.split(' ')
            x1.append(float(x1_el))
            x2.append(float(x2_el))
            x3.append(float(x3_el))
            x4.append(float(x4_el))
            y.append(float(y_el))
        return x1, x2, x3, x4, y