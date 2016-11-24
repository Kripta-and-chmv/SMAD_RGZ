import reading as r
import func as f
import multicoll as m
import heteroscedasticity as h
import autocorr as a
import numpy as np
init_data = r.ReadData()
x1, x2, x3, x4, y = init_data.get_data('data.txt')
N = 87
#z1 = [el1*el2 for el1, el2 in zip(x1, x2)]
#z2 = [el1*el2 for el1, el2 in zip(x1, x3)]
#z3 = [el1*el2 for el1, el2 in zip(x1, x4)]
#z4 = [el1*el2 for el1, el2 in zip(x2, x3)]
#z5 = [el1*el2 for el1, el2 in zip(x2, x4)]
#z6 = [el1*el2 for el1, el2 in zip(x3, x4)]
#f.Graph(z1, y)
#f.Graph(z2, y)
#f.Graph(z3, y)
#f.Graph(z4, y)
#f.Graph(z5, y)
#f.Graph(z6, y)

m.check_mult(x1, x2, x3, x4)
h.heterosced(x1, x2, x3, x4, y)
a.auto_corr(x1, x2, x3, x4, y)