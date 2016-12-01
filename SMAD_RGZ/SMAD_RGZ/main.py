import reading as r
import func as f
import multicoll as m
import heteroscedasticity as h
import autocorr as a

import selection_best_regression_model as s
import numpy as np

init_data = r.ReadData()
x1, x2, x3, x4, y = init_data.get_data('data.txt')
N = 87

#f.Graph(x1, y)
#f.Graph(x2, y)
#f.Graph(x3, y)
#f.Graph(x4, y)
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
#v1 = [el1*el2*el3 for el1, el2, el3 in zip(x1, x2, x3)]
#v2 = [el1*el2*el3 for el1, el2, el3 in zip(x1, x2, x4)]
#v3 = [el1*el2*el3 for el1, el2, el3 in zip(x1, x3, x4)]
#v4 = [el1*el2*el3 for el1, el2, el3 in zip(x2, x3, x4)]
#f.Graph(v1, y)
#f.Graph(v2, y)
#f.Graph(v3, y)
#f.Graph(v4, y)
#v1 = [el1**2  for el1 in x1]
#v2 = [el1**2  for el1 in x2]
#v3 = [el1**2  for el1 in x3]
#v4 = [el1**2 for el1 in x4]
#f.Graph(v1, y)
#f.Graph(v2, y)
#f.Graph(v3, y)
#f.Graph(v4, y)
#m.check_mult(x1, x2, x3, x4)
h.heterosced(x1, x2, x3, x4, y)
#a.auto_corr(x1, x2, x3, x4, y)
#s.select_best_regress_model(x1, x2, x3, x4, y)
#c.check_adequacy_of_the_model(x1, x2, x3, x4, y)
#f.freq_intervals_for_response(x1, x2, x3, x4, y)
#f.graf_tail(x1, x2, x3, x4, y)
#f.Check_adequacy(x1, x2, x3, x4, y, N)