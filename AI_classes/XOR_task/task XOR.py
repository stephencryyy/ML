import numpy as np
import matplotlib.pyplot as plt

def act(x):
    return 1 if x > 0 else 0

def gou_suka(C):
    x = np. array([C[0],C[1],1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden= np.array([w1, w2])
    w_output= np.array([-1, 1, -0.5])

    w_sum = np.dot(w_hidden, x)
    out = [act(x) for x in w_sum]
    out.append(1)
    out = np.array(out)

    w_sum = np.dot(w_output, out)
    y = act(w_sum)
    return y

C1 = [(1,0), (0,1)]
C2 = [(0,0), (1,1)]

print(gou_suka(C1[0]), gou_suka(C1[1]))
print(gou_suka(C2[0]), gou_suka(C2[1]))