import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

def E(y, a, b):
    ff = np.array([a * x + b for x in range(N)])
    return np.dot((y - ff).T, (y - ff))

def Eda(y, a, b):
    ff = np.array([a * x + b for x in range(N)])
    return -2 * np.dot((y-ff).T, range(N))

def Edb(y,a,b):
    ff = np.array([a * x + b for x in range(N)])
    return -2 * (y-ff).sum()

N = 100 #num of experiments
Niter = 50 #num of iterations
sigma = 3 #standart deviation
at = 0.5 #theoretical value of a
bt = 2 #theoretical value of b

aa = 0 #start approximation of a
bb = 0 #start approximation of b
lmb1 = 0.000001 #step of the method
lmb2 = 0.0005 #also a step

f = np.array([at * x + bt for x in range(N)])
y = np.array(f + np.random.normal(0, sigma, N))

a_plt = np.arange(-1, 2, 0.1)
b_plt = np.arange(0, 3, 0.1)
E_plt = np.array([[E(y, a, b) for a in a_plt] for b in b_plt])

plt.ion()
fig = plt.figure()
ax = Axes3D(fig)

a, b = np.meshgrid(a_plt, b_plt)
ax.plot_surface(a, b, E_plt, color='b', alpha=0.5)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('E')

point = ax.scatter(aa, bb, E(y, aa, bb), color='red')

aa1 = 0
bb1 = 0

for n in range(Niter):

    aa1 = aa - lmb1 * Eda(y, aa, bb)
    bb1 = bb - lmb2 * Edb(y, aa, bb)



    ax.scatter(aa1, bb1, E(y, aa, bb), color='y')

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)
    print(aa1, bb1)
    aa = aa1
    bb = bb1


plt.ioff()
plt.show()


