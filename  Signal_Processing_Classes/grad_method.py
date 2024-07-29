import time
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 5*x + 5

def df(x):
    return 2*x - 5

N = 20 #amount of iterations
xx = 0 #starting point(departure)
lmb = 0.1 #сonvergence step

x_plt = np.arange(0, 5, 0.1)
f_plt = [f(x) for x in x_plt]

plt.ion() #interactive mode for graphs visualisation
fig, ax = plt.subplots() #creating window and axises for graph
ax.grid(True)

ax.plot(x_plt, f_plt) #parabola visualisation
point = ax.scatter(xx, f(xx), c='red') #red point

for i in range(N):
    xx = xx - lmb * df(xx) #changing of the argumet per iteration
    point.set_offsets([xx, f(xx)]) #visualisation of the new position

    #restructurisation of the graph and delay per 20 sec
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)

plt.ioff()
print(xx)
ax.scatter(xx, f(xx), c='blue')
plt.show()