import numpy as np

def df(x):
    x = np.array(x)
    return -x * (1 - x)

def f(x):
    return 1/(1 + np.exp(x))

W1 = np.random.rand(2,3)
W2 = np.random.rand(3,2)
W3 = np.random.rand(1,2)

def go_forward(inp):
    sum = np.dot(inp, W1)
    out = np.array([f(x) for x in sum])

    sum = np.dot(out, W2)
    out1 = np.array([f(x) for x in sum])

    sum = np.dot(W3, out1)
    y = f(sum)
    return (out, out1, y)

def train(epoch):
    global W1, W2, W3
    lmb = 0.01
    N = 10000
    count = len(epoch)
    for i in range(N):
        x = epoch[np.random.randint(0, count)]
        out, out1, y = go_forward(x[0:2])

        e = y - x[-1]
        delta = e*df(y)
        for k in range(2):
            W3[0, k] = W3[0, k] - lmb * delta * out1[k]

        delta2 = W3 * delta * df(out1)
        for j in range(3):
            W2[j, 0] = W2[j, 0] - lmb * delta2[0, 0] * out[j]
            W2[j, 1] = W2[j, 1] - lmb * delta2[0, 1] * out[j]

        a = 0
        b = 0
        c = 0
        for k in range(2):
            a += delta2[0, k] * W2[0, k]
            b += delta2[0, k] * W2[1, k]
            c += delta2[0, k] * W2[2, k]

        sigma = np.array([[a],[b],[c]])
        delta3 = []
        for k in range(3):
            delta3.append(sigma[k] * df(out)[k])

        for n in range(2):
            for k in range(3):
                W1[n, k] = W1[n, k] - np.array(x[0:2][n]) * delta3[n] * lmb


epoch = [(-1, -1, -1),
 (-1, 1, 1),
 (1, -1, -1)]

train(epoch)
for x in epoch:
    y, out, out1 = go_forward(x[0:2])
    print(f"выходное значение НС: {y} => {x[-1]}")