import numpy as np

def act(x):
    return 0 if x < 0.5 else 1

def go(home, rock, attr):
    x = np.array([home, rock, attr])
    w11 = [0.3, -0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11,w12])
    weight2 = np.array([-1,1])

    sum_hidden = np.dot(weight1, x)
    print(sum_hidden, "значение сумм на нейронах скрытого слоя")
    out_hidden = np.array([act(x) for x in sum_hidden])
    print(out_hidden, "выходные значения СС")

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print(y, "выходное значение НС")

    return y

inp = []
for i in range(3):
    if i == 0:
        inp1 = int(input("enter 1 if you have a home, 0 if you don't have a home: "))
        inp.append(inp1)
    elif i == 1:
        inp2 = int(input("enter 1 if you like to listen to rock, 0 if you don't like to listen to rock: "))
        inp.append(inp2)
    else:
        inp3 = int(input("enter 1 if you attractive, 0 if you don't attractive: "))
        inp.append(inp3)

res = go(inp[0], inp[1], inp[2])
if res == 1:
    print("Your partner likes you")
else:
    print("So sad, you don't have any chances")