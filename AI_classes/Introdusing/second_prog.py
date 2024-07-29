import numpy as np
def act(x):
    return 0 if x < 0.5 else 1
def action(genre, duration, actor):
    x = np.array([genre, duration, actor])
    y = np.array([-0.3, 0.3, 0.9])
    y2 = np.array([0.4, 0.1, 0])
    f_weight = np.array([1, -1])

    results1 = act(np.dot(x, y))
    results2 = act(np.dot(x, y2))

    arr = np.array([results1, results2])
    final = np.dot(arr, f_weight)
    return final

if action(0,1,1) == 1:
    print("You're gonna enjoy this movie")
else:
    print("No way, this movie is out of your prefers")


