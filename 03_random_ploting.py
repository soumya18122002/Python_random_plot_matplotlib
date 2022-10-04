import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import random

reg = LinearRegression()
x_val = []
y_val = []

for i in range(1000):
    plt.clf()

    x_val.append(random.randint(0, 100))
    y_val.append(random.randint(0, 100))

    x = np.array(x_val)
    x = x.reshape(-1, 1)

    y = np.array(y_val)
    y = y.reshape(-1, 1)

    if i % 5 == 0:
        reg.fit(x, y)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.scatter(x_val, y_val, color='red')
        plt.plot(list(range(100)), reg.predict(
            np.array([x for x in range(100)]).reshape(-1, 1)))
        plt.pause(0.001)

plt.show()
