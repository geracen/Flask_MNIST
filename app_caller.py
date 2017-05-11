import requests
import numpy as np
import matplotlib.pyplot as plt
from util import get_data
import json



X,Y=get_data()
N=len(Y)

while True:
    i=np.random.choice(N)
    Z = np.ndarray.tolist(X[i])
    r = requests.post("http://localhost:5000/predict", json={"input":Z})
    print(r)
    j=r.json()
    print(j)
    print("target: ", Y[i])

    plt.imshow(X[i].reshape(28,28),cmap='gray')
    plt.show()

    response=input("Continue? Y/n")
    if response in ('n','N'):
        break
