# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:23:13 2021

@author: Akshatha
"""

import numpy as np
import math
import pandas as pd

def gradient_descent(x,y):
    mcurr = bcurr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.00009
    cost_prev = 0
    for i in range(iterations):
        ypred = mcurr*x + bcurr
        cost = (1/n)*sum([val**2 for val in (y-ypred)])
        md = -(2/n)*sum(x*(y-ypred))
        bd = -(2/n)*sum(y-ypred)
        mcurr = mcurr-learning_rate*md
        bcurr = bcurr-learning_rate*bd
        print("m {}, b {}, cost {}, iteration {}".format(mcurr,bcurr,cost,i))
        if math.isclose(cost, cost_prev,rel_tol=1e-20):
            break
        cost_prev = cost

if __name__ == '__main__':
    # x = np.array([1,2,3,4,5])
    # y = np.array([5,7,9,11,13])
    df = pd.read_csv("../datasets/test_scores.csv")
    x = np.array(df['math'])
    y = np.array(df['cs'])
    print(x)
    gradient_descent(x,y)

