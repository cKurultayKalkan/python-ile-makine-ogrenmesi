"""
Created on Friday Mar 29 Mart 2020 00:50

@author: ckurultaykalkan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

lin_reg = LinearRegression()
lin_reg.fit(X, Y)
predict_ = lin_reg.predict(X)
plt.scatter(X, Y, edgecolors="red", color="red")
plt.plot(x, predict_, color="blue")

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
plt.scatter(X, Y, edgecolors="red")
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(X)), color="blue")
plt.show()
