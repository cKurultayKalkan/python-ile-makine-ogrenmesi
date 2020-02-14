"""
Created on Friday Mar 14 Şubat 2020 01:49

@author: ckurultaykalkan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

veriler = pd.read_csv('veriler.csv')

boykilo = veriler[['boy', 'kilo']]

Yas = veriler.iloc[:, 1:4].values

ulke = veriler.iloc[:, 0:1].values

le = LabelEncoder()

ulke[:, 0] = le.fit_transform(ulke[:, 0])

ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()

c = veriler.iloc[:, -1:].values

lec = LabelEncoder()

c[:, 0] = lec.fit_transform(c[:, 0])

ohe = OneHotEncoder(categories='auto')

cinsiyet = ohe.fit_transform(c).toarray()

ulke_frame = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
print(ulke_frame)

yas_frame = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print(yas_frame)

cinsiyet_frame = pd.DataFrame(data=cinsiyet[:, :1], index=range(22), columns=["cinsiyet"])
print(cinsiyet_frame)

s = pd.concat([ulke_frame, yas_frame], axis=1)
print(s)

s2 = pd.concat([s, cinsiyet_frame], axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, cinsiyet_frame, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:, 3: 4].values

sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]

veri = pd.concat([sol, sag], axis=1)

x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train, y_train)

prd2 = r2.predict(x_test)

import statsmodels.api as sm

X = np.append(arr=np.ones((22, 1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:, [0, 1, 2, 3, 4, 5]].values

model = sm.OLS(boy, X_l).fit()

print(model.summary())

X = np.append(arr=np.ones((22, 1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:, [0, 1, 2, 3, 5]].values

model = sm.OLS(boy, X_l).fit()

print(model.summary())

X = np.append(arr=np.ones((22, 1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:, [0, 1, 2, 3]].values

model = sm.OLS(boy, X_l).fit()

print(model.summary())
