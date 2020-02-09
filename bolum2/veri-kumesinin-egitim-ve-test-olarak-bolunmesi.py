import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

veriler = pd.read_csv('eksikveriler.csv')

boy = veriler[['boy']]

boykilo = veriler[['boy', 'kilo']]

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:, 1:4].values

imputer = imputer.fit(Yas)

Yas[:] = imputer.transform(Yas)

ulke = veriler.iloc[:, 0:1].values

le = LabelEncoder()

ulke[:, 0] = le.fit_transform(ulke[:, 0])

ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()

sonuc = pd.DataFrame(data=ulke, index=range(22), columns=['fr', 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:, -1:].values

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat([s, sonuc3], axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)
