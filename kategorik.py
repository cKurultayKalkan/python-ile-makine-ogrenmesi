import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

veriler = pd.read_csv('veriler.csv')
ulke = veriler.iloc[:, 0:1].values

le = LabelEncoder()

ulke[:, 0] = le.fit_transform(ulke[:, 0])

ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()
