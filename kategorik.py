import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

veriler = pd.read_csv('veriler.csv')
ulke = veriler.iloc[:, 0:1].values

le = LabelEncoder()

ulke[:, 0] = le.fit_transform(ulke[:, 0])

ct = ColumnTransformer([("all", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(ulke)

## veya

ohe = OneHotEncoder(categories='auto')
ulke = ohe.fit_transform(ulke).toarray()
