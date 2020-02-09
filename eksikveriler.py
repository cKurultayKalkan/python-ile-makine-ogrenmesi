import pandas as pd
import numpy as np

veriler = pd.read_csv('eksikveriler.csv')

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

Yas = veriler.iloc[:, 1:4].values

imputer = imputer.fit(Yas)

Yas[:] = imputer.transform(Yas)
