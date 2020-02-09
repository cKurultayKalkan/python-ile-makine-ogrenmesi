import pandas as pd

veriler = pd.read_csv('veriler.csv')

boy = veriler[['boy']]

boykilo = veriler[['boy', 'kilo']]


class insan:
    boy = 180

    def kosmak(self, b):
        return b + 10


ali = insan()
print(ali.boy)
print(ali.kosmak(90))

liste = [1, 2, 3]
