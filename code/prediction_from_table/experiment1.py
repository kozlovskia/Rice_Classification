# Biblioteki

import sys
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Wydruk ścieżki do bieżącego katalogu:
print(os.getcwd())

# Wczytujemy dane

data = pd.read_excel(r'C:\\Users\\User\\Desktop\\Projekt uczenie maszynowe\\Rice_MSC_Dataset.xlsx')


data.head()

data['AREA'].head

# Sprawdzamy czy są jakieś missing values
data.isnull().values.any() # Są
ile_brakuje = data.isnull().sum()

print(ile_brakuje)

ile_brakuje[ile_brakuje > 0] # Zmienne dla których są jakies brakujące wartosci

# Usunę wiersze z brakującymi wartosciami.

data2 = data.dropna() # Nowe dane
len(data.index) - len(data2.index) # Czyli było 8 wierszy z brakującymi zmiennymi

data2 = data2.reset_index()

# Tworzymy roboczy zbiór danych bez ostatniej kolumny

data3 = data2.drop(['CLASS'], axis=1)

# Macierz korelacji
# Zbadamy skorelowanie poszczególnych zmiennych (docelowo objasniających), 
# żeby zdecydować których nie ma sensu wspólnie używać przy przewidywaniu rodzaju ryżu.

sns.heatmap(data3) # chyba nie bardzo

M = data3.corr()

# Wypisujemy "Mocno" dla elementów macierzy, w których korelacja > 0.9
M[abs(M) > 0.9] = 'Mocno'

# Rysujemy boxploty dotyczące AREA dla różnych rodzajów ryżu

data_powierzchnia = data2[['AREA','CLASS']]

box_area = sns.boxplot(x='CLASS', y='AREA', data=data_powierzchnia, color='#99c2a2')

Ipsala = data_powierzchnia.loc[data_powierzchnia['CLASS'] == 'Ipsala']
Ipsala = Ipsala.reset_index()


Ipsala[(Ipsala['AREA'] > 12500) & (Ipsala['AREA'] < 15000)].shape[0] # ponad połowa wartosci w srodku boxplota: 8827

Ipsala[Ipsala['AREA'] > 17000].shape[0] # 260
Ipsala[Ipsala['AREA'] > 20000].shape[0] # 1 mocny outlier

Ipsala[Ipsala['AREA'] < 10000].shape[0] # 111

Ipsala['AREA'].mean()
Ipsala['AREA'].median()

ax2 = sns.boxplot(x='CLASS', y='AREA', data=Ipsala, color='#99c2a2')

ax = sns.boxplot(x='CLASS', y='AREA', data=data_powierzchnia, color='#99c2a2')
# Boxplot stworzony dla wszystkich typów ryżu wskazuje na to, że są istotne różnice w powierzchni jeżeli chodzi o Ipsala, Karacadag, Jasmine,
# natomiast zbadanie powierzchni nie odróżni nam od siebie Basmati i Arborio.

# Weźmiemy jeszcze trwałosć, żeby może odróżnić dwa pozostałe rodzaje ryżu

data_trwalosc = data2[['SOLIDITY','CLASS']]

box_trwalosc = sns.boxplot(x='CLASS', y='SOLIDITY', data=data_trwalosc, color='#99c2a2')
# Powinno nam odróżnić basmati od arborio
# KNN
X,Y = data[['AREA','SOLIDITY']], data['CLASS']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
przewidziane = knn.predict(X_test)

print(classification_report(Y_test, przewidziane))

