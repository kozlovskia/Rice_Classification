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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif as MIC
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

################################################
#### CZĘŚĆ 1: Podstawowa eksploracja danych ####
################################################

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

data3 = data3.drop(['index'], axis=1)

data4 = data2[['CLASS']]

# Macierz korelacji
# Zbadamy skorelowanie poszczególnych zmiennych (docelowo objasniających), 
# żeby zdecydować których nie ma sensu wspólnie używać przy przewidywaniu rodzaju ryżu.

sns.heatmap(data3) # nie widać tutaj zbyt wiele

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

###################################################
## CZĘŚĆ 2: Feature selection - DRZEWA DECYZYJNE ##
###################################################

X_train, X_test, Y_train, Y_test = train_test_split(data3, data4, test_size=0.1)

mi_score = MIC(X_train, Y_train.values.ravel())
print(mi_score)

np.histogram(mi_score)
plt.hist(mi_score) # widzimy, że najwięcej zmiennych wpada w przedział [0.4,0.6]


mi_score_selected_index = np.where(mi_score > 0.5)[0] # wybiorę zmienne, które mają mi_score > 0.5

X_2 = data3[data3.columns[mi_score_selected_index - 1]] # wybieram zmienne z odpowiednio dużym mi_score



X_train_2, X_test_2, Y_train2, Y_test2 = train_test_split(X_2, data4, test_size=0.1)

model_1 = DTC().fit(X_train,Y_train)
model_2 = DTC().fit(X_train_2,Y_train2)

score_1 = model_1.score(X_test,Y_test)
score_2 = model_2.score(X_test_2,Y_test2)

print(f"score_1:{score_1}\n score_2:{score_2}\n")

# score_1: 0.9968
# score_2: 0.9957333333333334

# pozostałe kolumny w feature selection:
    
data3.columns[mi_score_selected_index - 1]

# liczba zmiennych objasniajacych które zostały:
    
len(data3.columns[mi_score_selected_index - 1]) # 63

# Czyli widzimy, że pomimo usunięcia 63 zmiennych, model praktycznie nie stracił na jakosci

################################################
# Spróbujmy pójsć dalej.
################################################

mi_score_selected_index2 = np.where(mi_score > 0.8)[0] # wybiorę zmienne, które mają mi_score > 0.5

X_3 = data3[data3.columns[mi_score_selected_index2 - 1]] # wybieram zmienne z odpowiednio dużym mi_score



X_train_3, X_test_3, Y_train3, Y_test3 = train_test_split(X_3, data4, test_size=0.1)

model_3 = DTC().fit(X_train_3,Y_train3)
score_3 = model_3.score(X_test_3,Y_test3)

print(f"score_1:{score_1}\n score_3:{score_3}\n")

# score_1: 0.9968
# score_3: 0.9906666666666667

# pozostałe kolumny w feature selection:
    
data3.columns[mi_score_selected_index2 - 1]

# liczba zmiennych objasniajacych które zostały:
    
len(data3.columns[mi_score_selected_index2 - 1]) # 26

################################################
# Wciąż jest bardzo dobrze, idziemy dalej.
################################################

mi_score_selected_index3 = np.where(mi_score > 0.95)[0] # wybiorę zmienne, które mają mi_score > 0.5

X_4 = data3[data3.columns[mi_score_selected_index2 - 1]] # wybieram zmienne z odpowiednio dużym mi_score



X_train_4, X_test_4, Y_train4, Y_test4 = train_test_split(X_4, data4, test_size=0.1)

model_4 = DTC().fit(X_train_4,Y_train4)
score_4 = model_4.score(X_test_4,Y_test4)

print(f"score_1:{score_1}\n score_4:{score_4}\n")

# score_1: 0.9968
# score_4: 0.9917333333333334

# pozostałe kolumny w feature selection:

data3.columns[mi_score_selected_index3 - 1]
    
# 'AREA', 'MAJOR_AXIS', 'MINOR_AXIS', 'EXTENT', 'ASPECT_RATIO',
# 'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_2'

# liczba zmiennych objasniajacych które zostały:

len(data3.columns[mi_score_selected_index3 - 1]) # 8

# Zostawiamy te 8 zmiennych.


################################################
############# CZĘŚĆ 3: LASY LOSOWE #############
################################################


model = RF()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Dla 8 zmiennych

n_scores = cross_val_score(model, X_train_4, Y_train4.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# Accuracy 0.995 - jestesmy bardzo zadowoleni

# Dla wszystkich zmiennych

n_scores2 = cross_val_score(model, X_train, Y_train.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores2), std(n_scores2)))
# Accuracy do sprawdzenia


#################################################
############# CZĘŚĆ 4: NAIWNY BAYES #############
#################################################

gnb = GaussianNB()

# Dla 8 zmiennych

Y_pred = gnb.fit(X_train_4, Y_train4.values.ravel()).predict(X_test_4)

print(classification_report(Y_test4, Y_pred))

# Sprawdzmy jeszcze dla wszystkich zmiennych

Y_pred_wiecej = gnb.fit(X_train, Y_train.values.ravel()).predict(X_test)

print(classification_report(Y_test, Y_pred_wiecej))


########################################
############# CZĘŚĆ 5: KNN #############
########################################

knn = KNeighborsClassifier()

# Dla 8 zmiennych

KNN_pred = knn.fit(X_train_4, Y_train4.values.ravel()).predict(X_test_4)

print(classification_report(Y_test4, KNN_pred))

# Dla wszystkich zmiennych

KNN_pred2 = knn.fit(X_train, Y_train.values.ravel()).predict(X_test)

print(classification_report(Y_test, KNN_pred2))


#####################################################
############# CZĘŚĆ 6: REGRESJA LOGISTYCZNA #########
#####################################################

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Bierzemy po kolei typy ryżu i sprawdzamy jakie wyjdzie accuracy modelu

# Dla 8 zmiennych

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

n_scores = cross_val_score(model, X_train_4, Y_train4.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# Mean Accuracy: 0.738 (0.004)


# Sprawdźmy jak przewidzi nam rodzaje ryżu z X_test_4, Y_test4

log_pred = model.fit(X_train_4, Y_train4.values.ravel()).predict(X_test_4)
print(classification_report(Y_test4, log_pred)) # czy tak można przy tym co robilismy?

print('Predicted Class: %s' % log_pred[0]) # dla pojedynczej obserwacji


# predict a multinomial probability distribution
yhat_1 = model.predict_proba(X_test_4)
# summarize the predicted probabilities
print('Predicted Probabilities: %s' % yhat_1) # i faktycznie wybiera te z największymi prawdopodobieństwami


# Dla wszystkich zmiennych

cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

n_scores2 = cross_val_score(model, X_train, Y_train.values.ravel(), scoring='accuracy', cv=cv2, n_jobs=-1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores2), std(n_scores2)))
# Mean Accuracy: 0.777 (0.004)


# Sprawdźmy jak przewidzi nam rodzaje ryżu z X_test, Y_test

log_pred2 = model.fit(X_train, Y_train.values.ravel()).predict(X_test)
print(classification_report(Y_test, log_pred2)) # czy tak można przy tym co robilismy?

print('Predicted Class: %s' % log_pred2[0]) # dla pojedynczej obserwacji

print('Predicted Class: %s' % log_pred2)
# predict a multinomial probability distribution
yhat_2 = model.predict_proba(X_test)
# summarize the predicted probabilities
print('Predicted Probabilities: %s' % yhat_2) # i faktycznie wybiera te z największymi prawdopodobieństwami


################################
######### CZĘŚĆ 7: PCA #########
################################

x = StandardScaler().fit_transform(X_train) # standaryzujemy

pca = PCA(n_components=8)
pca.fit(x)

print(pca.explained_variance_ratio_) # chcemy mieć przynajmniej około 60% wariancji, tyle kierunków bysmy chcieli

np.sum(pca.explained_variance_ratio_)

# dla 8 mamy 0.8897116519979509 wariancji
# Wniosek: W danych mamy bardzo dużo zmiennych, które niezbyt dobrze wyjasniają typ ryżu,
# a jest kilka zmiennych, które robią to bardzo dobrze.
# 8 składowych głównych wystarcza do klasyfikacji na wysokim poziomie.
# 14 składowych wyjasnia łącznie ponad 95% wariancji - klasyfikacji na niesamowitym poziomie.

