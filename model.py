# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 08:16:48 2022

@author: KiriBAss
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# функция определения ближайшего соседа
def Neighbor (train_x, test_x, train_y, test_y, n):
    # создаем модель ближ соседа с кол-вом соседей, равным n
    clsf = KNeighborsClassifier(n_neighbors= n)
    clsf.fit(train_x, train_y)
    y_pred = clsf.predict(test_x)
    return y_pred

#функция создания модели лог регрессии
def log_reg (train_x, test_x, train_y):
    # создаем модель лог регрессии с макс значением эпох, равным 2000
    clf = LogisticRegression(max_iter = 2000)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred
    
#функция создания модели дерева решений
def DesTree (train_x, test_x, train_y):
    # создаем дерево решений с параметрами
    clf = DecisionTreeClassifier(criterion = 'gini', min_samples_split= 5,
                                 min_samples_leaf= 5)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    # создание фигуры с разрешением 300
    fig = plt.figure(dpi = 300, figsize = (10, 8))
    # построение графика дерева решений
    plot_tree(clf, feature_names= train_x.columns)
    plt.show()
    return y_pred

#функция создания модели случайного леса
def Random_forest (train_x, test_x, train_y): 
    clf = RandomForestClassifier(criterion='gini', min_samples_split = 5, 
                                 min_samples_leaf=5)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    # по результатам коэф-та accuracy выбираю модель случайного леса. 
    #Судя по ост коэф-там, перевес данных, исходы которых 0
    with open ('model.pickle', 'wb') as f:
        pickle.dump(clf, f)
    return y_pred 

#функция создания модели нейронной сети
def MLP (train_x, test_x, train_y):
    clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic')
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    return y_pred

# функция верификации - процесс проверки модели на ее точность
def verif(y_pred, test_y):
    # расчет метрик точности
    print(accuracy_score(test_y, y_pred))
    print(recall_score(test_y, y_pred))
    print(precision_score(test_y, y_pred))
    print(f1_score(test_y, y_pred))


df = pd.read_csv('train_new.csv', delimiter = ',')
df = df[['contract_id', 'day_or_month_contract', 'num_event', 
                             'num_url', 'blocked']]
print(df.info())
print(df.head())
print(df.tail())
print(df['blocked'].value_counts()) # большая доля выходных параметров 0

    # разделение на обучающую и тестовую выборки
train_x, test_x, train_y, test_y = train_test_split(df[df.columns[1:-1]],
                                                        df[df.columns[-1]], random_state = 1)

print('======================Nearest_neighbor=============')
y = Neighbor (train_x, test_x, train_y, test_y, 4)
print(y)
    # вызов метода верификации
verif(y, test_y)

print('======================Logistic regression=============')
y2 = log_reg (train_x, test_x, train_y)
    # вызов метода верификации
verif(y2, test_y)
    
print('======================Decision tree=============')
y3 = DesTree (train_x, test_x, train_y)
    # вызов метода верификации
verif(y3, test_y)
    
print('======================neural_network=============')
y4 = MLP (train_x, test_x, train_y)
    # вызов метода верификации
verif(y4, test_y)
    
print('==========================Random forest======================')
y5 = Random_forest(train_x, test_x, train_y)
     # вызов метода верификации
verif(y5, test_y)
