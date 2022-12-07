# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:27:55 2022

@author: KiriBAss
"""

import pandas as pd
import numpy as np
import pickle

# загружаю обработанный файл test_new.csv для применения модели
df = pd.read_csv('test_new.csv', sep = ',')
# выборка иксов для обработки моделью
df_y_pred = df[['day_or_month_contract', 'num_event', 
                             'num_url']]
#загрузка модели случайных лесов и предсказание 1 или 0
with open('model.pickle', 'rb') as f:
    clf = pickle.load(f)
y_pred = clf.predict (df_y_pred)

#добавление столбца и форматирование датафрейма
df['blocked'] = np.array(y_pred)
df = df[['contract_id', 'day_or_month_contract', 'num_event', 
                             'num_url', 'blocked']]
print(df.info())
print(df['blocked'].value_counts())


#запись в файл для заполнения
df.to_csv('test_results.csv')