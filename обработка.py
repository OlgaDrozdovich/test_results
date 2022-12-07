# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:57:15 2022

@author: KiriBAss
"""
###### Обработка данных 

import pandas as pd

# объединение данных
def df_merge (df, df_log, df_named, df_contract):
    df_new = pd.merge(df, df_log, how='left')
    #заполняю пустые в столбце event_type на 0, т.к. это 0 обращений
    df_new = df_new.fillna(value = {'event_type': 0})
    df_new = pd.merge(df_new, df_named, how='left')
    #заполняю пустые в столбце url на 0, т.к. это 0 обращений
    df_new = df_new.fillna(value = {'url': 0})
    df_new = pd.merge(df_new, df_contract, how='left')
    df_new['num_event'] = df_new['event_type']   #переименование столбца
    df_new['num_url'] = df_new['url'] #переименование столбца
    df_new = df_new[['contract_id', 'day_or_month_contract', 'num_event', 
                                 'num_url', 'blocked']] #вывод результирующего столбца в конец
    return df_new

print('=============Обработка log.csv ================')
#содержит фичи с поведением пользователя и его обращениями
df_log = pd.read_csv('log.csv', sep = '\t')
#проверяю на дубликаты, удаляю дубликаты. Пустых значений нет
print(df_log.duplicated().value_counts())
df_log = df_log.drop_duplicates()
#группирую по contract_id, суммирую в event_type кол-во обращений (
# можно бы еще отфильтровать по видам сообщений, но из названий не всегда понятна суть обращения
df_log = df_log.groupby(by = ['contract_id']).event_type.count().reset_index()
print(df_log.head())
print(df_log.info())

print('=============Обработка named.csv ================')
# лог днс запросов к доменам конкурентов 
df_named = pd.read_csv('named.csv', sep = '\t')
df_named['date'] = pd.to_datetime(df_named['date'])
#проверяю на дубликаты, удаляю дубликаты (дубликатов много, надо бы уточнить данные)
# Пустых значений нет
print(df_named.duplicated().value_counts())
df_named = df_named.drop_duplicates()
# проверка уник значений, 
print(df_named.url.unique())
# при проверке выяснено, что действительно только конкуренты, группирую по id
df_named = df_named.groupby(by = ['contract_id']).url.count().reset_index()
print(df_named.head())
print(df_named.info())

print('=============Обработка type_contract.csv ================')
#тип списания у пользователей. 1 - посуточная, 0 - помесячная 
df_contract = pd.read_csv('type_contract.csv', sep = ';')
#проверяю на дубликаты, удаляю дубликаты, пустых нет
print(df_contract.duplicated().value_counts())
df_contract = df_contract.drop_duplicates()
print(df_contract.head())
print(df_contract.info())

print('=============Обработка train.csv ================')
#выборка train. 1 - клиент ушел от нас, 0 - остался 
df_train = pd.read_csv('train.csv', sep = ';')
#проверяю на дубликаты, удаляю дубликаты
print(df_train.duplicated().value_counts())
df_train = df_train.drop_duplicates()
#проверка уник значений выходной переменной
print(df_train.blocked.unique())
print(df_train.head())
print(df_train.info())

print('=============Обработка test.csv ================')
#выборка test.
df_test = pd.read_csv('test.csv', delimiter = ';')
#проверяю на дубликаты, удаляю дубликаты,
print(df_test.duplicated().value_counts())
df_test = df_test.drop_duplicates()
print(df_test.head())
print(df_test.info())


print('============= Создание train_new ================') 
df_train_new = df_merge (df_train, df_log, df_named, df_contract)
print(df_train_new.head())
print(df_train_new.info())
#запись в файл для использования в обучении модели
df_train_new.to_csv('train_new.csv')

print('============= Создание test_new ================') 
df_test_new = df_merge (df_test, df_log, df_named, df_contract)
print(df_test_new.head())
print(df_test_new.info())
#запись в файл для заполнения
df_test_new.to_csv('test_new.csv')









