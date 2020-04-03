# coding=utf8
#
# author: https://github.com/changyy
#

import datetime
import numpy as np
import pandas as pd

import sys
import os
src_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = './'

train_users = pd.read_csv(input_dir+'input/train_users_2.csv')

# 處理 Age 資料
data_checker = train_users.select_dtypes(include=['number']).copy()
data_checker = data_checker[ (data_checker.age > 1000) & (data_checker.age < 2010) ]
data_checker['age'] = 2015 - data_checker['age'] # new_value

for idx,row in data_checker.iterrows():
	train_users.at[idx,'age'] = row['age']

data_checker = train_users.select_dtypes(include=['number']).copy()
data_checker = data_checker[ (data_checker.age >= 2010) | (data_checker.age >= 100) | (data_checker.age < 13) ]
data_checker['age'] = np.nan
for idx,row in data_checker.iterrows():
	train_users.at[idx,'age'] = row['age']

# 處理時間資訊
data_checker = train_users.loc[:, 'timestamp_first_active'].copy()
data_checker = pd.to_datetime( (data_checker // 1000000), format='%Y%m%d')
train_users['timestamp_first_active'] = data_checker

str_to_datetime_fields = ['date_account_created', 'date_first_booking']

for field in str_to_datetime_fields:
	train_users[field] = pd.to_datetime(train_users[field])

# 處理 label 資料
categorical_features = [
	'affiliate_channel',
	'affiliate_provider',
	#'country_destination',
	'first_affiliate_tracked',
	'first_browser',
	'first_device_type',
	'gender',
	'language',
	'signup_app',
	'signup_method'
]
for categorical_feature in categorical_features:
	train_users[categorical_feature].replace('-unknown-', np.nan, inplace=True)
	train_users[categorical_feature].replace('NaN', np.nan, inplace=True)
	train_users[categorical_feature] = train_users[categorical_feature].astype('category')

train_users['first_active_weekday'] = train_users['timestamp_first_active'].dt.dayofweek
for field in str_to_datetime_fields:
	train_users[field+'_weekday'] = train_users[field].dt.dayofweek

train_users.drop(['id'], axis=1, inplace=True)
train_users.drop(str_to_datetime_fields, axis=1, inplace=True)
train_users.drop(['timestamp_first_active'], axis=1, inplace=True)

print(train_users.columns)

print(train_users.head())
