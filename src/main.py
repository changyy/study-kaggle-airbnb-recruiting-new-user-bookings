# coding=utf8
#
# author: https://github.com/changyy
#

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import datetime
import numpy as np
import pandas as pd

import sys
import os
src_dir = os.path.dirname(os.path.realpath(__file__))
input_dir = './'

train_users = pd.read_csv(input_dir+'input/train_users_2.csv')
test_users = pd.read_csv(input_dir+'input/test_users.csv')
sessions = pd.read_csv(input_dir+'input/sessions.csv')

# 設定

# 啟用 sample 
train_sample = 5000 
sessions_sample = 10000

sessions = None # 不使用Sessions 資料

# 處理 session 資料
secs_epapsed_info = None
if sessions is not None:
	print("Start to build the secs_epapsed_info ...")
	print(sessions.columns)
	job_start = datetime.datetime.now()
	
	if sessions_sample is not None:
		sessions = sessions.sample(n=sessions_sample,random_state=0).copy()
	secs_epapsed = sessions.groupby('user_id')['secs_elapsed']
	#print(secs_epapsed.first())
	
	def build_secs_epapsed_info(data):
		# data is pd.Series type
		prefix = 'secs_epapsed'
		return pd.Series({
			prefix+'_sum': data.sum(),
			# RuntimeWarning: Mean of empty slice
			prefix+'_mean': data.mean(),
			prefix+'_min': data.min(),
			prefix+'_max': data.max(),
			prefix+'_median': data.median(),
			prefix+'_std': data.std(),
			prefix+'_var': data.var(),
	
			prefix+'_3hour': data[ (data < 3600*3) ].size,
			prefix+'_day': data[ (data > 86400) & (data <= 86400* 7) ].size,
			prefix+'_week': data[ (data > 86400*7)].size,
		})
	
	secs_epapsed_info = secs_epapsed.apply(build_secs_epapsed_info).unstack()
	print("end secs_epapsed_info, cost: ", (datetime.datetime.now() - job_start))
	
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

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
# Convert categorical variable into dummy/indicator variables.
train_users = pd.get_dummies(train_users, columns=categorical_features)

train_users['first_active_weekday'] = train_users['timestamp_first_active'].dt.dayofweek
for field in str_to_datetime_fields:
	train_users[field+'_weekday'] = train_users[field].dt.dayofweek

# 將secs_epapsed_info整合進來
if secs_epapsed_info is not None:
	print("Start to merge user_info & session info ...")
	job_start = datetime.datetime.now()
	train_users = pd.merge(train_users, secs_epapsed_info, left_on='id', right_on='user_id', how = 'left')
	print("merge done, time cost: ", (datetime.datetime.now() - job_start))

train_users.drop(['id'], axis=1, inplace=True)
train_users.drop(str_to_datetime_fields, axis=1, inplace=True)
train_users.drop(['timestamp_first_active'], axis=1, inplace=True)
print(train_users.columns)

# 訓練
X = train_users.copy()
if train_sample is not None:
	X = train_users.sample(n=train_sample,random_state=0).copy()
y = X['country_destination'].copy()
X = X.drop(['country_destination'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

print("Start to train...")
job_start = datetime.datetime.now()
my_model = XGBClassifier()
my_model.fit(X_train, y_train)
print("training done, time cost: ", (datetime.datetime.now() - job_start))

job_start = datetime.datetime.now()
predictions = my_model.predict(X_valid)
print("predict done, time cost: ", (datetime.datetime.now() - job_start))

print("score:", accuracy_score(predictions, y_valid))
