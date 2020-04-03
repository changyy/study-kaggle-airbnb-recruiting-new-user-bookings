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

# 處理時間資訊
str_to_datetime_fields = ['date_account_created', 'date_first_booking']

for field in str_to_datetime_fields:
	train_users[field] = pd.to_datetime(train_users[field])

# 處理 label 資料
categorical_features = [
	'affiliate_channel',
	'affiliate_provider',
]
for categorical_feature in categorical_features:
	train_users[categorical_feature].replace('-unknown-', np.nan, inplace=True)
	train_users[categorical_feature].replace('NaN', np.nan, inplace=True)
	train_users[categorical_feature] = train_users[categorical_feature].astype('category')

print("affiliate_channel: ")
print(train_users['affiliate_channel'].unique())
print()
print("affiliate_provider: ")
print(train_users['affiliate_provider'].unique())
print()

def handle_booking_info(data):
	booking_count = data[ data['date_first_booking'].notnull() ].size
	return pd.Series({
		'total' : data.size,
		'booking' : booking_count,
		'rate' : booking_count / data.size,
	})

booking_info = train_users.groupby(['affiliate_channel', 'affiliate_provider']).apply(handle_booking_info).unstack()
print(booking_info)
#print(booking_info['rate']['google'])
#print(booking_info['rate'])
#sys.exit(0)

import math
total_count = 0
booking_count = 0
data = {'total':[], 'booking': [], 'rate': []}
index = []
for key1 in train_users['affiliate_provider'].unique():
	for key2 in train_users['affiliate_channel'].unique():
		key = key1+','+key2
		index.append(key)
		if key1 in booking_info['total'] and key2 in booking_info['total'][key1] and np.isnan(booking_info['total'][key1][key2]) == False:
			data['total'].append( booking_info['total'][key1][key2] )
			total_count = total_count + booking_info['total'][key1][key2]
			if key1 in booking_info['booking'] and key2 in booking_info['booking'][key1] and np.isnan(booking_info['booking'][key1][key2]) == False:
				data['booking'].append( booking_info['booking'][key1][key2] )
				booking_count = booking_count + booking_info['booking'][key1][key2]
				data['rate'].append ( booking_info['booking'][key1][key2] / booking_info['total'][key1][key2] )
			else:
				data['booking'].append(0)
				data['rate'].append(0)
		else:
			data['total'].append(0)
			data['booking'].append(0)
			data['rate'].append(0)

#print(data)

import matplotlib
import matplotlib.pyplot as plot

selected_data = {'user accounts for the overall proportion':[], 'booking count for the overall proportion': [], 'booking rate': []}
#selected_data = { 'booking': [], 'rate': []}
selected_index = []
#selected_data = data
#selected_index = index

#sorted_index = sorted(range(len(data['rate'])), key=lambda k: data['rate'][k] * -1)
#sorted_index = sorted(range(len(data['rate'])), key=lambda k: data['rate'][k] * (data['booking'][k] / booking_count) -1)
sorted_index = sorted(range(len(data['rate'])), key=lambda k: (data['booking'][k] / booking_count) * -1)
for i in range(12):
	selected_index.append( index[sorted_index[i]] )
	selected_data['user accounts for the overall proportion'].append( data['total'][sorted_index[i]] / total_count )
	selected_data['booking count for the overall proportion'].append( data['booking'][sorted_index[i]] / booking_count )
	selected_data['booking rate'].append( data['rate'][sorted_index[i]] )

dataFrame = pd.DataFrame(data=selected_data, index=selected_index)
dataFrame.plot.bar(rot=15, title="affiliate_provider x affiliate_channel");
plot.show(block=True);

"""
# verify
count_checker = {}
value_checker = {}

for index, row in train_users.iterrows():
	if row['affiliate_provider'] == 'facebook':
		if row['date_first_booking'] is not pd.NaT:
			if row['affiliate_channel'] not in value_checker:
				value_checker[ row['affiliate_channel'] ] = 1
			else:
				value_checker[ row['affiliate_channel'] ] = value_checker[ row['affiliate_channel'] ] + 1

		if row['affiliate_channel'] not in count_checker:
			count_checker[ row['affiliate_channel'] ] = 1
		else:
			count_checker[ row['affiliate_channel'] ] = count_checker[ row['affiliate_channel'] ] + 1

for key in count_checker.keys():
	if key in value_checker:
		print("%s rate: %f" % (key, value_checker[key] / count_checker[key]))
	else:
		print("%s rate: 0" % key)

#print(count_checker)
#print(value_checker)
"""
