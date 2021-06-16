import json
import requests
import pandas as pd


def load_dataset(store_id):

	test_df = pd.read_csv('../data/raw/test.csv', low_memory=False)
	store_df = pd.read_csv('../data/raw/store.csv', low_memory=False)


	# Merging store data and test data - all loaded in the begin
	df_test = test_df.merge(store_df, on='Store', how='left')
	df_test.drop('Id', inplace=True, axis=1)

	# chose store for prediction, remove closed days and nulls
	df_test = df_test.loc[(df_test['Store']==store_id) & (df_test['Open']!=0) & (~df_test['Open'].isna()),:]

	#Convvert Dataframe to json
	data = json.dumps(df_test.to_dict(orient = 'records'))
	
	return data

def predict(data):
	# API call
	url = 'https://rossmann-api.herokuapp.com/rossmann/predict' 
	header = {'Content-type': 'application/json'}
	data = data

	r = requests.post(url, data=data, headers=header)
	print(f'Status Code: {r.status_code} ')

	d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())
	
	return d1

#d2 = d1[['store','prediction']].groupby('store').sum().reset_index()

#for i in range(len(d2)):
#    print('Store number {} will sell R$ {:,.2f} in the next 6 weeks'.format( d2.loc[i, 'store'], d2.loc[i, 'prediction'] ))
