import math
import pickle
import datetime
import inflection
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




class Rossmann(object):
    
    def __init__(self):
        self.home_path = ''
        #self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameters/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameters/store_type_scaler.pkl', 'rb'))
        

    def data_cleaning(self, df_raw):
        
        # List old cols names
        cols_old = df_raw.columns.to_list()
        #cols_old.remove('Sales')
        #cols_old.remove('Customers')

        # Lambda function 
        snakecase = lambda x : inflection.underscore(x)

        # List of new cols name
        cols_new = list(map(snakecase, cols_old))

        # Rename
        df_raw.columns = cols_new

        # Change columns date dtype to datetime
        df_raw['date'] = pd.to_datetime(df_raw['date'])

        # competition_distance - Probably the NAs values are store that dont have a near competition or its to far lets fill them with a high number
        df_raw['competition_distance'] = df_raw['competition_distance'].apply(lambda x : 200000.0 if math.isnan(x) else x )

        # competition_open_since_month - Lets fill with the month of column date in this way we will be able to calculate(Feature Engineering) time related variables from it.
        df_raw['competition_open_since_month'] = df_raw.apply(lambda x : x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1 )

        # competition_open_since_year
        df_raw['competition_open_since_year'] = df_raw.apply(lambda x : x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1 )

        #  promo2_since_week
        df_raw['promo2_since_week'] = df_raw.apply(lambda x : x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # promo2_since_year
        df_raw['promo2_since_year'] = df_raw.apply(lambda x : x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # promo_interval
        month_map = {1:'Jan', 2:'Fev',3:'Mar',4:'Abr',5:'Mai',6:'Jun',7:'Jul',8:'Ago',9:'Set',10:'Out',11:'Nov',12:'Dez',}

        df_raw['promo_interval'].fillna(0, inplace=True)
        df_raw['month_map'] = df_raw['date'].dt.month.map(month_map)

        df_raw['is_promo'] = df_raw[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval']==0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        # competition_open_since variabels
        df_raw['competition_open_since_month'] = df_raw['competition_open_since_month'].astype(int)
        df_raw['competition_open_since_year'] = df_raw['competition_open_since_year'].astype(int)

        # promo2_since variables
        df_raw['promo2_since_week'] = df_raw['promo2_since_week'].astype(int)
        df_raw['promo2_since_year'] = df_raw['promo2_since_year'].astype(int)
        
        return df_raw
        
        
    def feature_engineering(self, df_feat):

        # year
        df_feat['year'] = df_feat['date'].dt.year

        # month
        df_feat['month'] = df_feat['date'].dt.month

        # day
        df_feat['day'] = df_feat['date'].dt.day

        # week of year
        df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week

        # year week
        df_feat['year_week'] = df_feat['date'].dt.strftime('%Y-%W')

        # competition since
        df_feat['competition_since'] = df_feat.apply(lambda x : datetime.datetime(year=x['competition_open_since_year'] , month=x['competition_open_since_month'] , day=1 ), axis=1)
        df_feat['competition_time_month']=((df_feat['date'] - df_feat['competition_since'])/30).apply(lambda x: x.days).astype('int')

        # promo since
        df_feat['promo_since'] = df_feat['promo2_since_year'].astype(str) + '-' + df_feat['promo2_since_week'].astype(str) 
        df_feat['promo_since'] = df_feat['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df_feat['promo_time_week'] = ((df_feat['date'] - df_feat['promo_since'])/7).apply(lambda x: x.days).astype(int)

        # assortment
        df_feat['assortment']=df_feat['assortment'].apply(lambda x : 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # state holiday
        df_feat['state_holiday']=df_feat['state_holiday'].apply(lambda x: 'public_holiday' if x=='a' else 'easter_holiday' if x=='b' else 'christmas' if x=='c' else 'regular_day')

        df_feat = df_feat.loc[(df_feat['open'] != 0), :]

        #'customers' - Modelo em separado para prever a quantidade de clientes daqui a 6 meses, sem isso não conseguimos usar
        cols_drop = ['open', 'promo_interval', 'month_map']
        df_feat = df_feat.drop(cols_drop, axis=1)
        
        return df_feat

    def data_preparation(self, df_prep):

        # competition distance, competition time month, promo time week - RobustScaler
        variaveis = ['competition_distance', 'competition_time_month', 'promo_time_week']

        for var in variaveis:
            df_prep[var] = self.competition_distance_scaler.fit_transform(np.array(df_prep[var]).reshape(-1,1))

        # year
        df_prep['year'] = self.year_scaler.fit_transform(np.array(df_prep['year']).reshape(-1,1))

        # state_holiday - One Hot Encoding
        df_prep = pd.get_dummies(df_prep, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df_prep['store_type'] = self.store_type_scaler.fit_transform(df_prep['store_type'])


        # assortment - Ordinal Encoding
        assortment_dict = {'basic' : 1, 'extra': 2, 'extended': 3}
        df_prep['assortment'] = df_prep['assortment'].map(assortment_dict)


        # Nature Transfomation due ciclical nature of variables
        clic_vars = ['day_of_week', 'day', 'month', 'week_of_year']


        for var in clic_vars:
            if var == 'month':
                df_prep[str(var)+'_sin'] = df_prep[var].apply( lambda x: np.sin(x *(2. * np.pi/12)))
                df_prep[str(var)+'_cos'] = df_prep[var].apply( lambda x: np.cos(x *(2. * np.pi/12)))

            elif var == 'day':
                df_prep[str(var)+'_sin'] = df_prep[var].apply( lambda x: np.sin(x *(2. * np.pi/30)))
                df_prep[str(var)+'_cos'] = df_prep[var].apply( lambda x: np.cos(x *(2. * np.pi/30)))

            elif var == 'day_of_week':
                df_prep[str(var)+'_sin'] = df_prep[var].apply( lambda x: np.sin(x *(2. * np.pi/7)))
                df_prep[str(var)+'_cos'] = df_prep[var].apply( lambda x: np.cos(x *(2. * np.pi/7)))

            else:
                df_prep[str(var)+'_sin'] = df_prep[var].apply( lambda x: np.sin(x *(2. * np.pi/52)))
                df_prep[str(var)+'_cos'] = df_prep[var].apply( lambda x: np.cos(x *(2. * np.pi/52)))
                
        # feature selection
        cols_selected = ['store',
                         'promo',
                         'store_type',
                         'assortment',
                         'competition_distance',
                         'competition_open_since_month',
                         'competition_open_since_year',
                         'promo2',
                         'promo2_since_week',
                         'promo2_since_year',
                         'competition_time_month',
                         'promo_time_week',
                         'day_of_week_sin',
                         'day_of_week_cos',
                         'day_sin',
                         'day_cos',
                         'month_cos',
                         'week_of_year_cos']
                
        
        return df_prep[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        
        #prediction
        pred = model.predict(test_data)
        
        #join pred into original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')