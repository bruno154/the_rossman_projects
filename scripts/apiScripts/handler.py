import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann import Rossmann

# loading model
#home_path = '/home/brunods/Documents/portfolio_github/the_rossman_projects/models'
home_path = 'models'
model = pickle.load(open(home_path + '/lgbm_model_rossmann.pkl','rb'))


# Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossman_predict():
    test_json = request.get_json()
    if test_json:
        
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())       

        # Intantiate
        pipeline = Rossmann()

        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        #feature engineering
        df2 = pipeline.feature_engineering(df1)

        #data preparation
        df3 = pipeline.data_preparation(df2)

        #prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
    
    else:
        return Response('{}', status=200, mimetype='application/json')
    
    return df_response

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)