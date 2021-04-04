import pickle
import pandas as pd
from flask import Flask, request, Response
#from rossmann.Rossmann import Rosmann
import rossmann

# loading model
home_path = '/home/brunods/Documents/portfolio/the_rossman_project/'
pickle.load(open(home_path + '/models/model_rossman.pkl','rb'))

# INitialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossman_predict():
    test_json = request.get_json()
    
    if test_json:
        
        if isinstace(test_json):
            test_raw = pd.DataFrame(test_json, index =[0])
        else:
            test_raw = pd.DataFrame(test_json, columns= test_json[0].keys)       
    else:
        return Response('{}', status=200, mimetype='application/json')
    
    # Intantiate
    pipeine = Rossmann()
    
    #data cleaning
    df1 = pipeline.data_cleaning(test_raw)
    
    #feature engineering
    df2 = pipeline.feature_engineering(df1)
    
    #data preparation
    df3 = pipeline.data_preparation(df2)
    
    #prediction
    df_response = pipeline.get_prediction(model, test_raw, df3)
    
    return df_response

if __name__ == '__main__':
    app.run('0.0.0.0')