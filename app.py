from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast

app = Flask(__name__)
api = Api(app)

def classify_product(description):
    # load model
    text_clf = load('text_clf.joblib') 
    predicted = text_clf.predict(pd.Series(test_data))
    
    

    return {'predicted': predicted}, 200  # return data and 200 OK


api.add_resource(Users, '/classify_product')  # add endpoints

if __name__ == '__main__':
    app.run()  # run our Flask app