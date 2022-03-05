# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from joblib import load
import numpy as np
import pandas as pd
import json

# Declaring FastAPI instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'please input name and description'}

# Defining path operation for /name endpoint
@app.get('/{input}')
def hello_name(input : str):

    input_json = json.loads(input)
    input_df = pd.DataFrame.from_records([input_json])
    
    clf = load('text_clf.joblib') 
    predicted = clf.predict(input_df)
    result = {'category': list(predicted)}
    
    return result
