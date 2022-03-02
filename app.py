# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from joblib import load
import numpy as np
import pandas as pd

 
# Declaring our FastAPI instance
app = FastAPI()
 
# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Welcome to GeeksforGeeks (basic)!'}
 
# # Defining path operation for /name endpoint
# @app.get('/{name}')
# def hello_name(name : str):
#     # Defining a function that takes only string as input and output the
#     # following message.
#     return {'message': f'Welcome to GeeksforGeeks!, {name}'}


# Defining path operation for /name endpoint
@app.get('/{description}')
def hello_name(description : str):
    # Defining a function that takes only string as input and output the
    # following message.
    
    clf = load('text_clf.joblib') 
    predicted = clf.predict(pd.Series(description))
    
    result = {'category': list(predicted)}
    
    
    return result


