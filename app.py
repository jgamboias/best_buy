# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from joblib import load
import numpy as np
import pandas as pd
import json

# import common
# import numpy as np
# import pandas as pd
# from joblib import dump, load
# from sklearn.pipeline import Pipeline
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import CountVectorizer

# Declaring our FastAPI instance
app = FastAPI()


 
# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'please input name and description'}
 
# # Defining path operation for /name endpoint
# @app.get('/{name}')
# def hello_name(name : str):
#     # Defining a function that takes only string as input and output the
#     # following message.
#     return {'message': f'Welcome to GeeksforGeeks!, {name}'}

# Defining path operation for /name endpoint
@app.get('/{input}')
def hello_name(input : str):
    # Defining a function that takes only string as input and output the
    # following message.
    input_json = json.loads(input)
    input_df = pd.DataFrame.from_records([input_json])
    
    clf = load('text_clf.joblib') 
    predicted = clf.predict(input_df)
    result = {'category': list(predicted)}
    # result_json = result.to_json()
    
    return result





