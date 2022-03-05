# Best Buy
===========================

The purpose of this project is to develop a classification system that receives the “name” and “description” of a new product as input parameters, and outputs the “category” that this new product should be in.
This classifier is used in an API endpoint that exposes the classifier as a solution to label new products.


## 1. Dataset

The dataset used for this model is the following: https://github.com/BestBuyAPIs/open-data-set 
The files in this dataset should be unzipped to a folder named "open-data-set-master"


## 2. Model training

The model is trained by running the notebook train_model.ipynb. This notebook generates a file "text_clf.joblib" in the same folder that will be read by the app.

## 3. API

The model is exposed trough an API that receives the “name” and “description” of a new product as input parameters, and outputs the “category” that this new product should be in.
This API is started in a command line by calling `uvicorn app:app`
