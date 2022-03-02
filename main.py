'''
Using the following dataset, https://github.com/BestBuyAPIs/open-data-set; build 
an API with one endpoint that receives the “name” and “description” of a 
new product as input parameters, and outputs the “category” or “categories” that 
this new product should be in.

Expectations:
    • You have a data pipeline that handles the dataset.
    • You build a classifier that predicts the category(s) of a new product.
        ◦ You can decide if your model would output one or multiple labels.
        ◦ You don't need to spend lots of time comparing different models.
        ◦ You don't need to spend lots of time on trying to have the state of the art feature engineering.
    • You build one API endpoint that exposes the classifier as a solution to label new products.
'''

import common
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

categories_path = 'open-data-set-master/categories.json'
products_path = 'open-data-set-master/products.json'
stores_path = 'open-data-set-master/stores.json'

categories = common.read_file(categories_path)
products = common.read_file(products_path)
stores = common.read_file(stores_path)

cat_df = pd.DataFrame.from_dict(categories)

prod_df = pd.DataFrame.from_dict(products)
cat_list = ['cat1','cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7']

prod_df[cat_list] = pd.DataFrame(prod_df.category.tolist(), index=prod_df.index)

# for now using only the first category, needs to include all of them
for cat in cat_list:
    print(cat)
    cat_col = prod_df[cat].values.tolist()
    cat_col = [{'id': None, 'name': None} if v is None else v for v in cat_col]
    cat_col = pd.DataFrame(cat_col)
    cat_col.columns = cat + '.' + cat_col.columns

    prod_df[cat_col.columns] = cat_col
    
# cat1_names = list(prod_df['cat1.name'].unique())
# cat2_names = list(prod_df['cat2.name'].unique())
# cat3_names = list(prod_df['cat3.name'].unique())
# cat4_names = list(prod_df['cat4.name'].unique())
# cat5_names = list(prod_df['cat5.name'].unique())
# cat6_names = list(prod_df['cat6.name'].unique())
# cat7_names = list(prod_df['cat7.name'].unique())
    
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat2.id'].unique()))   
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat3.id'].unique()))   
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat4.id'].unique()))   
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat5.id'].unique()))   
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat6.id'].unique()))   
# set(list(prod_df['cat1.id'].unique())) & set(list(prod_df['cat7.id'].unique())) 

train_data = prod_df['description'][0:40000]
test_data = prod_df['description'][40000:]

train_target = prod_df['cat1.id'][0:40000]
test_target = prod_df['cat1.id'][40000:]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, train_target)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(train_data, train_target)

import numpy as np
predicted = text_clf.predict(test_data)
np.mean(predicted == test_target)

from joblib import dump, load
dump(text_clf, 'text_clf.joblib') 







clf = load('text_clf.joblib') 
predicted = clf.predict(test_data)
np.mean(predicted == test_target)




pd.Series(test_data.reset_index(drop=True)[0:5])

pd.Series(test_data[40000:40004])

test_data[[40000]]

clf.predict(pd.Series(test_data.reset_index(drop=True)[0:5]))




aa = pd.Series(['batatas', 'cebolas'])
clf.predict(aa)




