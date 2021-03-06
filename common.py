from __future__ import print_function

import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


def read_file(path):
    
    f = open(path)
    data = json.load(f)
    f.close()
    
    return data

# ItemSelector and text_clf are based in 
# https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


text_clf = Pipeline([
    
    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the name
            ('name', Pipeline([
                ('selector', ItemSelector(key='name')),
                ('count_vec', CountVectorizer()),
                ('tf_idf', TfidfTransformer()),
            ])),

            # Pipeline for standard bag-of-words model for the description
            ('description', Pipeline([
                ('selector', ItemSelector(key='description')),
                ('count_vec', CountVectorizer()),
                ('tf_idf', TfidfTransformer()),
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'name': 1.0,
            'description': 1.0,
        },
    )),

    # Use a MultinomialNB classifier on the combined features
    ('clf', MultinomialNB()),

])
