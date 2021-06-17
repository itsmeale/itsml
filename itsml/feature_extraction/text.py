from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CountDocByTargetsVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, for_logistic_regression: bool = True, enrich_features: bool = True
    ):
        self.words_dict: Dict = dict()
        self.__for_logistic_regression = for_logistic_regression
        self.__enrich_features = enrich_features

    def fit(self, X, y=None):
        for document, target in zip(X, y):
            tokens = document.split()
            for token in tokens:
                key = (token, target)
                self.words_dict[key] = self.words_dict.get(key, 0) + 1
        return self

    def transform(self, X, y=None):
        features = list()
        for document in X:
            positive = 0
            negative = 0

            for token in document.split():
                positive_key = (token, 1)
                negative_key = (token, 0)
                positive += self.words_dict.get(positive_key, 0)
                negative += self.words_dict.get(negative_key, 0)

            if self.__enrich_features:
                pos_neg_diff_square = (positive - negative) ** 2

            doc_features = (positive, negative)

            if self.__for_logistic_regression:
                doc_features = (1, *doc_features)

            if self.__enrich_features:
                doc_features = (*doc_features, pos_neg_diff_square)

            features.append(doc_features)

        return np.array(features)
