import pandas as pd

from itsml.feature_extraction.text import CountDocByTargetsVectorizer

df = pd.DataFrame(
    {"documents": ["a itsml revolution point", "a negative example"], "targets": [1, 0]}
)

expectations = {
    "a itsml revolution point": [4, 1],
    "a negative example": [1, 3],
    "lala lla aal": [0, 0],
}

expectations_for_logistic_regression = {
    "a itsml revolution point": [1, 4, 1],
    "a negative example": [1, 1, 3],
    "lala lla aal": [1, 0, 0],
}

expectations_for_enrich_features = {
    "a itsml revolution point": [4, 1, 9],
    "a negative example": [1, 3, 4],
    "lala lla aal": [0, 0, 0],
}

expectations_for_enrich_features_and_logistics = {
    "a itsml revolution point": [1, 4, 1, 9],
    "a negative example": [1, 1, 3, 4],
    "lala lla aal": [1, 0, 0, 0],
}


def test_count_document_by_targets_vectorizer():
    X, y = df["documents"].values, df["targets"].values
    vectorizer = CountDocByTargetsVectorizer(
        for_logistic_regression=False, enrich_features=False
    )
    output = vectorizer.fit_transform(X, y)

    new_text = list(expectations.keys())[2]
    output_new = vectorizer.transform([new_text])

    assert (
        list(output[0]) == expectations[X[0]]
    ), "Error when trying to vectorize texts counting doc x class for positive classes"
    assert (
        list(output[1]) == expectations[X[1]]
    ), "Error when trying to vectorize texts counting doc x class for negative classes"
    assert expectations[new_text] == list(
        output_new[0]
    ), "Error when trying to vectorize texts counting doc x class for never seen texts"


def test_count_document_by_targets_vectorizer_for_enrich_features():
    X, y = df["documents"].values, df["targets"].values
    vectorizer = CountDocByTargetsVectorizer(
        for_logistic_regression=False, enrich_features=True
    )
    output = vectorizer.fit_transform(X, y)

    new_text = list(expectations_for_enrich_features.keys())[2]
    output_new = vectorizer.transform([new_text])

    assert (
        list(output[0]) == expectations_for_enrich_features[X[0]]
    ), "Error when trying to vectorize texts counting doc x class for positive classes"
    assert (
        list(output[1]) == expectations_for_enrich_features[X[1]]
    ), "Error when trying to vectorize texts counting doc x class for negative classes"
    assert expectations_for_enrich_features[new_text] == list(
        output_new[0]
    ), "Error when trying to vectorize texts counting doc x class for never seen texts"


def test_count_document_by_targets_vectorizer_for_logistic_regression_and_enrich_features():
    X, y = df["documents"].values, df["targets"].values
    vectorizer = CountDocByTargetsVectorizer(
        for_logistic_regression=True, enrich_features=True
    )
    output = vectorizer.fit_transform(X, y)

    new_text = list(expectations_for_enrich_features_and_logistics.keys())[2]
    output_new = vectorizer.transform([new_text])

    assert (
        list(output[0]) == expectations_for_enrich_features_and_logistics[X[0]]
    ), "Error when trying to vectorize texts counting doc x class for positive classes"
    assert (
        list(output[1]) == expectations_for_enrich_features_and_logistics[X[1]]
    ), "Error when trying to vectorize texts counting doc x class for negative classes"
    assert expectations_for_enrich_features_and_logistics[new_text] == list(
        output_new[0]
    ), "Error when trying to vectorize texts counting doc x class for never seen texts"
