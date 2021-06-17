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


def test_count_document_by_targets_vectorizer():
    X, y = df["documents"].values, df["targets"].values
    vectorizer = CountDocByTargetsVectorizer()
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
