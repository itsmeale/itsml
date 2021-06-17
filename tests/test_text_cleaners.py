import pandas as pd

from orbiaml.preprocessing.text import TextNormalizationTransformer

corpora = ["Órbia, the point!! Of revolution!! 1221 .,/;!@@!"]


def test_default_pipeline_of_text_normalization():
    transformer = TextNormalizationTransformer({}, min_token_size=0)
    processed_corpora = transformer.transform(corpora)
    assert (
        processed_corpora[0] == "orbia the point of revolution"
    ), "Invalid text normalization default pipelien"


def test_min_token_size_remotion():
    transformer = TextNormalizationTransformer({}, min_token_size=2)
    processed_corpora = transformer.transform(corpora)
    assert (
        processed_corpora[0] == "orbia the point revolution"
    ), "Error when removing tokens below min token size"
