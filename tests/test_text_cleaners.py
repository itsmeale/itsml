from itsml.preprocessing.text import TextNormalizationTransformer

corpora = ["Révolûtion, is the only!! solution!! 1221 .,/;!@@!"]


def test_default_pipeline_of_text_normalization():
    transformer = TextNormalizationTransformer({}, min_token_size=0)
    processed_corpora = transformer.transform(corpora)
    assert (
        processed_corpora[0] == "revolution is the only solution"
    ), "Invalid text normalization default pipeline"


def test_min_token_size_remotion():
    transformer = TextNormalizationTransformer({}, min_token_size=2)
    processed_corpora = transformer.transform(corpora)
    assert (
        processed_corpora[0] == "revolution the only solution"
    ), "Error when removing tokens below min token size"
