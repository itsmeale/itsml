from typing import Set

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def remove_stop_words(df: pd.DataFrame, column: str, stopwords: Set) -> pd.DataFrame:
    token_column: str = "__itsml_tokens"
    df[token_column] = df[column].str.split()
    df[column] = df[token_column].apply(
        lambda tokens: " ".join([token for token in tokens if token not in stopwords])
    )
    df = df.drop(columns=[token_column])
    return df


def lowercase(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].str.lower()
    return df


def remove_number(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].str.replace(r"\d+", " ", regex=True)
    return df


def remove_accents(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = (
        df[column]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    return df


def remove_diacritics(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].str.replace("[^a-zA-Z]+", " ", regex=True)
    return df


def remove_small_tokens(
    df: pd.DataFrame, column: str, min_token_size: int
) -> pd.DataFrame:
    regex = r"\b" + r"\w{0," + str(min_token_size) + r"}\b"
    df[column] = df[column].str.replace(regex, " ", regex=True)
    return df


def remove_extra_spaces(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].str.replace(r" +", " ", regex=True).str.strip()
    return df


def default_text_preprocessing_pipeline(
    df: pd.DataFrame, stopwords: Set[str], column: str, min_token_size: int = 0
) -> pd.DataFrame:
    df = (
        df.pipe(lowercase, column=column)
        .pipe(remove_number, column=column)
        .pipe(remove_accents, column=column)
        .pipe(remove_diacritics, column=column)
        .pipe(remove_small_tokens, column=column, min_token_size=min_token_size)
        .pipe(remove_extra_spaces, column=column)
        .pipe(remove_stop_words, column=column, stopwords=stopwords)
    )
    return df


class TextNormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords: Set[str], min_token_size: int = 0):
        self.stopwords = stopwords
        self.column = "__itsml_temp"
        self.min_token_size = min_token_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame({self.column: X}, dtype=str)
        default_text_preprocessing_pipeline(
            df, self.stopwords, self.column, self.min_token_size
        )
        return df[self.column].values
