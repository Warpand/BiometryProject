import pandas as pd


def l_than(df: pd.DataFrame, key: str, val: int) -> pd.DataFrame:
    return df.iloc[: df[key].searchsorted(val)]


def ge_than(df: pd.DataFrame, key: str, val: int) -> pd.DataFrame:
    return df.iloc[df[key].searchsorted(val) :]


def between(df: pd.DataFrame, key: str, left: int, right: int) -> pd.DataFrame:
    return df.iloc[df[key].searchsorted(left) : df[key].searchsorted(right)]
