from pathlib import Path

import pandas as pd


def _resolve_data_dir(data_dir: str | Path) -> Path:
    base = Path(data_dir)
    if (base / "movies.csv").exists():
        return base
    raw = base / "raw"
    if (raw / "movies.csv").exists():
        return raw
    raise FileNotFoundError(f"Cannot find MovieLens CSV files under {base}")


def _read_timestamped_csv(path: Path, timestamp_col: str, datetime_col: str, dtype: dict) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=dtype)
    df[datetime_col] = pd.to_datetime(df[timestamp_col], unit="s")
    return df.drop(columns=[timestamp_col])


def load_all(data_dir: str | Path):
    """
    Load các file CSV MovieLens theo đúng định dạng mà import_data.py cần.

    Returns:
        movies_df, ratings_df, tags_df, links_df
    """
    base = _resolve_data_dir(data_dir)

    movies_df = pd.read_csv(base / "movies.csv", dtype={"movieId": "int32"})
    movies_df["genres"] = (
        movies_df["genres"]
        .fillna("")
        .str.split("|")
        .apply(lambda genres: [g for g in genres if g and g != "(no genres listed)"])
    )

    ratings_df = _read_timestamped_csv(
        base / "ratings.csv",
        timestamp_col="timestamp",
        datetime_col="ratedAt",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )

    tags_df = _read_timestamped_csv(
        base / "tags.csv",
        timestamp_col="timestamp",
        datetime_col="taggedAt",
        dtype={"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"},
    )
    tags_df["tag"] = tags_df["tag"].fillna("").astype(str)

    links_df = pd.read_csv(
        base / "links.csv",
        dtype={"movieId": "int32", "imdbId": "string", "tmdbId": "float64"},
    )
    links_df["imdbId"] = links_df["imdbId"].fillna("").astype(str)

    return movies_df, ratings_df, tags_df, links_df
