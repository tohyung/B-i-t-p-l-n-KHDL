"""
data_loader.py
Người 1: Data + Database + EDA
Chức năng: Đọc và làm sạch 4 file CSV MovieLens ml-latest-small.
Output   : movies_df, ratings_df, tags_df, links_df (DataFrames sạch)
"""

import os
import sys
import pandas as pd

# Tránh lỗi UnicodeEncodeError khi chạy trên Windows terminal dùng code page cũ.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Đường dẫn mặc định đến thư mục chứa CSV ──────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc movies.csv, tách genres, trả về DataFrame sạch.
    Columns: movieId, title, genres (list), year (int|None)
    """
    path = os.path.join(data_dir, "movies.csv")
    df = pd.read_csv(path, dtype={"movieId": int, "title": str, "genres": str})

    # Xóa dòng trùng lặp
    df.drop_duplicates(subset="movieId", keep="first", inplace=True)

    # Xử lý genres: "(no genres listed)" → danh sách rỗng
    df["genres"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )

    # Trích năm sản xuất từ title  e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int64")

    df.reset_index(drop=True, inplace=True)
    return df


def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc ratings.csv, chuyển timestamp → datetime, trả về DataFrame sạch.
    Columns: userId, movieId, rating, timestamp, ratedAt
    """
    path = os.path.join(data_dir, "ratings.csv")
    df = pd.read_csv(
        path,
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    )

    # Xóa dòng trùng (cùng userId + movieId)
    df.drop_duplicates(subset=["userId", "movieId"], keep="last", inplace=True)

    # Chuyển timestamp → datetime UTC
    df["ratedAt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    df.reset_index(drop=True, inplace=True)
    return df


def load_tags(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc tags.csv, chuẩn hóa tag text, trả về DataFrame sạch.
    Columns: userId, movieId, tag, timestamp, taggedAt
    """
    path = os.path.join(data_dir, "tags.csv")
    df = pd.read_csv(
        path,
        dtype={"userId": int, "movieId": int, "tag": str, "timestamp": int},
    )

    # Xóa dòng thiếu tag
    df.dropna(subset=["tag"], inplace=True)

    # Chuẩn hóa: strip khoảng trắng, lowercase
    df["tag"] = df["tag"].str.strip().str.lower()

    # Xóa tag rỗng sau normalize
    df = df[df["tag"] != ""]

    # Xóa trùng (cùng userId + movieId + tag)
    df.drop_duplicates(subset=["userId", "movieId", "tag"], keep="first", inplace=True)

    df["taggedAt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    df.reset_index(drop=True, inplace=True)
    return df


def load_links(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Đọc links.csv, trả về DataFrame sạch.
    Columns: movieId, imdbId, tmdbId
    """
    path = os.path.join(data_dir, "links.csv")
    df = pd.read_csv(
        path,
        dtype={"movieId": int, "imdbId": str, "tmdbId": "Int64"},
    )

    df.drop_duplicates(subset="movieId", keep="first", inplace=True)

    # Pad imdbId thành 7 chữ số (chuẩn IMDb)
    df["imdbId"] = df["imdbId"].str.zfill(7)

    df.reset_index(drop=True, inplace=True)
    return df


def load_all(data_dir: str = DATA_DIR):
    """
    Tải cả 4 DataFrame cùng lúc.
    Returns: (movies_df, ratings_df, tags_df, links_df)
    """
    print("  Đang đọc movies.csv  ...", end=" ")
    movies_df  = load_movies(data_dir);  print(f"OK ({len(movies_df):,} rows)")

    print("  Đang đọc ratings.csv ...", end=" ")
    ratings_df = load_ratings(data_dir); print(f"OK ({len(ratings_df):,} rows)")

    print("  Đang đọc tags.csv    ...", end=" ")
    tags_df    = load_tags(data_dir);    print(f"OK ({len(tags_df):,} rows)")

    print("  Đang đọc links.csv   ...", end=" ")
    links_df   = load_links(data_dir);   print(f"OK ({len(links_df):,} rows)")

    return movies_df, ratings_df, tags_df, links_df


# ── Chạy thử độc lập ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    movies_df, ratings_df, tags_df, links_df = load_all()
    print("\n[movies_df]")
    print(movies_df.head(3).to_string(index=False))
    print("\n[ratings_df]")
    print(ratings_df.head(3).to_string(index=False))
