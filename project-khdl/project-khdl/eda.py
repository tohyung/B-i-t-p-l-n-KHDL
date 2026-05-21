"""
eda.py
Người 1: Data + Database + EDA
Chức năng: Exploratory Data Analysis - in toàn bộ kết quả ra terminal.
Cách dùng : python eda.py
            (hoặc gọi hàm run_eda() từ main.py của Người 3)
"""

import os
import sys
import math
import pandas as pd
from collections import Counter

# Tránh lỗi UnicodeEncodeError khi chạy trên Windows terminal dùng code page cũ.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Import data_loader từ cùng thư mục
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Helpers in terminal ────────────────────────────────────────────────────────
SEP  = "=" * 60
SEP2 = "-" * 60

def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def subheader(title: str):
    print(f"\n{SEP2}")
    print(f"  {title}")
    print(SEP2)

def bar_chart(label_values: list, width: int = 40, unit: str = ""):
    """In bar chart ngang đơn giản ra terminal."""
    if not label_values:
        return
    max_val = max(v for _, v in label_values)
    max_label_len = max(len(str(l)) for l, _ in label_values)
    for label, val in label_values:
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        val_str = f"{val:,.0f}{unit}"
        print(f"  {str(label):<{max_label_len}} │ {bar:<{width}} {val_str}")


def hist_terminal(series: pd.Series, bins: int = 10, width: int = 40):
    """In histogram ra terminal từ một Series số."""
    min_v, max_v = series.min(), series.max()
    step = (max_v - min_v) / bins
    if step == 0:
        return
    counts = []
    for i in range(bins):
        lo = min_v + i * step
        hi = lo + step
        if i < bins - 1:
            cnt = ((series >= lo) & (series < hi)).sum()
        else:
            cnt = ((series >= lo) & (series <= hi)).sum()
        label = f"{lo:.1f}-{hi:.1f}"
        counts.append((label, cnt))
    bar_chart(counts, width=width)


# ═════════════════════════════════════════════════════════════
# UC02 – DATASET SUMMARY
# ═════════════════════════════════════════════════════════════
def dataset_summary(movies_df, ratings_df, tags_df, links_df) -> dict:
    """In tổng quan dataset. Trả về dict để Người 3 dùng."""
    header("DATASET SUMMARY  (UC02)")

    n_users   = ratings_df["userId"].nunique()
    n_movies  = movies_df["movieId"].nunique()
    n_ratings = len(ratings_df)
    n_tags    = len(tags_df)
    avg_rating = ratings_df["rating"].mean()
    rating_min = ratings_df["rating"].min()
    rating_max = ratings_df["rating"].max()

    print(f"  {'Total users':<25}: {n_users:>10,}")
    print(f"  {'Total movies':<25}: {n_movies:>10,}")
    print(f"  {'Total ratings':<25}: {n_ratings:>10,}")
    print(f"  {'Total tags':<25}: {n_tags:>10,}")
    print(f"  {'Avg rating':<25}: {avg_rating:>10.4f}")
    print(f"  {'Rating range':<25}:   [{rating_min} – {rating_max}]")

    # Kiểm tra missing values
    subheader("Missing Values")
    for name, df in [("movies", movies_df), ("ratings", ratings_df),
                     ("tags", tags_df), ("links", links_df)]:
        missing = df.isnull().sum().sum()
        print(f"  {name:<12}: {missing:>5} missing values")

    # Kiểm tra trùng lặp
    subheader("Duplicates Check")
    dup_ratings = ratings_df.duplicated(subset=["userId", "movieId"]).sum()
    dup_tags    = tags_df.duplicated(subset=["userId", "movieId", "tag"]).sum()
    print(f"  ratings (userId+movieId) duplicates: {dup_ratings}")
    print(f"  tags (userId+movieId+tag) duplicates: {dup_tags}")

    return {
        "n_users":    n_users,
        "n_movies":   n_movies,
        "n_ratings":  n_ratings,
        "n_tags":     n_tags,
        "avg_rating": avg_rating,
    }


# ═════════════════════════════════════════════════════════════
# EDA 1 – PHÂN BỐ RATING
# ═════════════════════════════════════════════════════════════
def eda_rating_distribution(ratings_df):
    header("EDA 1 – PHÂN BỐ RATING")

    dist = ratings_df["rating"].value_counts().sort_index()
    print("  Rating │ Count      │ %")
    print("  " + "-" * 40)
    total = len(ratings_df)
    for val, cnt in dist.items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"   {val:<5} │ {cnt:>9,} │ {pct:5.1f}%  {bar}")

    print(f"\n  Mean  : {ratings_df['rating'].mean():.4f}")
    print(f"  Median: {ratings_df['rating'].median():.1f}")
    print(f"  Std   : {ratings_df['rating'].std():.4f}")


# ═════════════════════════════════════════════════════════════
# EDA 2 – TOP PHIM NHIỀU LƯỢT ĐÁNH GIÁ NHẤT
# ═════════════════════════════════════════════════════════════
def eda_top_rated_movies(ratings_df, movies_df, top_n: int = 15):
    header(f"EDA 2 – TOP {top_n} PHIM NHIỀU LƯỢT ĐÁNH GIÁ NHẤT")

    movie_stats = (
        ratings_df.groupby("movieId")["rating"]
        .agg(count="count", mean="mean")
        .reset_index()
    )
    movie_stats = movie_stats.merge(
        movies_df[["movieId", "title"]], on="movieId", how="left"
    )
    top = movie_stats.nlargest(top_n, "count")

    print(f"\n  {'Rank':<5} {'MovieID':<9} {'Count':>7} {'Avg':>6}  Title")
    print("  " + "-" * 70)
    for i, row in enumerate(top.itertuples(), 1):
        title = row.title[:42] if len(row.title) > 42 else row.title
        print(f"  {i:<5} {row.movieId:<9} {row.count:>7,} {row.mean:>6.2f}  {title}")

    # Bar chart
    print()
    bar_chart(
        [(row.title[:30], row.count) for row in top.itertuples()],
        unit=" ratings"
    )


# ═════════════════════════════════════════════════════════════
# EDA 3 – TOP GENRES PHỔ BIẾN NHẤT
# ═════════════════════════════════════════════════════════════
def eda_top_genres(movies_df, ratings_df, top_n: int = 20):
    header(f"EDA 3 – TOP {top_n} GENRES PHỔ BIẾN NHẤT")

    # Đếm số phim mỗi genre
    genre_movie_count: Counter = Counter()
    for genres in movies_df["genres"]:
        for g in genres:
            genre_movie_count[g] += 1

    # Đếm số lượt rating mỗi genre
    movie_genres = movies_df[["movieId", "genres"]].copy()
    movie_genres = movie_genres.explode("genres").rename(columns={"genres": "genre"})
    merged = ratings_df.merge(movie_genres, on="movieId", how="left")
    genre_rating_count = merged["genre"].value_counts()

    print(f"\n  {'Genre':<25} {'# Movies':>9} {'# Ratings':>12}")
    print("  " + "-" * 50)
    top_genres = sorted(genre_movie_count.items(), key=lambda x: -x[1])[:top_n]
    for genre, mc in top_genres:
        rc = genre_rating_count.get(genre, 0)
        print(f"  {genre:<25} {mc:>9,} {rc:>12,}")

    print()
    bar_chart(top_genres[:15], unit=" movies")


# ═════════════════════════════════════════════════════════════
# EDA 4 – PHÂN BỐ SỐ RATINGS THEO USER
# ═════════════════════════════════════════════════════════════
def eda_user_activity(ratings_df):
    header("EDA 4 – PHÂN BỐ HOẠT ĐỘNG NGƯỜI DÙNG")

    user_counts = ratings_df.groupby("userId")["movieId"].count()
    print(f"  {'Thống kê số phim đã đánh giá / user':}")
    print(f"  Min    : {user_counts.min():>6,}")
    print(f"  Max    : {user_counts.max():>6,}")
    print(f"  Mean   : {user_counts.mean():>6.1f}")
    print(f"  Median : {user_counts.median():>6.0f}")
    print(f"  Std    : {user_counts.std():>6.1f}")

    subheader("Histogram: Số lượt rating của mỗi user")
    hist_terminal(user_counts, bins=10)

    # Nhóm user theo mức độ hoạt động
    subheader("Phân loại user theo mức độ hoạt động")
    light  = (user_counts < 20).sum()
    medium = ((user_counts >= 20) & (user_counts < 100)).sum()
    heavy  = (user_counts >= 100).sum()
    total  = len(user_counts)
    print(f"  Light  (<20 ratings) : {light:>4} users ({light/total*100:.1f}%)")
    print(f"  Medium (20–99)       : {medium:>4} users ({medium/total*100:.1f}%)")
    print(f"  Heavy  (≥100)        : {heavy:>4} users ({heavy/total*100:.1f}%)")


# ═════════════════════════════════════════════════════════════
# EDA 5 – PHÂN BỐ SỐ RATINGS THEO PHIM
# ═════════════════════════════════════════════════════════════
def eda_movie_popularity(ratings_df, movies_df):
    header("EDA 5 – PHÂN BỐ ĐỘ PHỔ BIẾN CỦA PHIM")

    movie_counts = ratings_df.groupby("movieId")["userId"].count()
    print(f"  {'Thống kê số lượt được đánh giá / phim':}")
    print(f"  Min    : {movie_counts.min():>6,}")
    print(f"  Max    : {movie_counts.max():>6,}")
    print(f"  Mean   : {movie_counts.mean():>6.1f}")
    print(f"  Median : {movie_counts.median():>6.0f}")
    print(f"  Std    : {movie_counts.std():>6.1f}")

    subheader("Histogram: Số lượt rating / phim")
    hist_terminal(movie_counts, bins=10)

    never_rated = movies_df[~movies_df["movieId"].isin(ratings_df["movieId"].unique())]
    print(f"\n  Phim chưa được đánh giá lần nào: {len(never_rated):,}")


# ═════════════════════════════════════════════════════════════
# EDA 6 – PHÂN BỐ RATING THEO NĂM
# ═════════════════════════════════════════════════════════════
def eda_rating_over_time(ratings_df):
    header("EDA 6 – SỐ LƯỢT RATING THEO NĂM")

    ratings_df = ratings_df.copy()
    ratings_df["year"] = ratings_df["ratedAt"].dt.year
    by_year = ratings_df.groupby("year").size().reset_index(name="count")

    print(f"\n  {'Year':<8} {'# Ratings':>10}")
    print("  " + "-" * 22)
    for row in by_year.itertuples():
        print(f"  {row.year:<8} {row.count:>10,}")

    print()
    bar_chart(
        [(row.year, row.count) for row in by_year.itertuples()],
        unit=" ratings"
    )


# ═════════════════════════════════════════════════════════════
# EDA 7 – TAG PHỔ BIẾN NHẤT
# ═════════════════════════════════════════════════════════════
def eda_top_tags(tags_df, top_n: int = 20):
    header(f"EDA 7 – TOP {top_n} TAG PHỔ BIẾN NHẤT")

    top_tags = tags_df["tag"].value_counts().head(top_n)
    print(f"\n  {'Tag':<35} {'Count':>7}")
    print("  " + "-" * 45)
    for tag, cnt in top_tags.items():
        print(f"  {tag:<35} {cnt:>7,}")

    print()
    bar_chart(list(top_tags.items())[:15])


# ═════════════════════════════════════════════════════════════
# UC03 – XEM CHI TIẾT PHIM  (query helper cho Người 3)
# ═════════════════════════════════════════════════════════════
def movie_detail_query(movie_title: str, movies_df, ratings_df, tags_df, links_df):
    """
    Trả về chi tiết 1 phim dựa trên tên (tìm kiếm gần đúng).
    Người 3 gọi hàm này trong terminal menu.
    """
    movie_title = str(movie_title).strip()
    if not movie_title:
        print("\n  [!] Vui lòng nhập tên phim cần tìm.")
        return None

    mask = movies_df["title"].str.contains(
        movie_title, case=False, na=False, regex=False
    )
    results = movies_df[mask]

    if results.empty:
        print(f"\n  [!] Không tìm thấy phim: '{movie_title}'")
        return None

    # Lấy phim khớp đầu tiên
    movie = results.iloc[0]
    mid   = movie["movieId"]

    rating_info = ratings_df[ratings_df["movieId"] == mid]["rating"]
    avg_r  = rating_info.mean() if not rating_info.empty else None
    n_r    = len(rating_info)

    movie_tags = tags_df[tags_df["movieId"] == mid]["tag"].unique().tolist()
    link_row   = links_df[links_df["movieId"] == mid]

    print(f"\n  {'='*50}")
    print(f"  MOVIE DETAIL")
    print(f"  {'='*50}")
    print(f"  Movie ID : {mid}")
    print(f"  Title    : {movie['title']}")
    print(f"  Year     : {movie.get('year', 'N/A')}")
    print(f"  Genres   : {', '.join(movie['genres']) if movie['genres'] else '(no genres)'}")
    print(f"  Avg Rating: {avg_r:.2f} / 5.0  ({n_r:,} ratings)" if avg_r else "  Avg Rating: N/A")
    print(f"  Tags     : {', '.join(movie_tags[:10]) if movie_tags else '(no tags)'}")
    if not link_row.empty:
        row = link_row.iloc[0]
        print(f"  IMDb     : https://www.imdb.com/title/tt{row['imdbId']}")
        if pd.notna(row.get("tmdbId")):
            print(f"  TMDb     : https://www.themoviedb.org/movie/{int(row['tmdbId'])}")

    return movie


# ═════════════════════════════════════════════════════════════
# UC04 – XEM LỊCH SỬ RATING USER  (query helper cho Người 3)
# ═════════════════════════════════════════════════════════════
def user_rating_history_query(user_id: int, ratings_df, movies_df, top_n: int = 20):
    """
    Hiển thị lịch sử rating của một user.
    Người 3 gọi hàm này trong terminal menu.
    """
    user_ratings = ratings_df[ratings_df["userId"] == user_id].copy()

    if user_ratings.empty:
        print(f"\n  [!] Không tìm thấy userId: {user_id}")
        return None

    user_ratings = user_ratings.merge(
        movies_df[["movieId", "title"]], on="movieId", how="left"
    ).sort_values("ratedAt", ascending=False)

    print(f"\n  {'='*60}")
    print(f"  USER RATING HISTORY  –  User ID: {user_id}")
    print(f"  {'='*60}")
    print(f"  Tổng số phim đã đánh giá: {len(user_ratings):,}")
    print(f"  Rating trung bình        : {user_ratings['rating'].mean():.2f}")
    print(f"\n  {'Rank':<5} {'Rating':>6} {'Rated At':<20}  Title")
    print("  " + "-" * 78)

    for i, row in enumerate(user_ratings.head(top_n).itertuples(), 1):
        title = row.title[:45] if isinstance(row.title, str) else "N/A"
        rated_at = row.ratedAt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i:<5} {row.rating:>6.1f} {rated_at:<20}  {title}")

    if len(user_ratings) > top_n:
        print(f"  ... và {len(user_ratings) - top_n} phim khác")

    return user_ratings


# ═════════════════════════════════════════════════════════════
# MAIN – Chạy toàn bộ EDA
# ═════════════════════════════════════════════════════════════
def run_eda(data_dir: str = DATA_DIR):
    """Hàm chính – Người 3 có thể gọi hàm này từ main.py."""
    print("\n" + SEP)
    print("  MOVIELENS EDA  –  Người 1: Data + Database + EDA")
    print(SEP)
    print("  Đang tải dữ liệu ...\n")

    movies_df, ratings_df, tags_df, links_df = load_all(data_dir)

    dataset_summary(movies_df, ratings_df, tags_df, links_df)
    eda_rating_distribution(ratings_df)
    eda_top_rated_movies(ratings_df, movies_df)
    eda_top_genres(movies_df, ratings_df)
    eda_user_activity(ratings_df)
    eda_movie_popularity(ratings_df, movies_df)
    eda_rating_over_time(ratings_df)
    eda_top_tags(tags_df)

    print(f"\n{SEP}")
    print("  EDA hoàn tất!")
    print(SEP)

    return movies_df, ratings_df, tags_df, links_df


if __name__ == "__main__":
    run_eda()
