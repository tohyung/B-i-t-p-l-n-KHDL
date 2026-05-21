"""
import_data.py
Người 1: Data + Database + EDA
Chức năng: Import 4 DataFrame sạch vào MySQL (DBeaver).
Yêu cầu  : pip install mysql-connector-python pandas
Cách dùng: python import_data.py
           (chỉnh DB_CONFIG bên dưới trước khi chạy)
"""

import os
import sys
from typing import Optional
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Tránh lỗi UnicodeEncodeError khi chạy trên Windows terminal dùng code page cũ.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Cấu hình kết nối MySQL ────────────────────────────────────────────────────
# Chỉnh các giá trị này cho đúng với máy của bạn
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",          # tên user MySQL
    "password": "123456", # mật khẩu MySQL
    "database": "movielens",
    "charset":  "utf8mb4",
}

# ── Đường dẫn đến thư mục CSV ─────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ─────────────────────────────────────────────────────────────────────────────
def get_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"[LỖI] Không thể kết nối MySQL: {e}")
        sys.exit(1)


def execute_many(cursor, sql: str, data: list, batch_size: int = 2000):
    """Insert theo batch để tránh quá tải bộ nhớ."""
    total = len(data)
    for i in range(0, total, batch_size):
        cursor.executemany(sql, data[i : i + batch_size])


# ─────────────────────────────────────────────────────────────────────────────
def import_movies(cursor, movies_df: pd.DataFrame):
    print("\n[1/7] Import bảng Movies ...")
    rows = [(int(r.movieId), r.title) for r in movies_df.itertuples()]
    sql  = "INSERT IGNORE INTO Movies (movieId, title) VALUES (%s, %s)"
    execute_many(cursor, sql, rows)
    print(f"      → {len(rows):,} phim đã insert.")


def import_genres(cursor, movies_df: pd.DataFrame):
    print("[2/7] Import bảng Genres + MovieGenres ...")

    # Thu thập tất cả genres duy nhất
    all_genres = sorted({g for genres in movies_df["genres"] for g in genres})
    genre_sql  = "INSERT IGNORE INTO Genres (name) VALUES (%s)"
    execute_many(cursor, genre_sql, [(g,) for g in all_genres])

    # Lấy lại genreId từ DB
    cursor.execute("SELECT genreId, name FROM Genres")
    genre_map = {name: gid for gid, name in cursor.fetchall()}

    # Tạo MovieGenres
    mg_rows = [
        (int(r.movieId), genre_map[g])
        for r in movies_df.itertuples()
        for g in r.genres
        if g in genre_map
    ]
    mg_sql = "INSERT IGNORE INTO MovieGenres (movieId, genreId) VALUES (%s, %s)"
    execute_many(cursor, mg_sql, mg_rows)
    print(f"      → {len(all_genres)} genres, {len(mg_rows):,} movie-genre links.")


def import_users(cursor, ratings_df: pd.DataFrame, tags_df: Optional[pd.DataFrame] = None):
    print("[3/7] Import bảng Users ...")
    user_ids = set(ratings_df["userId"].unique())
    if tags_df is not None:
        user_ids.update(tags_df["userId"].unique())
    rows = [(int(uid),) for uid in sorted(user_ids)]
    sql  = "INSERT IGNORE INTO Users (userId) VALUES (%s)"
    execute_many(cursor, sql, rows)
    print(f"      → {len(rows):,} users đã insert.")


def import_ratings(cursor, ratings_df: pd.DataFrame):
    print("[4/7] Import bảng Ratings ...")
    rows = [
        (int(r.userId), int(r.movieId), float(r.rating),
         r.ratedAt.strftime("%Y-%m-%d %H:%M:%S"))
        for r in ratings_df.itertuples()
    ]
    sql = ("INSERT IGNORE INTO Ratings (userId, movieId, rating, ratedAt) "
           "VALUES (%s, %s, %s, %s)")
    execute_many(cursor, sql, rows)
    print(f"      → {len(rows):,} ratings đã insert.")


def import_tags(cursor, tags_df: pd.DataFrame):
    print("[5/7] Import bảng Tags ...")
    rows = [
        (int(r.userId), int(r.movieId), r.tag,
         r.taggedAt.strftime("%Y-%m-%d %H:%M:%S"))
        for r in tags_df.itertuples()
    ]
    sql = ("INSERT IGNORE INTO Tags (userId, movieId, tag, taggedAt) "
           "VALUES (%s, %s, %s, %s)")
    execute_many(cursor, sql, rows)
    print(f"      → {len(rows):,} tags đã insert.")


def import_links(cursor, links_df: pd.DataFrame):
    print("[6/7] Import bảng MovieLinks ...")
    rows = []
    for r in links_df.itertuples():
        tmdb = None if pd.isna(r.tmdbId) else int(r.tmdbId)
        rows.append((int(r.movieId), str(r.imdbId), tmdb))
    sql = ("INSERT IGNORE INTO MovieLinks (movieId, imdbId, tmdbId) "
           "VALUES (%s, %s, %s)")
    execute_many(cursor, sql, rows)
    print(f"      → {len(rows):,} links đã insert.")


def save_recommendations(conn, user_id: int, recommendations, method: str):
    """
    Lưu lịch sử recommendation cho UC08.

    recommendations có thể là list dict/DataFrame rows chứa movieId và score,
    hoặc list tuple dạng (movieId, score).
    """
    valid_methods = {"content_based", "collaborative", "hybrid"}
    if method not in valid_methods:
        raise ValueError(f"method phải thuộc {sorted(valid_methods)}")

    rows = []
    for rec in recommendations:
        if isinstance(rec, dict):
            movie_id = rec.get("movieId")
            score = rec.get("score")
        elif hasattr(rec, "movieId"):
            movie_id = rec.movieId
            score = getattr(rec, "score", None)
        else:
            movie_id = rec[0]
            score = rec[1] if len(rec) > 1 else None
        rows.append((int(user_id), int(movie_id), method, None if score is None else float(score)))

    sql = (
        "INSERT INTO Recommendations (userId, movieId, method, score) "
        "VALUES (%s, %s, %s, %s)"
    )
    cursor = conn.cursor()
    try:
        execute_many(cursor, sql, rows)
        conn.commit()
    except Error:
        conn.rollback()
        raise
    finally:
        cursor.close()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Bước 1: Load dữ liệu
    print("=" * 50)
    print(" IMPORT MOVIELENS → MYSQL")
    print("=" * 50)
    print("\nĐọc dữ liệu từ CSV ...")

    # Import data_loader từ cùng thư mục
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all

    movies_df, ratings_df, tags_df, links_df = load_all(DATA_DIR)

    # Bước 2: Kết nối MySQL
    print("\nKết nối MySQL ...")
    conn   = get_connection()
    cursor = conn.cursor()
    print(f"  Đã kết nối: {DB_CONFIG['host']}:{DB_CONFIG['port']}/"
          f"{DB_CONFIG['database']}")

    # Bước 3: Import từng bảng theo thứ tự (FK constraint)
    try:
        import_movies(cursor, movies_df)
        conn.commit()

        import_genres(cursor, movies_df)
        conn.commit()

        import_users(cursor, ratings_df, tags_df)
        conn.commit()

        import_ratings(cursor, ratings_df)
        conn.commit()

        import_tags(cursor, tags_df)
        conn.commit()

        import_links(cursor, links_df)
        conn.commit()

        print("\n[7/7] Kiểm tra số bản ghi trong database ...")
        for table in ["Users", "Movies", "Genres", "MovieGenres",
                      "Ratings", "Tags", "MovieLinks"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"      {table:<15}: {count:>8,} rows")

        print("\n[OK] Import hoàn tất!")

    except Error as e:
        conn.rollback()
        print(f"\n[LỖI] {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
