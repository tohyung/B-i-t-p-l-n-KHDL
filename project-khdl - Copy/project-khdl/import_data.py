"""
import_data.py

Người 1: Data + Database + EDA
Chức năng: Import dữ liệu MovieLens đã load sạch vào MySQL.
Yêu cầu  : pip install mysql-connector-python pandas
Cách dùng: python import_data.py
           Chỉnh DB_CONFIG bên dưới trước khi chạy.
"""

import os
import sys
from typing import Optional
import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error
except ImportError:
    mysql = None
    Error = Exception

# Tránh lỗi UnicodeEncodeError khi chạy trên Windows terminal dùng code page cũ.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Cấu hình kết nối MySQL. Chỉnh các giá trị này cho đúng máy của bạn.
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",          # Tên user MySQL
    "password": "123456",    # Mật khẩu MySQL
    "database": "movielens",
    "charset": "utf8mb4",
}

# Đường dẫn đến thư mục CSV.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def get_connection():
    """Tạo kết nối MySQL và dừng chương trình nếu chưa cài connector."""
    if mysql is None:
        print("[LOI] Chua cai mysql-connector-python. Chay: pip install mysql-connector-python")
        sys.exit(1)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"[LOI] Khong the ket noi MySQL: {e}")
        sys.exit(1)


def execute_many(cursor, sql: str, data: list, batch_size: int = 2000):
    """Insert theo batch để tránh quá tải bộ nhớ."""
    total = len(data)
    for i in range(0, total, batch_size):
        cursor.executemany(sql, data[i : i + batch_size])


def import_movies(cursor, movies_df: pd.DataFrame):
    """Import bảng Movies."""
    print("\n[1/7] Import bang Movies ...")
    rows = [(int(r.movieId), r.title) for r in movies_df.itertuples()]
    sql = "INSERT IGNORE INTO Movies (movieId, title) VALUES (%s, %s)"
    execute_many(cursor, sql, rows)
    print(f"      -> {len(rows):,} phim da insert.")


def import_genres(cursor, movies_df: pd.DataFrame):
    """Import bảng Genres và bảng nối MovieGenres."""
    print("[2/7] Import bang Genres + MovieGenres ...")

    # Thu thập toàn bộ genre duy nhất từ movies_df.
    all_genres = sorted({g for genres in movies_df["genres"] for g in genres})
    genre_sql = "INSERT IGNORE INTO Genres (name) VALUES (%s)"
    execute_many(cursor, genre_sql, [(g,) for g in all_genres])

    # Lấy lại genreId từ database sau khi insert.
    cursor.execute("SELECT genreId, name FROM Genres")
    genre_map = {name: gid for gid, name in cursor.fetchall()}

    # Tạo dữ liệu bảng nối movie-genre.
    mg_rows = [
        (int(r.movieId), genre_map[g])
        for r in movies_df.itertuples()
        for g in r.genres
        if g in genre_map
    ]
    mg_sql = "INSERT IGNORE INTO MovieGenres (movieId, genreId) VALUES (%s, %s)"
    execute_many(cursor, mg_sql, mg_rows)
    print(f"      -> {len(all_genres)} genres, {len(mg_rows):,} movie-genre links.")


def import_users(cursor, ratings_df: pd.DataFrame, tags_df: Optional[pd.DataFrame] = None):
    """Import bảng Users từ ratings và tags."""
    print("[3/7] Import bang Users ...")
    user_ids = set(ratings_df["userId"].unique())
    if tags_df is not None:
        user_ids.update(tags_df["userId"].unique())
    rows = [(int(uid),) for uid in sorted(user_ids)]
    sql = "INSERT IGNORE INTO Users (userId) VALUES (%s)"
    execute_many(cursor, sql, rows)
    print(f"      -> {len(rows):,} users da insert.")


def import_ratings(cursor, ratings_df: pd.DataFrame):
    """Import bảng Ratings."""
    print("[4/7] Import bang Ratings ...")
    rows = [
        (int(r.userId), int(r.movieId), float(r.rating), r.ratedAt.strftime("%Y-%m-%d %H:%M:%S"))
        for r in ratings_df.itertuples()
    ]
    sql = (
        "INSERT IGNORE INTO Ratings (userId, movieId, rating, ratedAt) "
        "VALUES (%s, %s, %s, %s)"
    )
    execute_many(cursor, sql, rows)
    print(f"      -> {len(rows):,} ratings da insert.")


def import_tags(cursor, tags_df: pd.DataFrame):
    """Import bảng Tags."""
    print("[5/7] Import bang Tags ...")
    rows = [
        (int(r.userId), int(r.movieId), r.tag, r.taggedAt.strftime("%Y-%m-%d %H:%M:%S"))
        for r in tags_df.itertuples()
    ]
    sql = (
        "INSERT IGNORE INTO Tags (userId, movieId, tag, taggedAt) "
        "VALUES (%s, %s, %s, %s)"
    )
    execute_many(cursor, sql, rows)
    print(f"      -> {len(rows):,} tags da insert.")


def import_links(cursor, links_df: pd.DataFrame):
    """Import bảng MovieLinks."""
    print("[6/7] Import bang MovieLinks ...")
    rows = []
    for r in links_df.itertuples():
        tmdb = None if pd.isna(r.tmdbId) else int(r.tmdbId)
        rows.append((int(r.movieId), str(r.imdbId), tmdb))
    sql = "INSERT IGNORE INTO MovieLinks (movieId, imdbId, tmdbId) VALUES (%s, %s, %s)"
    execute_many(cursor, sql, rows)
    print(f"      -> {len(rows):,} links da insert.")


def save_recommendations(conn, user_id: int, recommendations, method: str):
    """Lưu lịch sử recommendation vào bảng Recommendations.

    ``recommendations`` có thể là list dict/DataFrame row chứa movieId và score,
    hoặc list tuple dạng ``(movieId, score)``.
    """
    valid_methods = {"content_based", "collaborative", "hybrid"}
    if method not in valid_methods:
        raise ValueError(f"method phai thuoc {sorted(valid_methods)}")

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


def main():
    """Import MovieLens CSV vào MySQL theo đúng thứ tự khóa ngoại."""
    if mysql is None:
        print("[LOI] Chua cai mysql-connector-python. Chay: pip install mysql-connector-python")
        sys.exit(1)

    print("=" * 50)
    print(" IMPORT MOVIELENS -> MYSQL")
    print("=" * 50)
    print("\nDoc du lieu tu CSV ...")

    # Import data_loader từ cùng thư mục.
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all

    movies_df, ratings_df, tags_df, links_df = load_all(DATA_DIR)

    print("\nKet noi MySQL ...")
    conn = get_connection()
    cursor = conn.cursor()
    print(f"  Da ket noi: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

    # Import theo thứ tự để thỏa mãn ràng buộc khóa ngoại.
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

        print("\n[7/7] Kiem tra so ban ghi trong database ...")
        for table in ["Users", "Movies", "Genres", "MovieGenres", "Ratings", "Tags", "MovieLinks"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"      {table:<15}: {count:>8,} rows")

        print("\n[OK] Import hoan tat!")

    except Error as e:
        conn.rollback()
        print(f"\n[LOI] {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
