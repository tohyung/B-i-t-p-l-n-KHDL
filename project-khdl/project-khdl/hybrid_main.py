import os
import sys
import warnings
import pandas as pd

# Bỏ qua cảnh báo liên quan đến SQLAlchemy khi dùng Pandas với raw MySQL connection
warnings.filterwarnings("ignore", category=UserWarning)

# Import module Content-based từ thư mục src
from src.content_based import ContentBasedRecommender
# Import các module vừa viết
from src.collaborative_filtering import SVDRecommender
from src.hybrid_recommender import HybridRecommender

# Tái sử dụng kết nối database từ code của Người 1
from import_data import get_connection

# Tránh lỗi hiển thị ký tự đặc biệt trên terminal của Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def load_recommenders_from_db():
    """
    Kết nối MySQL, tải dữ liệu ra DataFrame và khởi tạo các hệ thống gợi ý.
    Tuyệt đối không sửa file src/content_based.py mà dùng thủ thuật tạo object (__new__)
    để nạp dữ liệu từ DB thay vì đọc file CSV theo cơ chế mặc định.
    """
    print("\n[HỆ THỐNG] Đang kết nối database MySQL...")
    conn = get_connection()
    print("[HỆ THỐNG] Kết nối thành công! Đang tải dữ liệu để train mô hình...\n")
    
    # 1. Tải và thiết lập Content-Based Recommender (Người 2)
    print("  -> Tải dữ liệu Phim và Tags từ DB (Content-based)...")
    # Lấy thông tin phim và nối các genres bằng dấu "|" giống định dạng movies.csv
    movies_query = """
        SELECT m.movieId, m.title, IFNULL(GROUP_CONCAT(g.name SEPARATOR '|'), '') AS genres
        FROM Movies m
        LEFT JOIN MovieGenres mg ON m.movieId = mg.movieId
        LEFT JOIN Genres g ON mg.genreId = g.genreId
        GROUP BY m.movieId, m.title
    """
    movies_df = pd.read_sql(movies_query, conn)
    
    # Lấy thông tin tags
    tags_query = "SELECT userId, movieId, tag FROM Tags"
    tags_df = pd.read_sql(tags_query, conn)
    
    # Khởi tạo giả mạo để lướt qua hàm __init__ đọc CSV của file cũ
    cb_model = ContentBasedRecommender.__new__(ContentBasedRecommender)
    cb_model.movies_path = None
    cb_model.tags_path = None
    cb_model.movies = movies_df
    cb_model.tags = tags_df
    # Tái sử dụng các hàm có sẵn
    cb_model.preprocess_data()
    cb_model.build_model()
    
    # 2. Tải và thiết lập SVD Recommender (Người 3)
    print("  -> Tải dữ liệu Ratings (SVD Collaborative Filtering)...")
    ratings_query = "SELECT userId, movieId, rating FROM Ratings"
    ratings_df = pd.read_sql(ratings_query, conn)
    
    conn.close()
    
    print("  -> Đang huấn luyện ma trận SVD...")
    svd_model = SVDRecommender(k=50)
    svd_model.fit(ratings_df)
    
    print("\n[HỆ THỐNG] Huấn luyện hoàn tất!\n")
    return cb_model, svd_model

def print_hybrid_recommendations(response):
    """
    Hàm in kết quả Hybrid tương tự như giao diện mẫu của Người 2
    """
    if response.get("status") != "success":
        print("\nMovie not found.")
        if "suggestions" in response:
            suggestions = response["suggestions"]
            if suggestions.empty:
                print("No similar movie title found.")
            else:
                print("\nDid you mean:")
                for row in suggestions.itertuples(index=False):
                    print(f"- {row.title} | Movie ID: {row.movieId} | Genres: {row.genres}")
        return

    input_movie = response["input_movie"]
    user_id = response["user_id"]
    recommendations = response["recommendations"]

    print("\n" + "=" * 105)
    print("HYBRID RECOMMENDATION (SVD Collaborative Filtering + Content-based)")
    print("Weight: 0.7 * SVD_Score + 0.3 * Content_Score")
    print("=" * 105)

    print(f"\nUser ID      : {user_id}")
    print(f"Input Movie  : {input_movie['title']} (ID: {input_movie['movieId']})")

    print("\nRecommended Movies:\n")
    print(f"{'Rank':<5} {'Movie ID':<9} {'Title':<40} {'SVD Score':<12} {'Content':<12} {'Final Score':<12}")
    print("-" * 105)

    for rank, row in enumerate(recommendations.itertuples(index=False), start=1):
        title = row.title
        if len(title) > 38:
            title = title[:35] + "..."

        print(
            f"{rank:<5} "
            f"{row.movieId:<9} "
            f"{title:<40} "
            f"{row.svd_score:>8.4f}     "
            f"{row.content_score:>7.4f}      "
            f"{row.final_score:>9.4f}"
        )

def main():
    try:
        cb_model, svd_model = load_recommenders_from_db()
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Không thể tải dữ liệu và khởi tạo model. Lỗi: {e}")
        print("Hãy chắc chắn rằng database MySQL đang chạy và thông tin đăng nhập trong 'import_data.py' là chính xác.")
        return

    hybrid_system = HybridRecommender(cb_model, svd_model)
    
    while True:
        print("\n" + "=" * 60)
        print("MOVIELENS HYBRID RECOMMENDATION SYSTEM")
        print("=" * 60)
        print("1. Recommend movies (Hybrid: SVD + Content-Based)")
        print("0. Exit")

        choice = input("\nSelect option: ")

        if choice == "1":
            user_id_str = input("Enter User ID (e.g. 1): ")
            movie_title = input("Enter Favorite Movie Title (e.g. Toy Story): ")
            top_n_str = input("Enter number of recommendations (default 10): ")

            try:
                user_id = int(user_id_str)
            except ValueError:
                print("Invalid User ID. Defaulting to 1.")
                user_id = 1
                
            try:
                top_n = int(top_n_str)
            except ValueError:
                top_n = 10

            response = hybrid_system.recommend(user_id, movie_title, top_n)
            print_hybrid_recommendations(response)

        elif choice == "0":
            print("Exit program.")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
