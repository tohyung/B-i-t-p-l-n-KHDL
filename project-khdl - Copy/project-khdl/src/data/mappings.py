import joblib
import os

class IDMapper:
    """
    Tạo và lưu mapping số nguyên liên tục cho userId/movieId gốc.
    Mapping này cần thiết để build sparse matrix hiệu quả.
    """
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

    def fit(self, ratings_df):
        """
        Tạo mapping dựa trên các user và movie xuất hiện trong tập train.
        """
        print("[Mappings] Building ID mappings...")
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        
        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = {i: m for i, m in enumerate(unique_movies)}
        
        print(f"[Mappings] Mapped {len(unique_users):,} users and {len(unique_movies):,} movies.")

    def save(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "id_mappings.joblib")
        joblib.dump({
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'movie_to_idx': self.movie_to_idx,
            'idx_to_movie': self.idx_to_movie
        }, path)
        print(f"[Mappings] Mappings saved to {path}")

    def load(self):
        path = os.path.join(self.artifacts_dir, "id_mappings.joblib")
        mappings = joblib.load(path)
        self.user_to_idx = mappings['user_to_idx']
        self.idx_to_user = mappings['idx_to_user']
        self.movie_to_idx = mappings['movie_to_idx']
        self.idx_to_movie = mappings['idx_to_movie']
        print(f"[Mappings] Mappings loaded from {path}")
