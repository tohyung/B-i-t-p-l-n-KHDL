import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import joblib
import os

class SparseMatrixBuilder:
    """
    Build scipy.sparse.csr_matrix theo cách tiết kiệm bộ nhớ.
    Không tạo dense matrix.
    Chỉ mean-center trên các phần tử rating tồn tại.
    """
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.user_means = None

    def build_and_demean(self, ratings_df, mapper):
        """
        Build sparse matrix đã demean để làm input cho SVD.
        
        Args:
            ratings_df: DataFrame chứa userId, movieId, rating.
            mapper: IDMapper đã fit.
            
        Returns:
            scipy.sparse.csr_matrix đã demean.
        """
        print("[SparseBuilder] Calculating user means...")
        # 1. Tính rating trung bình của từng user
        user_means_series = ratings_df.groupby('userId')['rating'].mean()
        
        # Lưu user mean để cộng lại khi reconstruct prediction
        # Map userId gốc sang user_idx nội bộ để khớp với hàng của matrix
        num_users = len(mapper.user_to_idx)
        self.user_means = np.zeros(num_users, dtype=np.float32)
        
        for raw_uid, mean_val in user_means_series.items():
            if raw_uid in mapper.user_to_idx:
                idx = mapper.user_to_idx[raw_uid]
                self.user_means[idx] = mean_val

        print("[SparseBuilder] Demeaning ratings efficiently...")
        # 2. Trừ mean trực tiếp trên giá trị rating trước khi build sparse matrix
        # Map user mean về từng dòng rating
        ratings_df['user_mean'] = ratings_df['userId'].map(user_means_series)
        ratings_df['demeaned_rating'] = (ratings_df['rating'] - ratings_df['user_mean']).astype(np.float32)

        print("[SparseBuilder] Mapping IDs to continuous indices...")
        # Map ID gốc sang index liên tục; drop dòng không map được để an toàn
        mapped_df = ratings_df.dropna(subset=['demeaned_rating']).copy()
        user_indices = mapped_df['userId'].map(mapper.user_to_idx).values
        movie_indices = mapped_df['movieId'].map(mapper.movie_to_idx).values
        demeaned_values = mapped_df['demeaned_rating'].values

        print("[SparseBuilder] Constructing scipy.sparse.csr_matrix...")
        # 3. Build sparse matrix chỉ từ các rating đã demean
        num_movies = len(mapper.movie_to_idx)
        
        sparse_matrix = csr_matrix(
            (demeaned_values, (user_indices, movie_indices)),
            shape=(num_users, num_movies),
            dtype=np.float32
        )
        
        print(f"[SparseBuilder] Matrix constructed: {sparse_matrix.shape} with {sparse_matrix.nnz:,} stored elements.")
        
        # Xóa cột tạm để giải phóng bộ nhớ
        ratings_df.drop(columns=['user_mean', 'demeaned_rating'], inplace=True)
        
        return sparse_matrix

    def save(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "user_means.joblib")
        joblib.dump(self.user_means, path)
        print(f"[SparseBuilder] User means saved to {path}")

    def load(self):
        path = os.path.join(self.artifacts_dir, "user_means.joblib")
        self.user_means = joblib.load(path)
        print(f"[SparseBuilder] User means loaded from {path}")
