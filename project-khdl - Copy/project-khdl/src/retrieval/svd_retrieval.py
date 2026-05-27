import numpy as np
from scipy.sparse.linalg import svds
import joblib
import os

class SVDRetriever:
    """
    Collaborative Filtering bằng SVD để sinh candidate.
    Chỉ dùng các mảng tiết kiệm bộ nhớ và dự đoán theo nhu cầu.
    """
    def __init__(self, k=100, artifacts_dir="artifacts"):
        self.k = k
        self.artifacts_dir = artifacts_dir
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_means = None

    def fit(self, sparse_matrix, user_means):
        """
        Train Truncated SVD trên sparse matrix đã demean.
        """
        print(f"[SVDRetriever] Training SVD with k={self.k} on matrix of shape {sparse_matrix.shape}...")
        
        # Đảm bảo k nhỏ hơn min(shape)
        max_k = min(sparse_matrix.shape) - 1
        actual_k = min(self.k, max_k)
        
        # svds xử lý scipy sparse matrix hiệu quả
        U, sigma, Vt = svds(sparse_matrix, k=actual_k)
        
        # svds trả singular values theo thứ tự tăng dần, nên cần đảo lại
        idx = np.argsort(sigma)[::-1]
        self.U = U[:, idx]
        self.sigma = sigma[idx]
        self.Vt = Vt[idx, :]
        self.user_means = user_means
        
        print("[SVDRetriever] Training complete.")

    def get_candidate_movies(self, user_idx, watched_movie_indices, top_n=100):
        """
        Lấy Top-N candidate movie cho một user bằng partial sort.
        Không reconstruct toàn bộ dense prediction matrix.
        
        Args:
            user_idx: Index nội bộ của user.
            watched_movie_indices: Danh sách index movie user đã tương tác.
            top_n: Số lượng candidate cần lấy.
            
        Returns:
            candidate_indices (list), svd_scores (numpy array)
        """
        # Dự đoán theo nhu cầu cho một user: (1, k) @ (k, k) @ (k, num_movies)
        # Thực tế U[user_idx] có shape (k,), sigma là (k,), Vt là (k, num_movies)
        # Điểm = U[user_idx] * sigma @ Vt + mean
        
        user_latent = self.U[user_idx] * self.sigma  # Nhân từng phần tử, kích thước (k,)
        predicted_scores = user_latent @ self.Vt  # Dot product, kích thước (num_movies,)
        predicted_scores += self.user_means[user_idx]  # Cộng lại user mean
        
        # Loại các phim user đã xem
        predicted_scores[list(watched_movie_indices)] = -np.inf
        
        # Partial sort nhanh bằng argpartition
        # argpartition O(n), nhanh hơn full sort O(n log n) khi số phim lớn
        if top_n >= len(predicted_scores):
            top_n = len(predicted_scores) - 1
            
        # argpartition đưa top_n phần tử về cuối mảng nhưng chưa sort nội bộ
        partitioned_idx = np.argpartition(predicted_scores, -top_n)[-top_n:]
        
        # Chỉ sort đầy đủ trên top_n phần tử
        top_scores = predicted_scores[partitioned_idx]
        sorted_top_idx = np.argsort(top_scores)[::-1]
        
        candidate_indices = partitioned_idx[sorted_top_idx]
        candidate_scores = top_scores[sorted_top_idx]
        
        return candidate_indices.tolist(), candidate_scores

    def predict_for_user_items(self, user_idx, movie_indices):
        """
        Dự đoán rating SVD đã reconstruct cho một user và một tập movie.
        """
        movie_indices = np.asarray(movie_indices, dtype=np.int64)
        if len(movie_indices) == 0:
            return np.array([], dtype=np.float32)

        user_latent = self.U[user_idx] * self.sigma
        scores = user_latent @ self.Vt[:, movie_indices]
        scores = scores + self.user_means[user_idx]
        return scores.astype(np.float32)

    def save(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "svd_model.joblib")
        joblib.dump({
            'U': self.U,
            'sigma': self.sigma,
            'Vt': self.Vt
        }, path)
        print(f"[SVDRetriever] SVD model saved to {path}")

    def load(self, user_means):
        path = os.path.join(self.artifacts_dir, "svd_model.joblib")
        model = joblib.load(path)
        self.U = model['U']
        self.sigma = model['sigma']
        self.Vt = model['Vt']
        self.user_means = user_means
        print(f"[SVDRetriever] SVD model loaded from {path}")
