import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class GenomeProcessor:
    """
    Xử lý genome-scores để tạo semantic vector cho movie.
    Tính similarity theo nhu cầu để tránh tạo ma trận dense O(N^2).
    """
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        self.movie_genome_matrix = None # Kích thước: (num_mapped_movies, 1128)
        self.num_tags = 1128 # Số tag cố định trong MovieLens

    def fit(self, genome_scores_df, mapper):
        """
        Build dense matrix (num_movies, 1128).
        Kích thước này vẫn nằm trong mức RAM chấp nhận được.
        """
        print("[GenomeProcessor] Building movie genome vectors...")
        num_movies = len(mapper.movie_to_idx)
        
        # Dùng float32 để tiết kiệm bộ nhớ
        self.movie_genome_matrix = np.zeros((num_movies, self.num_tags), dtype=np.float32)
        
        # genome_scores_df gồm movieId, tagId, relevance
        # Tag ID trong genome-tags.csv bắt đầu từ 1
        
        # Chỉ giữ các movie tồn tại trong mapping train
        valid_mask = genome_scores_df['movieId'].isin(mapper.movie_to_idx)
        df_filtered = genome_scores_df[valid_mask].copy()
        
        # Map ID gốc sang index nội bộ
        df_filtered['movie_idx'] = df_filtered['movieId'].map(mapper.movie_to_idx)
        df_filtered['tag_idx'] = df_filtered['tagId'] - 1 # Chuyển về 0-indexed
        
        movie_indices = df_filtered['movie_idx'].values
        tag_indices = df_filtered['tag_idx'].values
        relevance = df_filtered['relevance'].values
        
        # Gán nhanh bằng numpy
        self.movie_genome_matrix[movie_indices, tag_indices] = relevance
        
        # L2 normalize vector movie để dot product chính là cosine similarity
        norms = np.linalg.norm(self.movie_genome_matrix, axis=1, keepdims=True)
        # Tránh chia cho 0
        norms[norms == 0] = 1.0
        self.movie_genome_matrix = self.movie_genome_matrix / norms
        
        print(f"[GenomeProcessor] Constructed normalized genome matrix of shape {self.movie_genome_matrix.shape}")

    def build_user_profile(self, user_idx, train_ratings_df, mapper, min_rating=3.5):
        """
        Build semantic profile cho user bằng cách lấy trung bình genome vector
        của các phim user thích.
        """
        raw_user_id = mapper.idx_to_user.get(user_idx)
        if raw_user_id is None:
            return np.zeros(self.num_tags, dtype=np.float32)
            
        user_history = train_ratings_df[(train_ratings_df['userId'] == raw_user_id) & 
                                      (train_ratings_df['rating'] >= min_rating)]
        
        if user_history.empty:
            return np.zeros(self.num_tags, dtype=np.float32)
            
        liked_movie_indices = user_history['movieId'].map(mapper.movie_to_idx).dropna().astype(int).values
        if len(liked_movie_indices) == 0:
            return np.zeros(self.num_tags, dtype=np.float32)
            
        # Lấy vector và tính trung bình
        liked_vectors = self.movie_genome_matrix[liked_movie_indices]
        user_profile = np.mean(liked_vectors, axis=0)
        
        # L2 normalize user profile
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm
            
        return user_profile

    def compute_similarity(self, user_profile, candidate_movie_indices):
        """
        Tính cosine similarity giữa user profile và candidate movie.
        Vì cả hai đã L2-normalize, cosine similarity chính là dot product.
        """
        if np.sum(user_profile) == 0:
            # User chưa có profile, trả similarity bằng 0
            return np.zeros(len(candidate_movie_indices), dtype=np.float32)
            
        candidate_vectors = self.movie_genome_matrix[candidate_movie_indices]
        
        # Dot product: (num_candidates, 1128) @ (1128,) -> (num_candidates,)
        similarities = candidate_vectors @ user_profile
        return similarities

    def save(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "genome_matrix.joblib")
        joblib.dump(self.movie_genome_matrix, path)
        print(f"[GenomeProcessor] Genome matrix saved to {path}")

    def load(self):
        path = os.path.join(self.artifacts_dir, "genome_matrix.joblib")
        self.movie_genome_matrix = joblib.load(path)
        print(f"[GenomeProcessor] Genome matrix loaded from {path}")
