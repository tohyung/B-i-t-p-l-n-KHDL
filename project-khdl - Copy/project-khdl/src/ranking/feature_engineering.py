import pandas as pd
import numpy as np
import joblib
import os

class FeatureEngineer:
    """
    Tính các feature collaborative, movie, temporal và content.
    Dùng để tạo dữ liệu đầu vào cho XGBoost Ranker.
    """
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        
        # Metadata đã tính sẵn
        self.user_stats = None  # DataFrame: userId -> user_avg_rating, user_activity_level
        self.movie_stats = None # DataFrame: movieId -> avg_movie_rating, rating_count, popularity_rank
        self.movie_metadata = None # DataFrame: movieId -> year, genres
        self.user_liked_genres = None # Dict: userId -> tập genre user thích
        
    def fit(self, train_ratings_df, movies_df):
        """
        Tính trước các thống kê global chỉ dựa trên tập train.
        """
        print("[FeatureEngineer] Computing base statistics...")
        
        # 1. Thống kê user
        self.user_stats = train_ratings_df.groupby('userId').agg(
            user_avg_rating=('rating', 'mean'),
            user_activity_level=('rating', 'count')
        ).reset_index()
        
        # 2. Thống kê movie
        self.movie_stats = train_ratings_df.groupby('movieId').agg(
            avg_movie_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        
        # Thứ hạng phổ biến: 1 là phim phổ biến nhất
        self.movie_stats['movie_popularity_rank'] = self.movie_stats['rating_count'].rank(ascending=False, method='min')
        
        # Xu hướng thời gian: số rating trong 30 ngày cuối của timeline train
        max_timestamp = train_ratings_df['timestamp'].max()
        thirty_days_sec = 30 * 24 * 60 * 60
        recent_df = train_ratings_df[train_ratings_df['timestamp'] >= (max_timestamp - thirty_days_sec)]
        
        recent_trend = recent_df.groupby('movieId').size().reset_index(name='recent_trend_score')
        self.movie_stats = pd.merge(self.movie_stats, recent_trend, on='movieId', how='left')
        self.movie_stats['recent_trend_score'] = self.movie_stats['recent_trend_score'].fillna(0)
        
        # 3. Metadata của movie
        self.movie_metadata = movies_df[['movieId', 'year', 'genres']].copy()
        # Chuyển chuỗi genre "Action|Adventure" thành list
        self.movie_metadata['genres_list'] = self.movie_metadata['genres'].fillna('').str.split('|')
        
        # 4. Genre user thích
        print("[FeatureEngineer] Computing user genre profiles...")
        # Lấy các phim user rating >= 3.5
        liked_interactions = train_ratings_df[train_ratings_df['rating'] >= 3.5][['userId', 'movieId']]
        liked_joined = pd.merge(liked_interactions, self.movie_metadata[['movieId', 'genres_list']], on='movieId')
        
        # Explode genre và lấy tập genre duy nhất theo user
        exploded = liked_joined.explode('genres_list')
        self.user_liked_genres = exploded.groupby('userId')['genres_list'].apply(set).to_dict()
        
        print("[FeatureEngineer] Base statistics computed.")

    def generate_features(self, user_id, candidate_movie_ids, svd_scores, genome_similarities):
        """
        Sinh ma trận feature cho một user và các candidate movie.
        
        Args:
            user_id: User ID gốc.
            candidate_movie_ids: Danh sách Movie ID gốc.
            svd_scores: Mảng điểm dự đoán SVD tương ứng candidate.
            genome_similarities: Mảng similarity theo genome.
            
        Returns:
            Pandas DataFrame chứa các feature sẵn sàng cho XGBoost.
        """
        df = pd.DataFrame({
            'userId': user_id,
            'movieId': candidate_movie_ids,
            'svd_score': svd_scores,
            'genome_similarity': genome_similarities
        })
        
        # Ghép thống kê user
        df = pd.merge(df, self.user_stats, on='userId', how='left')
        global_user_avg = self.user_stats['user_avg_rating'].mean() if not self.user_stats.empty else 3.0
        df['user_avg_rating'] = df['user_avg_rating'].fillna(global_user_avg)
        df['user_activity_level'] = df['user_activity_level'].fillna(0)
        
        # Ghép thống kê movie
        df = pd.merge(df, self.movie_stats, on='movieId', how='left')
        global_movie_avg = self.movie_stats['avg_movie_rating'].mean() if not self.movie_stats.empty else 3.0
        max_popularity_rank = self.movie_stats['movie_popularity_rank'].max() if not self.movie_stats.empty else 100000
        df['avg_movie_rating'] = df['avg_movie_rating'].fillna(global_movie_avg)
        df['rating_count'] = df['rating_count'].fillna(0)
        df['movie_popularity_rank'] = df['movie_popularity_rank'].fillna(max_popularity_rank)
        df['recent_trend_score'] = df['recent_trend_score'].fillna(0)
        
        # Ghép metadata movie
        df = pd.merge(df, self.movie_metadata, on='movieId', how='left')
        
        # Tính tuổi phim, giả định năm hiện tại là 2026
        df['movie_age'] = 2026 - df['year']
        median_year = self.movie_metadata['year'].median() if not self.movie_metadata.empty else 2000
        fallback_age = 2026 - median_year
        df['movie_age'] = df['movie_age'].fillna(fallback_age)
        df['recency_score'] = 1.0 / (df['movie_age'] + 1.0)  # Điểm recency dạng nghịch đảo đơn giản
        
        # Tính độ trùng genre
        user_genres = self.user_liked_genres.get(user_id, set())
        
        def compute_overlap(movie_genres):
            if not isinstance(movie_genres, list) or not movie_genres:
                return 0.0
            intersection = user_genres.intersection(set(movie_genres))
            # Dùng số genre trùng nhau để giữ cách tính đơn giản
            return float(len(intersection))
            
        df['genre_overlap'] = df['genres_list'].apply(compute_overlap)
        
        # Chọn các feature cuối cùng
        feature_cols = [
            'svd_score', 'user_avg_rating', 'user_activity_level',
            'avg_movie_rating', 'rating_count', 'movie_popularity_rank',
            'movie_age', 'recency_score', 'recent_trend_score',
            'genre_overlap', 'genome_similarity'
        ]
        
        return df[feature_cols].astype(np.float32)

    def save(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "feature_metadata.joblib")
        joblib.dump({
            'user_stats': self.user_stats,
            'movie_stats': self.movie_stats,
            'movie_metadata': self.movie_metadata,
            'user_liked_genres': self.user_liked_genres
        }, path)
        print(f"[FeatureEngineer] Feature metadata saved to {path}")

    def load(self):
        path = os.path.join(self.artifacts_dir, "feature_metadata.joblib")
        data = joblib.load(path)
        self.user_stats = data['user_stats']
        self.movie_stats = data['movie_stats']
        self.movie_metadata = data['movie_metadata']
        self.user_liked_genres = data['user_liked_genres']
        print(f"[FeatureEngineer] Feature metadata loaded from {path}")
