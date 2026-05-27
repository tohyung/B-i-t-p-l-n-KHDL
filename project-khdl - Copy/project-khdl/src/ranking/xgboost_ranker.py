import pandas as pd
import numpy as np
import joblib
import os

try:
    import xgboost as xgb
except ImportError:
    xgb = None

class RecommendationRanker:
    """
    XGBoost Ranker dùng pairwise loss cho bài toán gợi ý phim.
    """
    def __init__(self, artifacts_dir="artifacts", n_estimators=100, learning_rate=0.1, max_depth=6):
        self.artifacts_dir = artifacts_dir
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None

    def _require_xgboost(self):
        if xgb is None:
            raise ImportError(
                "xgboost is required to train/load RecommendationRanker. "
                "Install it with: pip install xgboost"
            )

    def create_training_data(
        self,
        train_ratings_df,
        feature_engineer,
        genome_processor,
        mapper,
        retriever,
        sample_users=1000,
        max_positive_per_user=20,
        negative_ratio=1,
        min_user_ratings=10,
        random_state=42,
    ):
        """
        Tạo tập train cho ranker.
        Để tránh tràn bộ nhớ, chỉ sample một phần user và dùng lịch sử tương tác
        của họ để build feature matrix.
        """
        print(f"[XGBRanker] Generating training data for {sample_users} sampled users...")
        rng = np.random.default_rng(random_state)
        
        # Sample các user có đủ số lượng rating tối thiểu
        user_counts = train_ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index.to_numpy()
        if len(valid_users) == 0:
            raise ValueError("No users have enough ratings to train the ranker.")

        sampled_users = rng.choice(valid_users, size=min(sample_users, len(valid_users)), replace=False)
        sampled_users = np.sort(sampled_users)
        all_movie_ids = np.array(list(mapper.movie_to_idx.keys()))
        sample_df = train_ratings_df[train_ratings_df['userId'].isin(sampled_users)].copy()
        user_groups = dict(tuple(sample_df.groupby('userId', sort=True)))

        feature_frames = []
        labels = []
        qids = []

        for n, user_id in enumerate(sampled_users, start=1):
            user_rows = user_groups.get(user_id)
            if user_rows is None:
                continue
            if user_rows.empty or user_id not in mapper.user_to_idx:
                continue

            positives = user_rows[['movieId', 'rating']].drop_duplicates('movieId')
            if len(positives) > max_positive_per_user:
                positives = positives.sample(n=max_positive_per_user, random_state=random_state)

            watched = set(user_rows['movieId'].values)
            negative_count = max(1, len(positives) * negative_ratio)
            negatives = []
            attempts = 0
            while len(negatives) < negative_count and attempts < negative_count * 20:
                sampled_movie = int(rng.choice(all_movie_ids))
                attempts += 1
                if sampled_movie not in watched:
                    negatives.append(sampled_movie)

            candidate_movie_ids = positives['movieId'].astype(int).tolist() + negatives
            if len(candidate_movie_ids) < 2:
                continue

            user_idx = mapper.user_to_idx[user_id]
            candidate_indices = [mapper.movie_to_idx[m] for m in candidate_movie_ids if m in mapper.movie_to_idx]
            candidate_movie_ids = [mapper.idx_to_movie[idx] for idx in candidate_indices]
            if len(candidate_indices) < 2:
                continue

            svd_scores = retriever.predict_for_user_items(user_idx, candidate_indices)
            user_profile = self._build_user_profile_from_rows(user_rows, genome_processor, mapper)
            genome_similarities = genome_processor.compute_similarity(user_profile, candidate_indices)
            feature_frame = feature_engineer.generate_features(
                int(user_id),
                candidate_movie_ids,
                svd_scores,
                genome_similarities,
            )

            positive_labels = positives.set_index('movieId')['rating'].to_dict()
            y = [float(positive_labels.get(movie_id, 0.0)) for movie_id in candidate_movie_ids]

            feature_frames.append(feature_frame)
            labels.extend(y)
            qids.extend([int(user_id)] * len(candidate_movie_ids))

            if n % 100 == 0:
                print(f"[XGBRanker] Prepared {n:,}/{len(sampled_users):,} users...")

        if not feature_frames:
            raise ValueError("No training rows were generated for the ranker.")

        X = pd.concat(feature_frames, ignore_index=True).astype(np.float32)
        y = np.asarray(labels, dtype=np.float32)
        qid = np.asarray(qids, dtype=np.int32)
        print(f"[XGBRanker] Training rows: {len(X):,}, groups: {len(np.unique(qid)):,}")
        return X, y, qid

    def _build_user_profile_from_rows(self, user_rows, genome_processor, mapper, min_rating=3.5):
        liked_movie_ids = user_rows.loc[user_rows['rating'] >= min_rating, 'movieId']
        liked_movie_indices = liked_movie_ids.map(mapper.movie_to_idx).dropna().astype(int).values
        if len(liked_movie_indices) == 0:
            return np.zeros(genome_processor.num_tags, dtype=np.float32)

        user_profile = np.mean(genome_processor.movie_genome_matrix[liked_movie_indices], axis=0)
        norm = np.linalg.norm(user_profile)
        if norm > 0:
            user_profile = user_profile / norm
        return user_profile.astype(np.float32)

    def fit(self, X, y, qid):
        """
        Train XGBRanker.
        X: Ma trận feature.
        y: Điểm relevance, ví dụ rating.
        qid: Query ID, ở đây là User ID, phải được group/sort đúng.
        """
        self._require_xgboost()
        print("[XGBRanker] Training XGBoost Ranker (rank:pairwise)...")
        
        self.model = xgb.XGBRanker(
            tree_method='hist',
            objective='rank:pairwise',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        
        # Huấn luyện model
        self.model.fit(X, y, qid=qid, verbose=True)
        print("[XGBRanker] Training complete.")

    def predict(self, X):
        """
        Trả ranking score cho các candidate.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict(X)

    def save(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        if not hasattr(self.model, "save_model"):
            raise ValueError("Current model object cannot be saved as an XGBoost model.")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "xgb_ranker.json")
        self.model.save_model(path)
        print(f"[XGBRanker] Model saved to {path}")

    def load(self):
        self._require_xgboost()
        path = os.path.join(self.artifacts_dir, "xgb_ranker.json")
        self.model = xgb.XGBRanker()
        self.model.load_model(path)
        print(f"[XGBRanker] Model loaded from {path}")
