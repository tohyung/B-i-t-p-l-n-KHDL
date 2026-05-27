import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.data.loader import DataLoader
from src.data.preprocess import Preprocessor
from src.data.mappings import IDMapper
from src.data.sparse_builder import SparseMatrixBuilder
from src.retrieval.svd_retrieval import SVDRetriever
from src.ranking.genome_processor import GenomeProcessor
from src.ranking.feature_engineering import FeatureEngineer
from src.ranking.xgboost_ranker import RecommendationRanker
from src.reranking.lightweight_reranker import LightweightReranker
from src.explainability.explainer import Explainer
from src.evaluation.metrics import Evaluator


def load_artifacts():
    mapper = IDMapper()
    mapper.load()

    sparse_builder = SparseMatrixBuilder()
    sparse_builder.load()

    retriever = SVDRetriever()
    retriever.load(sparse_builder.user_means)

    genome_proc = GenomeProcessor()
    genome_proc.load()

    feature_eng = FeatureEngineer()
    feature_eng.load()

    ranker = None
    ranker_path = os.path.join("artifacts", "xgb_ranker.json")
    if os.path.exists(ranker_path):
        ranker = RecommendationRanker()
        ranker.load()

    return mapper, retriever, genome_proc, feature_eng, ranker


def build_user_profile_from_history(user_history, genome_proc, mapper, min_rating=3.5):
    liked_movie_indices = (
        user_history.loc[user_history['rating'] >= min_rating, 'movieId']
        .map(mapper.movie_to_idx)
        .dropna()
        .astype(int)
        .values
    )
    if len(liked_movie_indices) == 0:
        return np.zeros(genome_proc.num_tags, dtype=np.float32)

    user_profile = np.mean(genome_proc.movie_genome_matrix[liked_movie_indices], axis=0)
    norm = np.linalg.norm(user_profile)
    if norm > 0:
        user_profile = user_profile / norm
    return user_profile.astype(np.float32)


def generate_recommendations(
    user_id,
    ratings_df,
    mapper,
    retriever,
    genome_proc,
    feature_eng,
    ranker=None,
    top_k=10,
    candidate_top_n=100,
):
    if user_id not in mapper.user_to_idx:
        return [], None

    user_idx = mapper.user_to_idx[user_id]
    user_history = ratings_df[ratings_df['userId'] == user_id]
    watched_movie_ids = user_history['movieId'].values
    watched_indices = [mapper.movie_to_idx[m] for m in watched_movie_ids if m in mapper.movie_to_idx]

    candidate_indices, svd_scores = retriever.get_candidate_movies(
        user_idx,
        watched_indices,
        top_n=candidate_top_n,
    )
    candidate_movie_ids = [mapper.idx_to_movie[idx] for idx in candidate_indices]
    user_profile = build_user_profile_from_history(user_history, genome_proc, mapper)
    genome_sims = genome_proc.compute_similarity(user_profile, candidate_indices)
    features_df = feature_eng.generate_features(user_id, candidate_movie_ids, svd_scores, genome_sims)

    feature_rows = features_df.copy()
    feature_rows.insert(0, 'movieId', candidate_movie_ids)

    if ranker is not None:
        ranking_scores = ranker.predict(features_df)
    else:
        ranking_scores = (
            features_df['svd_score'].values * 0.6
            + features_df['genome_similarity'].values * 2.0
            + features_df['recent_trend_score'].values * 0.001
        )

    reranker = LightweightReranker(genome_proc)
    final_indices = reranker.rerank(candidate_indices, ranking_scores, top_n=top_k)
    final_movie_ids = [mapper.idx_to_movie[idx] for idx in final_indices]
    return final_movie_ids, feature_rows


def train_pipeline():
    print("="*60)
    print("STAGE 1: DATA LOADING & PREPROCESSING")
    print("="*60)
    
    loader = DataLoader("data/raw")
    
    # 1. Load dữ liệu
    ratings_df = loader.load_ratings()
    movies_df = loader.load_movies()
    genome_df = loader.load_genome_scores()
    
    # 2. Tiền xử lý và chia train/test
    movies_df = Preprocessor.extract_year_from_title(movies_df)
    train_ratings, test_ratings = Preprocessor.temporal_train_test_split(ratings_df, test_fraction=0.1) # Dùng 10% test để tiết kiệm RAM
    
    # Giải phóng bộ nhớ của ratings gốc
    del ratings_df
    
    # 3. Tạo mapping ID
    mapper = IDMapper()
    mapper.fit(train_ratings)
    mapper.save()
    
    # 4. Build sparse matrix an toàn
    sparse_builder = SparseMatrixBuilder()
    sparse_matrix = sparse_builder.build_and_demean(train_ratings, mapper)
    sparse_builder.save()
    
    print("\n" + "="*60)
    print("STAGE 2: RETRIEVAL & GENOME PROCESSING")
    print("="*60)
    
    # 5. Retrieval bằng SVD
    retriever = SVDRetriever(k=100)
    retriever.fit(sparse_matrix, sparse_builder.user_means)
    retriever.save()
    
    # Giải phóng sparse matrix sau khi đã train SVD
    del sparse_matrix
    
    # 6. Xử lý genome
    genome_proc = GenomeProcessor()
    genome_proc.fit(genome_df, mapper)
    genome_proc.save()
    del genome_df
    
    print("\n" + "="*60)
    print("STAGE 3: FEATURE ENGINEERING")
    print("="*60)
    
    # 7. Tạo feature nền
    feature_eng = FeatureEngineer()
    feature_eng.fit(train_ratings, movies_df)
    feature_eng.save()
    
    print("\n" + "="*60)
    print("STAGE 4: RANKING MODEL TRAINING (XGBOOST)")
    print("="*60)
    
    # 8. Train XGBoost Ranker trên tập learning-to-rank được sample.
    ranker = RecommendationRanker(
        n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "100")),
        learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.1")),
        max_depth=int(os.getenv("XGB_MAX_DEPTH", "6")),
    )
    X_train, y_train, qid_train = ranker.create_training_data(
        train_ratings,
        feature_eng,
        genome_proc,
        mapper,
        retriever,
        sample_users=int(os.getenv("XGB_SAMPLE_USERS", "1000")),
        max_positive_per_user=int(os.getenv("XGB_MAX_POSITIVE_PER_USER", "20")),
        negative_ratio=int(os.getenv("XGB_NEGATIVE_RATIO", "1")),
        min_user_ratings=int(os.getenv("XGB_MIN_USER_RATINGS", "10")),
        random_state=int(os.getenv("XGB_RANDOM_STATE", "42")),
    )
    ranker.fit(X_train, y_train, qid_train)
    ranker.save()
    
    print("\n>>> PIPELINE TRAINING COMPLETE AND ARTIFACTS SAVED! <<<")


def recommend_pipeline(user_id=1, top_k=10):
    print("="*60)
    print(f"RECOMMENDATION PIPELINE FOR USER {user_id}")
    print("="*60)
    
    try:
        mapper, retriever, genome_proc, feature_eng, ranker = load_artifacts()
    except Exception as exc:
        print("Error: Models not found. Please run 'train' first.")
        print(f"Details: {exc}")
        return
    
    # Kiểm tra user có trong mapping train hay không
    if user_id not in mapper.user_to_idx:
        print(f"User {user_id} not found in training data (Cold Start).")
        return
        
    user_idx = mapper.user_to_idx[user_id]
    
    # Load lịch sử phim user đã xem/rating
    loader = DataLoader("data/raw")
    ratings = loader.load_ratings()
    
    print("\n--- Phase 1: Candidate Retrieval (SVD) ---")
    
    print("\n--- Phase 2: Feature Extraction ---")
    
    print("\n--- Phase 3: ML Ranking (XGBoost) ---")
    if ranker is None:
        print("[Pipeline] XGBoost artifact not found. Falling back to weighted score.")
    
    print("\n--- Phase 4: Lightweight Re-ranking & Diversity ---")
    final_movie_ids, feature_rows = generate_recommendations(
        user_id,
        ratings,
        mapper,
        retriever,
        genome_proc,
        feature_eng,
        ranker=ranker,
        top_k=top_k,
        candidate_top_n=100,
    )
    
    print("\n" + "="*60)
    print(f"FINAL TOP {top_k} RECOMMENDATIONS FOR USER {user_id}")
    print("="*60)
    
    movies_df = loader.load_movies()
    movie_titles = movies_df.set_index('movieId')['title'].to_dict()
    
    for i, raw_movie_id in enumerate(final_movie_ids):
        title = movie_titles.get(raw_movie_id, f"Unknown Movie {raw_movie_id}")
        # Lấy feature để sinh giải thích
        feat_row = feature_rows[feature_rows['movieId'] == raw_movie_id].iloc[0].to_dict()
        explanation = Explainer.explain(feat_row)
        
        print(f"\n{i+1}. {title} (ID: {raw_movie_id})")
        print(f"   {explanation}")


def evaluate_pipeline(top_k=10):
    print("="*60)
    print("EVALUATION PIPELINE")
    print("="*60)

    try:
        mapper, retriever, genome_proc, feature_eng, ranker = load_artifacts()
    except Exception as exc:
        print("Error: Models not found. Please run 'train' first.")
        print(f"Details: {exc}")
        return

    loader = DataLoader("data/raw")
    ratings_df = loader.load_ratings()
    train_ratings, test_ratings = Preprocessor.temporal_train_test_split(
        ratings_df,
        test_fraction=float(os.getenv("EVAL_TEST_FRACTION", "0.1")),
    )
    del ratings_df

    relevance_threshold = float(os.getenv("EVAL_RELEVANCE_THRESHOLD", "4.0"))
    sample_users = int(os.getenv("EVAL_SAMPLE_USERS", "500"))
    rating_sample_rows = int(os.getenv("EVAL_RATING_ROWS", "100000"))
    candidate_top_n = int(os.getenv("EVAL_CANDIDATE_TOP_N", "100"))
    random_state = int(os.getenv("EVAL_RANDOM_STATE", "42"))

    print("\n--- Rating Prediction Metrics (SVD) ---")
    mapped_users = set(mapper.user_to_idx)
    mapped_movies = set(mapper.movie_to_idx)
    mapped_test = test_ratings[
        test_ratings['userId'].isin(mapped_users)
        & test_ratings['movieId'].isin(mapped_movies)
    ].copy()
    if len(mapped_test) > rating_sample_rows:
        mapped_test = mapped_test.sample(n=rating_sample_rows, random_state=random_state)

    y_true = []
    y_pred = []
    for user_id, group in mapped_test.groupby('userId', sort=False):
        user_idx = mapper.user_to_idx[user_id]
        movie_indices = [mapper.movie_to_idx[m] for m in group['movieId'].values]
        preds = retriever.predict_for_user_items(user_idx, movie_indices)
        y_true.extend(group['rating'].values.astype(np.float32))
        y_pred.extend(np.clip(preds, 0.5, 5.0))

    if y_true:
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        print(f"RMSE: {Evaluator.rmse(y_true, y_pred):.4f}")
        print(f"MAE : {Evaluator.mae(y_true, y_pred):.4f}")
        print(f"Rating rows evaluated: {len(y_true):,}")
    else:
        print("No mapped test ratings available for RMSE/MAE.")

    print("\n--- Top-K Ranking Metrics ---")
    relevant_df = test_ratings[
        (test_ratings['rating'] >= relevance_threshold)
        & test_ratings['userId'].isin(mapped_users)
        & test_ratings['movieId'].isin(mapped_movies)
    ]
    relevant_by_user = relevant_df.groupby('userId')['movieId'].apply(list)
    candidate_users = relevant_by_user.index.to_numpy()

    if len(candidate_users) == 0:
        print("No users with relevant test items for top-k evaluation.")
        return

    rng = np.random.default_rng(random_state)
    sampled_users = rng.choice(candidate_users, size=min(sample_users, len(candidate_users)), replace=False)
    train_user_groups = dict(tuple(train_ratings[train_ratings['userId'].isin(sampled_users)].groupby('userId')))

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    evaluated_users = 0

    for i, user_id in enumerate(sampled_users, start=1):
        user_history = train_user_groups.get(user_id)
        if user_history is None or user_history.empty:
            continue

        recommended_items, _ = generate_recommendations(
            int(user_id),
            user_history,
            mapper,
            retriever,
            genome_proc,
            feature_eng,
            ranker=ranker,
            top_k=top_k,
            candidate_top_n=candidate_top_n,
        )
        if not recommended_items:
            continue

        relevant_items = relevant_by_user.loc[user_id]
        precision_scores.append(Evaluator.precision_at_k(recommended_items, relevant_items, k=top_k))
        recall_scores.append(Evaluator.recall_at_k(recommended_items, relevant_items, k=top_k))
        ndcg_scores.append(Evaluator.ndcg_at_k(recommended_items, relevant_items, k=top_k))
        evaluated_users += 1

        if i % 100 == 0:
            print(f"[Evaluation] Processed {i:,}/{len(sampled_users):,} sampled users...")

    if evaluated_users == 0:
        print("No users could be evaluated for top-k metrics.")
        return

    print(f"Precision@{top_k}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{top_k}   : {np.mean(recall_scores):.4f}")
    print(f"NDCG@{top_k}     : {np.mean(ndcg_scores):.4f}")
    print(f"Users evaluated : {evaluated_users:,}")
    print(f"Relevant threshold: rating >= {relevance_threshold}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|recommend|evaluate]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "train":
        train_pipeline()
    elif command == "recommend":
        uid = int(input("Enter User ID to recommend for: "))
        recommend_pipeline(user_id=uid)
    elif command == "evaluate":
        evaluate_pipeline(top_k=int(os.getenv("EVAL_TOP_K", "10")))
    else:
        print("Unknown command.")
