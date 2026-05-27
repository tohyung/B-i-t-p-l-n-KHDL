import pandas as pd
import numpy as np
import os

class DataLoader:
    """
    Loader tối ưu cho dữ liệu MovieLens kích thước lớn.
    Ép dtype để giảm RAM khi xử lý hàng chục triệu rating.
    """
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir

    def load_ratings(self):
        """
        Load ratings.csv với dtype tối ưu bộ nhớ.
        userId -> int32, movieId -> int32, rating -> float32, timestamp -> int64
        """
        filepath = os.path.join(self.data_dir, "ratings.csv")
        print(f"[Loader] Loading ratings from {filepath}...")
        
        dtype_dict = {
            'userId': np.int32,
            'movieId': np.int32,
            'rating': np.float32,
            'timestamp': np.int64
        }
        
        df = pd.read_csv(filepath, dtype=dtype_dict)
        print(f"[Loader] Loaded {len(df):,} ratings. Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        return df

    def load_movies(self):
        """
        Load movies.csv.
        movieId -> int32
        """
        filepath = os.path.join(self.data_dir, "movies.csv")
        dtype_dict = {'movieId': np.int32}
        df = pd.read_csv(filepath, dtype=dtype_dict)
        print(f"[Loader] Loaded {len(df):,} movies.")
        return df

    def load_genome_scores(self):
        """
        Load genome-scores.csv.
        movieId -> int32, tagId -> int32, relevance -> float32
        """
        filepath = os.path.join(self.data_dir, "genome-scores.csv")
        dtype_dict = {
            'movieId': np.int32,
            'tagId': np.int32,
            'relevance': np.float32
        }
        df = pd.read_csv(filepath, dtype=dtype_dict)
        print(f"[Loader] Loaded {len(df):,} genome scores. Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        return df
