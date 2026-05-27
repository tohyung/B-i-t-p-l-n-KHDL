import pandas as pd
import numpy as np

class Preprocessor:
    """
    Xử lý các bước tiền xử lý như chia dữ liệu theo thời gian và làm sạch cơ bản.
    """
    
    @staticmethod
    def temporal_train_test_split(ratings_df, test_fraction=0.2):
        """
        Chia ratings thành train/test theo timestamp toàn cục.
        Tương tác cũ hơn -> Train
        Tương tác mới hơn -> Test
        Cách chia này mô phỏng đúng bối cảnh production hơn random split.
        """
        print(f"[Preprocess] Performing temporal split (test_fraction={test_fraction})...")
        
        # Sắp xếp toàn bộ dữ liệu theo timestamp để tạo timeline
        ratings_df = ratings_df.sort_values('timestamp')
        
        split_idx = int(len(ratings_df) * (1 - test_fraction))
        
        train_df = ratings_df.iloc[:split_idx].copy()
        test_df = ratings_df.iloc[split_idx:].copy()
        
        print(f"[Preprocess] Train size: {len(train_df):,}, Test size: {len(test_df):,}")
        return train_df, test_df

    @staticmethod
    def extract_year_from_title(movies_df):
        """
        Tách năm phát hành từ title phim, ví dụ 'Toy Story (1995)' -> 1995.
        """
        # Tách 4 chữ số nằm trong ngoặc ở gần cuối chuỗi
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        # Điền năm bị thiếu bằng median
        movies_df['year'] = movies_df['year'].fillna(movies_df['year'].median())
        return movies_df
