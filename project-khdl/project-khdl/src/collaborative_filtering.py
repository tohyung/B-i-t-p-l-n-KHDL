import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

class SVDRecommender:
    """
    Hệ thống Collaborative Filtering sử dụng phân rã ma trận SVD.
    Người 3 phát triển, không phụ thuộc vào scikit-surprise.
    """
    def __init__(self, k=50):
        self.k = k
        self.user_item_matrix_df = None
        self.user_ratings_mean = None
        self.preds_matrix = None
        self.users_list = None
        self.movies_list = None

    def fit(self, ratings_df):
        """
        Huấn luyện mô hình SVD từ DataFrame ratings
        """
        # Xóa duplicate nếu có (giữ lần cuối user rate phim)
        ratings = ratings_df.drop_duplicates(subset=["userId", "movieId"], keep="last")
        
        # Tạo ma trận User-Item (Pivot Table): rows = userId, cols = movieId, values = rating
        self.user_item_matrix_df = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.users_list = list(self.user_item_matrix_df.index)
        self.movies_list = list(self.user_item_matrix_df.columns)
        
        # Chuyển đổi sang numpy array
        R = self.user_item_matrix_df.to_numpy()
        
        # Mean centering: trừ đi điểm trung bình của mỗi user để chuẩn hóa
        self.user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - self.user_ratings_mean.reshape(-1, 1)
        
        # Áp dụng SVD (Singular Value Decomposition)
        # Số features k không được vượt quá số lượng user hoặc item nhỏ nhất trừ 1
        max_k = min(R.shape) - 1
        if max_k <= 0:
            # Fallback nếu dữ liệu quá nhỏ (vd <= 1 dòng/cột), svds sẽ báo lỗi k=0
            all_user_predicted_ratings = np.tile(self.user_ratings_mean.reshape(-1, 1), (1, R.shape[1]))
        else:
            actual_k = min(self.k, max_k)
            # Khởi tạo svds với R_demeaned dạng float
            U, sigma, Vt = svds(R_demeaned.astype(float), k=actual_k)
            
            # Đưa sigma về dạng ma trận đường chéo
            sigma = np.diag(sigma)
            
            # Tái tạo lại ma trận rating đầy đủ (predicted ratings)
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + self.user_ratings_mean.reshape(-1, 1)
        
        # Chuyển kết quả sang Pandas DataFrame để dễ dàng map với userId và movieId
        self.preds_matrix = pd.DataFrame(
            all_user_predicted_ratings, 
            columns=self.movies_list, 
            index=self.users_list
        )
        
    def predict(self, user_id, movie_ids=None):
        """
        Lấy điểm dự đoán (predicted rating) của user_id cho các movie_ids.
        - user_id: ID của người dùng.
        - movie_ids: Danh sách ID các phim cần dự đoán. Nếu None, lấy tất cả phim.
        Trả về: Pandas Series chứa điểm dự đoán, index là movieId.
        """
        # Xử lý trường hợp user lạ (Cold Start)
        if user_id not in self.users_list:
            if movie_ids is not None:
                return pd.Series(0.0, index=movie_ids)
            return pd.Series(0.0, index=self.movies_list)
            
        user_preds = self.preds_matrix.loc[user_id]
        
        if movie_ids is not None:
            # Reindex để tự động lấy điểm của các movieId có trong hệ thống,
            # và tự động điền 0.0 cho các movieId không có mặt (cold start items).
            # Cách này an toàn hơn so với việc gán tay và tránh lỗi index duplicate.
            return user_preds.reindex(movie_ids, fill_value=0.0)
            
        return user_preds
