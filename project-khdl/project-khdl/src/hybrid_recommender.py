import pandas as pd

class HybridRecommender:
    """
    Hệ thống Hybrid kết hợp giữa Collaborative Filtering (SVD) và Content-based.
    Trọng số cấu hình: 0.7 * SVD_score + 0.3 * Content_score
    """
    def __init__(self, content_recommender, svd_recommender):
        self.content_recommender = content_recommender
        self.svd_recommender = svd_recommender

    def normalize(self, series):
        """
        Chuẩn hóa Min-Max (Min-Max Scaling) để đưa mảng điểm về thang đo [0.0, 1.0].
        Giúp 2 hệ điểm có thể được cộng lại với nhau hợp lý.
        """
        if len(series) == 0:
            return series
            
        s_min = series.min()
        s_max = series.max()
        
        if s_max == s_min:
            # Nếu tất cả điểm bằng nhau, coi như chuẩn hóa tất cả về 1.0
            return pd.Series(1.0, index=series.index)
            
        return (series - s_min) / (s_max - s_min)

    def recommend(self, user_id, movie_title, top_n=10, candidate_n=100):
        """
        Gợi ý phim (Hybrid) cho một user_id cụ thể dựa trên một phim yêu thích.
        Quy trình:
        1. Tìm top 100 phim giống nhất với movie_title bằng Content-based.
        2. Dùng mô hình SVD dự đoán rating của user_id cho 100 phim đó.
        3. Chuẩn hóa 2 hệ điểm về [0, 1].
        4. Tính Final Score = 0.7*SVD + 0.3*Content.
        5. Trả về top N phim cao điểm nhất.
        """
        # Bước 1: Lấy danh sách candidate từ Content-based model
        # Lấy số lượng ứng viên (candidate_n) lớn hơn số lượng top_n cần lấy
        content_response = self.content_recommender.recommend(movie_title, top_n=candidate_n)
        
        if content_response["status"] != "success":
            return content_response  # Trả về luôn lỗi hoặc danh sách gợi ý viết sai chính tả
            
        content_result = content_response["recommendations"].copy()
        input_movie = content_response["input_movie"]
        
        # Đổi tên cột similarity thành content_score cho dễ hiểu
        content_result = content_result.rename(columns={"similarity": "content_score"})
        
        # Bước 2: Lấy điểm dự đoán CF (SVD) cho danh sách candidate
        candidate_movie_ids = content_result["movieId"].tolist()
        svd_scores = self.svd_recommender.predict(user_id, candidate_movie_ids)
        
        # Chuyển Series SVD scores thành DataFrame để merge
        svd_df = pd.DataFrame({
            "movieId": svd_scores.index,
            "svd_score": svd_scores.values
        })
        
        # Ghép 2 bảng điểm lại theo movieId
        hybrid_df = content_result.merge(svd_df, on="movieId", how="inner")
        
        # Lọc ra các bộ phim user đã từng rate (thường recommend thì ko recommend lại phim đã xem)
        # Tuy nhiên ở hệ thống đơn giản này, ta có thể bỏ qua hoặc để nguyên. 
        # (Ở đây ta cứ để nguyên chấm điểm, ai thích xem lại cũng được)
        
        # Bước 3: Chuẩn hóa điểm về hệ quy chiếu [0, 1]
        hybrid_df["content_score_norm"] = self.normalize(hybrid_df["content_score"])
        hybrid_df["svd_score_norm"] = self.normalize(hybrid_df["svd_score"])
        
        # Bước 4: Tính Final Score theo công thức yêu cầu
        hybrid_df["final_score"] = (
            0.7 * hybrid_df["svd_score_norm"] + 
            0.3 * hybrid_df["content_score_norm"]
        )
        
        # Bước 5: Sắp xếp theo Final Score giảm dần và cắt lấy Top N
        hybrid_df = hybrid_df.sort_values(by="final_score", ascending=False).head(top_n)
        
        # Trích xuất các cột quan trọng
        final_result = hybrid_df[[
            "movieId", "title", "genres", 
            "svd_score", "content_score", "final_score"
        ]].copy()
        
        return {
            "status": "success",
            "input_movie": input_movie,
            "user_id": user_id,
            "recommendations": final_result
        }
