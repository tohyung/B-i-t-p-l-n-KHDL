class Explainer:
    """
    Lớp giải thích recommendation dựa trên template/rule.
    """

    @staticmethod
    def _value(features_dict, key, default=0.0):
        return float(features_dict.get(key, default) or default)

    @staticmethod
    def explain(features_dict):
        """
        Sinh chuỗi giải thích dựa trên giá trị feature.
        """
        reasons = []

        svd_score = Explainer._value(features_dict, 'svd_score')
        genome_similarity = Explainer._value(features_dict, 'genome_similarity')
        genre_overlap = Explainer._value(features_dict, 'genre_overlap')
        recent_trend_score = Explainer._value(features_dict, 'recent_trend_score')
        avg_movie_rating = Explainer._value(features_dict, 'avg_movie_rating')

        if svd_score >= 4.0:
            reasons.append("Những user có gu tương tự đánh giá phim này cao.")

        if genome_similarity >= 0.75:
            reasons.append("Phim có độ tương đồng semantic cao với các phim bạn thích.")

        if genre_overlap >= 2:
            reasons.append("Phim trùng nhiều thể loại với sở thích của bạn.")

        if recent_trend_score >= 200:
            reasons.append("Phim đang có xu hướng được nhiều user rating gần đây.")

        if avg_movie_rating >= 4.0:
            reasons.append("Phim có điểm rating trung bình cao trong cộng đồng.")

        if not reasons:
            reasons.append("Phim được đề xuất bởi mô hình collaborative filtering.")

        return "Được gợi ý vì:\n  * " + "\n  * ".join(reasons)
