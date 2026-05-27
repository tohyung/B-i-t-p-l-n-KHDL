import numpy as np

class LightweightReranker:
    """
    Rerank nhẹ để tăng đa dạng và lọc các movie quá trùng semantic.
    """
    def __init__(self, genome_processor=None):
        self.genome_processor = genome_processor

    def rerank(self, candidate_movie_indices, candidate_scores, top_n=10):
        """
        Rerank candidate bằng cách bỏ qua movie có semantic vector quá giống
        các movie đã chọn nếu có genome_processor. Nếu không có genome_processor,
        chỉ sort theo candidate_scores.
        """
        if len(candidate_movie_indices) == 0 or top_n <= 0:
            return []

        # Nếu không có genome_processor thì chỉ sort theo score
        sorted_order = np.argsort(candidate_scores)[::-1]
        
        if self.genome_processor is None or self.genome_processor.movie_genome_matrix is None:
            return np.array(candidate_movie_indices)[sorted_order][:top_n]

        # Logic lọc trùng semantic để tăng diversity
        selected_indices = []
        
        # Bắt đầu bằng item có score cao nhất
        candidates = np.array(candidate_movie_indices)[sorted_order]
        scores = np.array(candidate_scores)[sorted_order]
        
        selected_indices.append(candidates[0])
        
        # Chọn tuần tự item có score cao nhưng không quá giống item đã chọn
        for i in range(1, len(candidates)):
            if len(selected_indices) >= top_n:
                break
                
            candidate = candidates[i]
            # Tính similarity lớn nhất với các item đã chọn
            sims = self.genome_processor.movie_genome_matrix[candidate] @ self.genome_processor.movie_genome_matrix[selected_indices].T
            max_sim = np.max(sims)
            
            # Nếu gần như trùng semantic (>0.95) thì bỏ qua
            if max_sim > 0.95:
                continue
                
            selected_indices.append(candidate)
            
        # Bù thêm nếu lọc quá nhiều item
        if len(selected_indices) < top_n:
            remaining = [c for c in candidates if c not in selected_indices]
            selected_indices.extend(remaining[:top_n - len(selected_indices)])
            
        return selected_indices
