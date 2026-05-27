import numpy as np
from scipy import sparse


class DiversityRebalancer:
    """Rerank candidate list để tăng đa dạng nội bộ.

    Class này cần một CSR matrix đã normalize theo hàng, trong đó mỗi hàng là
    một movie và các cột là genome tag. Vì các hàng đã L2-normalize, dot product
    chính là cosine similarity.
    """

    def __init__(self, matrix: sparse.csr_matrix, movie_to_idx: dict[int, int]):
        self.matrix = matrix
        self.movie_to_idx = movie_to_idx

    def _avg_similarity(self, idx: int, selected_idxs: list[int]) -> float:
        """Tính similarity trung bình giữa item hiện tại và các item đã chọn."""
        if not selected_idxs:
            return 0.0
        vec = self.matrix.getrow(idx)
        sel_mat = self.matrix[selected_idxs]
        # Dot product với các hàng đã chọn vì tất cả đã normalize.
        sims = vec @ sel_mat.T
        return float(sims.mean())

    def diversify(self, candidates: list[int], top_k: int = 10, penalty: float = 0.2) -> list[int]:
        """Trả danh sách tối đa top_k movie ID sau khi phạt độ giống nhau.

        ``penalty`` điều chỉnh mức độ similarity làm giảm score candidate.
        Thuật toán greedy: chọn item tốt nhất, rồi tiếp tục chọn item còn lại
        sau khi trừ ``penalty * avg_similarity`` với các item đã chọn.
        """
        if not candidates:
            return []
        # Giả định candidates đã được sort theo relevance giảm dần.
        selected: list[int] = []
        remaining = candidates.copy()
        # Ở đây dùng thứ tự ban đầu làm proxy cho relevance score.
        while remaining and len(selected) < top_k:
            best_id = None
            best_score = -np.inf
            for mid in remaining:
                idx = self.movie_to_idx[mid]
                # Item đứng càng sớm thì base_score càng cao.
                base_score = -remaining.index(mid)
                penalty_score = penalty * self._avg_similarity(idx, [self.movie_to_idx[m] for m in selected])
                adjusted = base_score - penalty_score
                if adjusted > best_score:
                    best_score = adjusted
                    best_id = mid
            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)
        return selected


# Ví dụ sử dụng tham khảo:
# from src.retrieval.genome_retrieval import GenomeRetriever
# retr = GenomeRetriever('data/raw')
# retr.fit()
# candidates = [1, 2, 3, 4, 5]
# rebal = DiversityRebalancer(retr.matrix, retr.movie_to_idx)
# diverse = rebal.diversify(candidates, top_k=5)
# print(diverse)
