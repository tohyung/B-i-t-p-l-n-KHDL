import numpy as np
from scipy import sparse
from ..preprocessing.genome_preprocessing import GenomePreprocessor


class GenomeRetriever:
    """Retrieval semantic dựa trên genome vector.

    Luồng xử lý:
    1. ``fit`` load ``genome-scores.csv`` qua ``GenomePreprocessor`` và lưu
       CSR matrix đã normalize theo hàng.
    2. ``compute_similarity`` tính cosine similarity giữa một movie truy vấn
       và toàn bộ movie còn lại.
    3. ``get_similar_movies`` trả top-k movie giống nhất.
    4. ``get_semantic_candidates`` có thể dùng để sinh candidate semantic
       trước bước ranking.
    """

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.preprocessor = GenomePreprocessor(self.data_dir)
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.matrix: sparse.csr_matrix | None = None

    def fit(self) -> None:
        """Load và normalize genome matrix."""
        self.preprocessor.fit()
        self.matrix = self.preprocessor.get_matrix()
        (
            self.movie_to_idx,
            self.idx_to_movie,
            _,
            _,
        ) = self.preprocessor.get_mappings()

    def _ensure_fitted(self) -> None:
        if self.matrix is None:
            raise RuntimeError("GenomeRetriever must be fitted before use.")

    def compute_similarity(self, query_movie_id: int) -> np.ndarray:
        """Trả cosine similarity của ``query_movie_id`` với toàn bộ movie."""
        self._ensure_fitted()
        if query_movie_id not in self.movie_to_idx:
            raise KeyError(f"Movie ID {query_movie_id} not found in genome data.")
        q_idx = self.movie_to_idx[query_movie_id]
        # CSR row một dòng đại diện cho movie truy vấn.
        query_vec = self.matrix.getrow(q_idx)
        # Dot product với toàn bộ matrix vì các vector đã normalize.
        sims = query_vec @ self.matrix.T
        return sims.toarray().ravel()

    def get_similar_movies(self, query_movie_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Trả top-k movie giống nhất dưới dạng ``(movie_id, similarity)``."""
        sims = self.compute_similarity(query_movie_id)
        # Loại chính movie truy vấn khỏi kết quả.
        q_idx = self.movie_to_idx[query_movie_id]
        sims[q_idx] = -np.inf
        # Dùng argpartition để lấy top-k nhanh hơn full sort.
        if top_k >= len(sims):
            top_idx = np.argsort(sims)[::-1]
        else:
            part = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = part[np.argsort(sims[part])[::-1]]
        return [(self.idx_to_movie[int(idx)], float(sims[int(idx)])) for idx in top_idx]

    def get_semantic_candidates(self, query_movie_id: int, similarity_threshold: float = 0.5) -> list[int]:
        """Trả các movie ID có similarity vượt ngưỡng cho trước."""
        sims = self.compute_similarity(query_movie_id)
        q_idx = self.movie_to_idx[query_movie_id]
        sims[q_idx] = -np.inf
        candidate_idxs = np.where(sims >= similarity_threshold)[0]
        return [self.idx_to_movie[int(idx)] for idx in candidate_idxs]
