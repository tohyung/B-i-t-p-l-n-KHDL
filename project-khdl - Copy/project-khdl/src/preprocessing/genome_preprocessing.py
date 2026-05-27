import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path


class GenomePreprocessor:
    """Load và tiền xử lý genome data thành sparse matrix tiết kiệm bộ nhớ.

    File ``genome-scores.csv`` chứa relevance score của tag cho từng movie.
    Class này đọc file theo chunk, build ``scipy.sparse.csr_matrix`` và trả
    mapping để tầng retrieval có thể tái sử dụng.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.scores_path = self.data_dir / "genome-scores.csv"
        self.tags_path = self.data_dir / "genome-tags.csv"
        self.movies_path = self.data_dir / "movies.csv"
        # Các mapping này được điền sau khi gọi ``fit``.
        self.movie_id_to_idx: dict[int, int] = {}
        self.idx_to_movie_id: dict[int, int] = {}
        self.tag_id_to_idx: dict[int, int] = {}
        self.idx_to_tag_id: dict[int, int] = {}
        self.matrix: sparse.csr_matrix | None = None

    def _load_tag_mapping(self) -> None:
        """Đọc ``genome-tags.csv`` và tạo mapping tag ID <-> index."""
        tags_df = pd.read_csv(self.tags_path, usecols=["tagId"], dtype={"tagId": "int32"})
        tag_ids = tags_df["tagId"].unique()
        self.tag_id_to_idx = {tid: i for i, tid in enumerate(tag_ids)}
        self.idx_to_tag_id = {i: tid for tid, i in self.tag_id_to_idx.items()}

    def _load_movie_mapping(self) -> None:
        """Đọc ``movies.csv`` và tạo mapping movie ID <-> index."""
        movies_df = pd.read_csv(self.movies_path, usecols=["movieId"], dtype={"movieId": "int32"})
        movie_ids = movies_df["movieId"].unique()
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}

    def fit(self, chunk_size: int = 500_000) -> None:
        """Build sparse genome matrix.

        Args:
            chunk_size: Số dòng đọc mỗi lần từ ``genome-scores.csv`` để giảm
                peak memory.
        """
        self._load_tag_mapping()
        self._load_movie_mapping()

        n_movies = len(self.movie_id_to_idx)
        n_tags = len(self.tag_id_to_idx)
        # Chuẩn bị dữ liệu theo COO format rồi chuyển sang CSR.
        data = []
        rows = []
        cols = []

        # ``genome-scores.csv`` gồm các cột: movieId, tagId, relevance.
        for chunk in pd.read_csv(
            self.scores_path,
            usecols=["movieId", "tagId", "relevance"],
            dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
            chunksize=chunk_size,
        ):
            # Map ID gốc sang index matrix và bỏ các dòng không map được.
            row_idx = chunk["movieId"].map(self.movie_id_to_idx)
            col_idx = chunk["tagId"].map(self.tag_id_to_idx)
            valid = row_idx.notna() & col_idx.notna()
            rows.extend(row_idx[valid].astype(np.int32).values)
            cols.extend(col_idx[valid].astype(np.int32).values)
            data.extend(chunk.loc[valid, "relevance"].values)

        # Build COO matrix rồi chuyển sang CSR để slice theo hàng hiệu quả.
        coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_movies, n_tags), dtype=np.float32)
        self.matrix = coo.tocsr()
        # Normalize từng hàng về L2 norm = 1 để dot product là cosine similarity.
        eps = 1e-9
        row_norms = np.sqrt(self.matrix.multiply(self.matrix).sum(axis=1)).A1 + eps
        inv_norms = 1.0 / row_norms
        # Scale theo hàng bằng broadcasting.
        self.matrix = self.matrix.multiply(inv_norms[:, None])

    def get_matrix(self) -> sparse.csr_matrix:
        if self.matrix is None:
            raise RuntimeError("GenomePreprocessor.fit() must be called before accessing the matrix.")
        return self.matrix

    def get_mappings(self) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
        """Trả các mapping movie ID <-> index và tag ID <-> index."""
        return (
            self.movie_id_to_idx,
            self.idx_to_movie_id,
            self.tag_id_to_idx,
            self.idx_to_tag_id,
        )
