import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict


class GenreBalancer:
    """Cân bằng phân phối genre trong danh sách recommendation cuối.

    Input là danh sách ``(movie_id, score)`` đã sort theo score giảm dần và
    file ``movies.csv`` có cột ``genres``. Thuật toán cố giữ phân phối genre
    trong top-N gần với catalog hơn nhưng vẫn ưu tiên relevance.
    """

    def __init__(self, movies_path: str = "data/raw/movies.csv", top_n: int = 10):
        self.movies_path = movies_path
        self.top_n = top_n
        self.movie_genres: Dict[int, List[str]] = {}
        self._load_movies()
        # Tần suất genre toàn catalog, dùng làm phân phối tham chiếu.
        all_genres = [g for gs in self.movie_genres.values() for g in gs]
        self.global_counts = Counter(all_genres)
        self.total_movies = len(self.movie_genres)

    def _load_movies(self) -> None:
        df = pd.read_csv(self.movies_path, usecols=["movieId", "genres"], dtype={"movieId": "int32", "genres": "object"})
        for _, row in df.iterrows():
            gid = int(row["movieId"])
            # Genre được ngăn cách bằng dấu | và có thể chứa "(no genres listed)".
            raw = row["genres"] or ""
            genres = [g.strip() for g in raw.split("|") if g and g != "(no genres listed)"]
            self.movie_genres[gid] = genres if genres else ["Unknown"]

    def _genre_score(self, movie_id: int) -> float:
        """Tính điểm ưu tiên cho các genre đang ít xuất hiện trong danh sách đã chọn."""
        selected = self._selected_counts
        # Cộng điểm inverse frequency cho từng genre của movie.
        score = 0.0
        for g in self.movie_genres.get(movie_id, ["Unknown"]):
            freq = selected.get(g, 0) / max(1, self._selected_total)
            # Genre càng ít xuất hiện thì điểm càng cao.
            score += 1.0 - freq
        return score / max(1, len(self.movie_genres.get(movie_id, [])))

    def balance(self, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Trả danh sách tối đa ``top_n`` candidate sau khi cân bằng genre.

        Input phải được sort theo relevance score giảm dần. Thuật toán greedy
        chọn candidate tối đa hóa tổng có trọng số giữa relevance và genre score.
        """
        self._selected_counts = Counter()
        self._selected_total = 0
        selected: List[Tuple[int, float]] = []
        remaining = candidates.copy()
        lambda_balance = 0.3  # Mức đánh đổi giữa relevance và cân bằng genre.
        while remaining and len(selected) < self.top_n:
            best_idx = None
            best_score = -float("inf")
            for i, (mid, rel_score) in enumerate(remaining):
                bal_score = self._genre_score(mid)
                combined = (1 - lambda_balance) * rel_score + lambda_balance * bal_score
                if combined > best_score:
                    best_score = combined
                    best_idx = i
            if best_idx is None:
                break
            chosen_mid, chosen_rel = remaining.pop(best_idx)
            selected.append((chosen_mid, chosen_rel))
            # Cập nhật bộ đếm genre đã chọn.
            for g in self.movie_genres.get(chosen_mid, []):
                self._selected_counts[g] += 1
                self._selected_total += 1
        return selected
