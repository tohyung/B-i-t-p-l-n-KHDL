# MovieLens Recommendation Pipeline

Repo này xây dựng một pipeline gợi ý phim trên dữ liệu MovieLens. Chương trình chính chạy bằng CLI Python, gồm các bước: xử lý dữ liệu, sinh candidate bằng SVD, tạo đặc trưng semantic bằng genome, train XGBoost learning-to-rank, rerank để tăng đa dạng, rồi in danh sách phim gợi ý kèm giải thích.

## Cấu Trúc Dự Án

```text
.
+-- main.py                         # CLI chính: train và recommend
+-- import_data.py                  # Import CSV vào MySQL, không bắt buộc
+-- data_loader.py                  # Loader phục vụ import_data.py
+-- schema.sql                      # Schema MySQL, không bắt buộc
+-- data/raw/                       # Dữ liệu MovieLens CSV
+-- artifacts/                      # Model và metadata sau khi train
+-- src/
    +-- data/                       # Load CSV, preprocess, mapping ID, sparse matrix
    +-- retrieval/                  # SVD retrieval và genome retrieval utility
    +-- ranking/                    # Genome processor, feature engineering, XGBoost ranker
    +-- reranking/                  # Rerank theo đa dạng semantic/genre
    +-- explainability/             # Giải thích recommendation bằng rule
    +-- evaluation/                 # RMSE, MAE, Precision@K, Recall@K, NDCG@K
```

## Dữ Liệu Đầu Vào

Pipeline cần các file MovieLens trong thư mục `data/raw/`:

```text
data/raw/
+-- ratings.csv
+-- movies.csv
+-- tags.csv
+-- links.csv
+-- genome-scores.csv
+-- genome-tags.csv
```

Pipeline train/recommend đọc trực tiếp các file CSV và artifacts. MySQL là phần phụ, không bắt buộc để chạy gợi ý phim.

## Cách Chạy Chương Trình

Mở terminal IDE tại thư mục repo:

```powershell
cd D:\code\project-khdl\project-khdl
```

Nếu dùng môi trường `uv` hiện tại:

```powershell
$env:UV_CACHE_DIR="D:\code\project-khdl\project-khdl\.uv-cache"
```

### Train Toàn Bộ Pipeline

```powershell
uv run python main.py train
```

Sau khi train xong, chương trình sẽ ghi các artifact vào thư mục `artifacts/`.

### Chạy Recommendation

```powershell
uv run python main.py recommend
```

Chương trình sẽ yêu cầu nhập `userId`, ví dụ:

```text
1
```

Sau đó terminal sẽ in top phim được gợi ý cho user đó.

### Chạy Evaluation

```powershell
uv run python main.py evaluate
```

Command này đánh giá model trên phần test theo temporal split và in các metric:

- `RMSE`
- `MAE`
- `Precision@K`
- `Recall@K`
- `NDCG@K`

Có thể giảm thời gian chạy evaluation bằng sampling:

```powershell
$env:EVAL_SAMPLE_USERS="100"
$env:EVAL_RATING_ROWS="10000"
$env:EVAL_TOP_K="10"
uv run python main.py evaluate
```

## Luồng Train

File chính: `main.py`

Khi chạy:

```powershell
uv run python main.py train
```

Pipeline thực hiện các bước sau:

1. **Load dữ liệu**
   - Đọc `ratings.csv`
   - Đọc `movies.csv`
   - Đọc `genome-scores.csv`

2. **Tiền xử lý**
   - Tách năm phát hành từ tên phim, ví dụ `Toy Story (1995)` -> `1995`.
   - Chia train/test theo thời gian bằng `timestamp`.
   - Rating cũ hơn vào train, rating mới hơn vào test.

3. **Mapping ID**
   - Chuyển `userId` và `movieId` gốc thành index liên tục.
   - Cần bước này để build sparse matrix hiệu quả.
   - Lưu vào `artifacts/id_mappings.joblib`.

4. **Build sparse user-movie matrix**
   - Tính rating trung bình của từng user.
   - Demean rating theo user.
   - Build `scipy.sparse.csr_matrix`.
   - Lưu user mean vào `artifacts/user_means.joblib`.

5. **Train SVD retrieval**
   - Train truncated SVD trên sparse matrix đã demean.
   - Lưu model vào `artifacts/svd_model.joblib`.
   - Model này dùng để lấy candidate movie cho user.

6. **Build genome matrix**
   - Dùng `genome-scores.csv` để tạo vector semantic cho từng movie.
   - Normalize vector để tính cosine similarity nhanh.
   - Lưu vào `artifacts/genome_matrix.joblib`.

7. **Feature engineering**
   - Tính thống kê user: rating trung bình, mức độ hoạt động.
   - Tính thống kê movie: rating trung bình, số rating, popularity rank.
   - Tính recent trend score.
   - Tính tuổi phim, genre, genre overlap.
   - Lưu metadata vào `artifacts/feature_metadata.joblib`.

8. **Train XGBoost ranker**
   - Sinh tập train dạng learning-to-rank bằng sampling.
   - Feature gồm: SVD score, genome similarity, user stats, movie stats, popularity, recency, genre overlap.
   - Train `xgboost.XGBRanker` với objective `rank:pairwise`.
   - Lưu model vào `artifacts/xgb_ranker.json`.

Do dữ liệu MovieLens lớn, XGBoost không train trên toàn bộ mọi cặp user-movie. Pipeline đang dùng sampling để tránh tràn RAM.

Có thể chỉnh sampling bằng biến môi trường:

```powershell
$env:XGB_SAMPLE_USERS="1000"
$env:XGB_MAX_POSITIVE_PER_USER="20"
$env:XGB_NEGATIVE_RATIO="1"
$env:XGB_N_ESTIMATORS="100"
$env:XGB_MAX_DEPTH="6"
$env:XGB_LEARNING_RATE="0.1"
uv run python main.py train
```

## Luồng Recommend

Khi chạy:

```powershell
uv run python main.py recommend
```

Pipeline thực hiện các bước sau:

1. **Load artifacts**
   - `id_mappings.joblib`
   - `user_means.joblib`
   - `svd_model.joblib`
   - `genome_matrix.joblib`
   - `feature_metadata.joblib`
   - `xgb_ranker.json`

2. **Lấy lịch sử user**
   - Load `ratings.csv`.
   - Lọc các phim user đã xem/rating.
   - Loại các phim đã xem khỏi danh sách gợi ý.

3. **Candidate retrieval bằng SVD**
   - SVD dự đoán điểm cho movie chưa xem.
   - Lấy top candidate movie.

4. **Feature extraction**
   - Build user semantic profile từ các phim user thích.
   - Tính genome similarity giữa user profile và candidate movie.
   - Tạo feature ranking cho từng candidate.

5. **Ranking bằng XGBoost**
   - Load `artifacts/xgb_ranker.json`.
   - Dự đoán ranking score cho candidate.
   - Nếu chưa có model XGBoost, chương trình fallback sang weighted score.

6. **Reranking**
   - Dùng genome vector để giảm các phim quá giống nhau về semantic.
   - Mục tiêu là danh sách cuối cùng vừa liên quan vừa đa dạng hơn.

7. **Explainability**
   - Sinh giải thích bằng rule dựa trên feature.
   - Ví dụ:
     - Similar users highly rated this.
     - Strong semantic similarity to your favorite movies.
     - Matches multiple of your preferred genres.
     - Critically acclaimed.

## Output Khi Chạy Recommend

Output sẽ có log các phase và danh sách top phim:

```text
FINAL TOP 10 RECOMMENDATIONS FOR USER 1

1. Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) (ID: 1198)
   Recommended because:
  * Similar users highly rated this.
  * Strong semantic similarity to your favorite movies.
```

Mỗi item gồm:

- Thứ hạng
- Tên phim
- `movieId`
- Lý do được gợi ý

## Artifacts Sau Khi Train

Sau khi train thành công, thư mục `artifacts/` sẽ có:

```text
artifacts/
+-- id_mappings.joblib
+-- user_means.joblib
+-- svd_model.joblib
+-- genome_matrix.joblib
+-- feature_metadata.joblib
+-- xgb_ranker.json
```

Các artifact này đủ để chạy recommendation, miễn là vẫn còn `data/raw/ratings.csv` và `data/raw/movies.csv`.

## Vai Trò Của MySQL

MySQL là phần phụ.

Pipeline chính trong `main.py train` và `main.py recommend` không dùng MySQL. Nó đọc CSV và artifacts trực tiếp.

MySQL chỉ cần nếu muốn:

- Import dữ liệu MovieLens vào database.
- Dùng DBeaver/MySQL để xem dữ liệu.
- Phục vụ báo cáo hoặc tích hợp app sau này.

### Tạo Database

Mở `schema.sql` bằng DBeaver hoặc MySQL rồi execute. File này tạo các bảng:

- `Users`
- `Movies`
- `Genres`
- `MovieGenres`
- `Ratings`
- `Tags`
- `MovieLinks`
- `Recommendations`

### Import Dữ Liệu Vào MySQL

Sửa `DB_CONFIG` trong `import_data.py` cho đúng máy:

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "movielens",
    "charset": "utf8mb4",
}
```

Sau đó chạy:

```powershell
$env:UV_CACHE_DIR="D:\code\project-khdl\project-khdl\.uv-cache"
uv run python import_data.py
```

## Genome Đang Được Dùng Như Thế Nào

Genome data đã được nối vào pipeline chính.

Nó được dùng để:

- Tạo vector semantic cho movie.
- Tạo user semantic profile từ các phim user thích.
- Tính feature `genome_similarity`.
- Giúp XGBoost rank candidate chính xác hơn.
- Rerank để giảm phim quá giống nhau.
- Sinh explanation khi semantic similarity cao.

Lưu ý: candidate retrieval chính hiện vẫn là SVD. Genome được dùng ở bước feature extraction, ranking, reranking và explainability. Module `GenomeRetriever` tồn tại trong `src/retrieval/genome_retrieval.py`, nhưng chưa phải nguồn candidate chính của `main.py`.

## EDA

EDA hiện chưa được nối vào `main.py`.

Output hiện tại của chương trình gồm:

- Log train pipeline.
- Log recommend pipeline.
- Log evaluation pipeline.
- RMSE, MAE, Precision@K, Recall@K, NDCG@K.
- Danh sách recommendation.
- Explanation cho từng phim.

`import_data.py` chỉ import dữ liệu vào MySQL và in số lượng record trong từng bảng. Nó chưa sinh biểu đồ hoặc báo cáo EDA.

Các output EDA có thể bổ sung sau:

- Phân phối rating.
- Số lượng user, movie, rating, tag.
- Top genre.
- Phim có nhiều rating nhất.
- Phim có rating trung bình cao nhất.
- Rating theo thời gian.
- Độ sparse của user-movie matrix.
- Độ phủ genome.

## Dependencies

Môi trường hiện đã dùng được với:

- Python 3.14.3 qua `uv`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `joblib`
- `xgboost==3.2.0`
- `mysql-connector-python==9.7.0`

Nếu thiếu package, cài bằng:

```powershell
uv pip install numpy pandas scipy scikit-learn joblib xgboost mysql-connector-python
```

## Giới Hạn Hiện Tại

- XGBoost train bằng sampling để tránh OOM, chưa train exhaustive trên toàn bộ mọi cặp user-movie.
- Chưa hỗ trợ cold-start user trong CLI chính.
- `main.py recommend` chưa ghi recommendation vào MySQL.
- EDA chưa phải một command riêng.
- Negative sample của XGBoost hiện là phim chưa rating, được gán relevance `0`.
