Bước 1: Chuẩn bị môi trường
Cài Python libraries cần thiết:
pip install pandas mysql-connector-python
Tổ chức thư mục như sau:
project/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── tags.csv
│   └── links.csv
├── schema.sql
├── data_loader.py
├── import_data.py
└── eda.py

Bước 2: Tạo database trong DBeaver

Mở DBeaver → click New Database Connection → chọn MySQL
Điền thông tin: Host localhost, Port 3306, User/Password của bạn → Finish
Mở tab SQL Editor (Ctrl+])
Mở file schema.sql, copy toàn bộ nội dung vào SQL Editor
Bấm Execute All (Ctrl+Shift+Enter)
Sau khi chạy xong, F5 để refresh → bạn sẽ thấy database movielens với 8 bảng
Nếu trước đó đã tạo database bằng bản schema cũ, nên drop database movielens rồi chạy lại schema.sql để các unique key mới được tạo đúng.


Bước 3: Cấu hình kết nối MySQL trong import_data.py
Mở file import_data.py, sửa phần DB_CONFIG:
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",          # ← tên user MySQL của bạn
    "password": "your_password", # ← mật khẩu MySQL của bạn
    "database": "movielens",
    "charset":  "utf8mb4",
}
Và sửa DATA_DIR trỏ đúng đến thư mục chứa 4 file CSV:
DATA_DIR = "data"   # nếu chạy từ thư mục project/

Bước 4: Import dữ liệu vào MySQL
Mở terminal, cd vào thư mục project/, rồi chạy:
python import_data.py
Output terminal sẽ trông như sau:
==================================================
 IMPORT MOVIELENS → MYSQL
==================================================

Đọc dữ liệu từ CSV ...
  Đang đọc movies.csv  ... OK (9,742 rows)
  Đang đọc ratings.csv ... OK (100,836 rows)
  Đang đọc tags.csv    ... OK (3,683 rows)
  Đang đọc links.csv   ... OK (9,742 rows)

Kết nối MySQL ...
  Đã kết nối: localhost:3306/movielens

[1/7] Import bảng Movies ...      → 9,742 phim đã insert.
[2/7] Import bảng Genres ...      → 19 genres, 22,050 movie-genre links.
[3/7] Import bảng Users ...       → 610 users đã insert.
[4/7] Import bảng Ratings ...     → 100,836 ratings đã insert.
[5/7] Import bảng Tags ...        → 3,683 tags đã insert.
[6/7] Import bảng MovieLinks ...  → 9,742 links đã insert.

[7/7] Kiểm tra số bản ghi ...
      Users          :      610 rows
      Movies         :    9,742 rows
      ...
[OK] Import hoàn tất!
Sau đó vào DBeaver nhấn F5, click vào từng bảng → Data để xem dữ liệu đã vào chưa.

Bước 5: Chạy EDA
python eda.py
Sẽ in ra terminal 7 phân tích liên tiếp: dataset summary, phân bố rating, top phim, top genres, hoạt động user, rating theo năm, top tags.

Bước 6: Cung cấp dữ liệu cho Người 2 và Người 3
Người 2 và Người 3 chỉ cần thêm 2 dòng này vào đầu file của họ:
from data_loader import load_all

movies_df, ratings_df, tags_df, links_df = load_all("data")
Người 3 muốn gọi EDA hoặc UC03/UC04 trong terminal menu thì thêm:
from eda import run_eda, movie_detail_query, user_rating_history_query

# Gọi toàn bộ EDA
run_eda()

# Xem chi tiết phim (UC03)
movie_detail_query("Toy Story", movies_df, ratings_df, tags_df, links_df)

# Xem lịch sử rating user (UC04)
user_rating_history_query(1, ratings_df, movies_df)
