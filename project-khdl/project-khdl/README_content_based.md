1. Vai trò của module

Module này là phần Content-based Recommendation trong hệ thống khuyến nghị phim sử dụng bộ dữ liệu MovieLens.

Phần này do người 2 phụ trách.

Mục tiêu của module:

Người dùng nhập tên một bộ phim
→ Hệ thống phân tích genres và tags của phim đó
→ Hệ thống tìm các phim có nội dung tương tự
→ Trả về danh sách Top N phim gợi ý

Module này sẽ được người 3 sử dụng để ghép với phần Collaborative Filtering và có thể dùng tiếp cho Hybrid Recommendation.
2. Dữ liệu đầu vào

Module này sử dụng 2 file chính:

data/movies.csv
data/tags.csv

File ratings.csv không được dùng trong module Content-based. File đó sẽ được người 3 dùng cho Collaborative Filtering.

2.1. movies.csv

File này chứa thông tin phim.

Các cột cần dùng:

movieId,title,genres

Ví dụ:

1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy

Ý nghĩa:

Cột	Ý nghĩa
movieId	ID của phim
title	Tên phim
genres	Danh sách thể loại phim, ngăn cách bằng dấu `	`
2.2. tags.csv

File này chứa tag do user gắn cho phim.

Các cột cần dùng:

userId,movieId,tag,timestamp

Ý nghĩa:

Cột	Ý nghĩa
userId	ID người dùng
movieId	ID phim
tag	Tag mô tả phim
timestamp	Thời điểm user gắn tag

Ví dụ tag:

pixar
fun
superhero
sci-fi
dark
time travel
3. Cấu trúc thư mục đề xuất

Người 3 khi tải folder của người 2 về nên giữ nguyên cấu trúc như sau:

movie-recommendation-system/
│
├── data/
│   ├── movies.csv
│   ├── tags.csv
│   └── ratings.csv
│
├── src/
│   └── content_based.py
│
├── main.py
└── README.md

Trong đó:

Thành phần	Người phụ trách	Vai trò
data/movies.csv	Người 1	Dữ liệu phim
data/tags.csv	Người 1	Dữ liệu tag
data/ratings.csv	Người 1 / Người 3	Dữ liệu rating cho Collaborative Filtering
src/content_based.py	Người 2	Code Content-based Recommendation
main.py	Người 2 / Người 3	File chạy demo terminal
README.md	Người 2	Hướng dẫn sử dụng module


4. Cài đặt thư viện

Trước khi chạy code, cần cài các thư viện sau:

pip install pandas scikit-learn

Nếu lệnh trên lỗi, dùng:

python -m pip install pandas scikit-learn

Các thư viện được sử dụng:

Thư viện	Mục đích
pandas	Đọc và xử lý dữ liệu CSV
scikit-learn	Vector hóa TF-IDF và tính cosine similarity
re	Làm sạch text trong tag

5. Nguyên lý hoạt động

Module này hoạt động theo hướng Content-based Filtering.

Ý tưởng chính:

Nếu hai bộ phim có nội dung giống nhau
→ hệ thống xem chúng là tương tự nhau
→ có thể gợi ý phim này cho người thích phim kia

Trong project này, nội dung phim được xây dựng từ:

genres + tags

Ví dụ:

Toy Story (1995)


Genres:
Adventure | Animation | Children | Comedy | Fantasy


Tags:
pixar, fun, animation, children

Sau khi xử lý, nội dung phim có thể được biểu diễn thành chuỗi:

adventure animation children comedy fantasy
adventure animation children comedy fantasy
adventure animation children comedy fantasy
pixar fun animation children

Trong code hiện tại, genres được lặp lại 3 lần để genres có trọng số mạnh hơn tags.

Lý do:

- genres là thông tin chính thức, ổn định hơn
- tags do user nhập nên có thể bị nhiễu
- lặp genres giúp kết quả gợi ý không bị tags chi phối quá nhiều
6. Quy trình xử lý trong code

File chính của module là:

src/content_based.py

Trong file này có class:

ContentBasedRecommender

Class này thực hiện các bước sau:

Bước 1: Đọc movies.csv và tags.csv
Bước 2: Làm sạch genres
Bước 3: Làm sạch tags
Bước 4: Gom tất cả tag của cùng một movieId
Bước 5: Ghép tags vào bảng movies theo movieId
Bước 6: Tạo cột content = genres + genres + genres + tags
Bước 7: Vector hóa content bằng TF-IDF
Bước 8: Khi user nhập tên phim, tìm phim tương ứng
Bước 9: Tính cosine similarity giữa phim đầu vào và tất cả phim còn lại
Bước 10: Sắp xếp similarity giảm dần
Bước 11: Trả về Top N phim tương tự
7. Các hàm chính
7.1. Hàm khởi tạo
recommender = ContentBasedRecommender(
    movies_path="data/movies.csv",
    tags_path="data/tags.csv"
)

Hàm này sẽ tự động:

- đọc dữ liệu
- tiền xử lý dữ liệu
- xây dựng TF-IDF matrix
7.2. Hàm recommend
response = recommender.recommend(movie_title, top_n)

Input:

Tham số	Kiểu dữ liệu	Ý nghĩa
movie_title	string	Tên phim hoặc một phần tên phim
top_n	int	Số lượng phim muốn gợi ý

Ví dụ:

response = recommender.recommend("Toy Story", 10)

Output của hàm là một dictionary.

Nếu tìm thấy phim:

{
    "status": "success",
    "input_movie": {
        "movieId": 1,
        "title": "Toy Story (1995)",
        "genres": "Adventure|Animation|Children|Comedy|Fantasy"
    },
    "recommendations": DataFrame
}

Trong đó recommendations là DataFrame có các cột:

movieId
title
genres
similarity

Nếu không tìm thấy phim:

{
    "status": "not_found",
    "message": "Movie not found.",
    "suggestions": DataFrame
}

Trong đó suggestions là danh sách phim có tên gần giống với input.

8. Cách chạy demo trên terminal

Tại thư mục gốc của project, chạy:

python main.py

Menu terminal:

============================================================
MOVIELENS CONTENT-BASED RECOMMENDATION SYSTEM
============================================================
1. Recommend similar movies
0. Exit


Select option:

Ví dụ nhập:

Select option: 1
Enter movie title: Toy Story
Enter number of recommendations: 10

Output mẫu:

==========================================================================================
CONTENT-BASED RECOMMENDATION
Based on: Genres + Tags
==========================================================================================


Input Movie:
Movie ID : 1
Title    : Toy Story (1995)
Genres   : Adventure|Animation|Children|Comedy|Fantasy


Recommended Movies:


Rank   Movie ID   Title                                         Similarity
------------------------------------------------------------------------------------------
1      3114       Toy Story 2 (1999)                            0.9234
2      4886       Monsters, Inc. (2001)                         0.8650
3      78499      Toy Story 3 (2010)                            0.8412
...

Lưu ý: Điểm similarity có thể khác nhau tùy dữ liệu tags và cách xử lý text.

9. Cách người 3 sử dụng module này

Người 3 cần import class ContentBasedRecommender từ file của người 2.

Ví dụ trong file của người 3, chẳng hạn:

src/hybrid_recommender.py

hoặc trong:

main.py

sử dụng như sau:

from src.content_based import ContentBasedRecommender


content_model = ContentBasedRecommender(
    movies_path="data/movies.csv",
    tags_path="data/tags.csv"
)


response = content_model.recommend("Toy Story", 10)

Sau đó kiểm tra trạng thái:

if response["status"] == "success":
    content_result = response["recommendations"]
    print(content_result)
else:
    print("Movie not found.")
10. Output người 3 cần lấy

Người 3 chủ yếu cần DataFrame này:

content_result = response["recommendations"]

DataFrame này có dạng:

movieId | title | genres | similarity

Trong đó:

Cột	Ý nghĩa
movieId	ID phim được gợi ý
title	Tên phim được gợi ý
genres	Thể loại phim
similarity	Điểm tương đồng theo content-based

Người 3 có thể dùng cột movieId để ghép với kết quả Collaborative Filtering.

11. Cách ghép với Collaborative Filtering

Phần Collaborative Filtering của người 3 thường sẽ trả về kết quả dạng:

movieId | title | cf_score

Phần Content-based của người 2 trả về:

movieId | title | similarity

Người 3 có thể merge hai kết quả theo movieId.

Ví dụ:

hybrid_df = cf_result.merge(
    content_result[["movieId", "similarity"]],
    on="movieId",
    how="outer"
)

Sau đó đổi tên cột:

hybrid_df = hybrid_df.rename(columns={
    "similarity": "content_score"
})

Nếu có giá trị bị thiếu:

hybrid_df["cf_score"] = hybrid_df["cf_score"].fillna(0)
hybrid_df["content_score"] = hybrid_df["content_score"].fillna(0)
12. Công thức Hybrid Recommendation đề xuất

Vì cf_score và content_score có thể khác thang đo, người 3 nên chuẩn hóa trước khi cộng điểm.

Ví dụ:

def normalize(series):
    if series.max() == series.min():
        return series
    return (series - series.min()) / (series.max() - series.min())

Sau đó:

hybrid_df["cf_score_norm"] = normalize(hybrid_df["cf_score"])
hybrid_df["content_score_norm"] = normalize(hybrid_df["content_score"])

Công thức hybrid:

hybrid_df["hybrid_score"] = (
    0.6 * hybrid_df["cf_score_norm"] +
    0.4 * hybrid_df["content_score_norm"]
)

Sắp xếp kết quả:

hybrid_df = hybrid_df.sort_values(
    by="hybrid_score",
    ascending=False
)

Lấy top N:

top_recommendations = hybrid_df.head(10)
13. Ví dụ người 3 dùng output của người 2 trong Hybrid
from src.content_based import ContentBasedRecommender


content_model = ContentBasedRecommender(
    movies_path="data/movies.csv",
    tags_path="data/tags.csv"
)


content_response = content_model.recommend("Toy Story", 50)


if content_response["status"] == "success":
    content_result = content_response["recommendations"]


    # Giả sử cf_result là output từ Collaborative Filtering
    # cf_result gồm các cột: movieId, title, cf_score


    hybrid_df = cf_result.merge(
        content_result[["movieId", "similarity"]],
        on="movieId",
        how="outer"
    )


    hybrid_df = hybrid_df.rename(columns={
        "similarity": "content_score"
    })


    hybrid_df["cf_score"] = hybrid_df["cf_score"].fillna(0)
    hybrid_df["content_score"] = hybrid_df["content_score"].fillna(0)


    def normalize(series):
        if series.max() == series.min():
            return series
        return (series - series.min()) / (series.max() - series.min())


    hybrid_df["cf_score_norm"] = normalize(hybrid_df["cf_score"])
    hybrid_df["content_score_norm"] = normalize(hybrid_df["content_score"])


    hybrid_df["hybrid_score"] = (
        0.6 * hybrid_df["cf_score_norm"] +
        0.4 * hybrid_df["content_score_norm"]
    )


    hybrid_df = hybrid_df.sort_values(
        by="hybrid_score",
        ascending=False
    )


    print(hybrid_df.head(10))
14. Lưu ý quan trọng cho người 3
14.1. Không nên khởi tạo ContentBasedRecommender nhiều lần

Không nên viết như sau trong vòng lặp:

while True:
    content_model = ContentBasedRecommender(...)

Vì mỗi lần khởi tạo, chương trình phải đọc dữ liệu và build TF-IDF lại từ đầu.

Nên khởi tạo một lần:

content_model = ContentBasedRecommender(
    movies_path="data/movies.csv",
    tags_path="data/tags.csv"
)

Sau đó dùng lại:

response = content_model.recommend("Toy Story", 10)
response = content_model.recommend("Matrix", 10)
response = content_model.recommend("Jumanji", 10)
14.2. Dùng movieId để merge, không dùng title

Khi ghép output của người 2 với output của người 3, nên merge bằng:

movieId

Không nên merge bằng:

title

Lý do:

- title có thể dài
- title có thể chứa dấu phẩy, năm phát hành
- title có thể bị sai khác format
- movieId là khóa ổn định hơn
14.3. Điểm similarity nằm trong khoảng 0 đến 1

Cột similarity của content-based có ý nghĩa:

0.0 = không giống
1.0 = rất giống

Trong khi điểm Collaborative Filtering có thể nằm trong thang khác, ví dụ:

0.5 đến 5.0

Vì vậy khi làm Hybrid, cần chuẩn hóa điểm trước khi cộng.

14.4. Có thể lấy top_n lớn hơn khi làm hybrid

Nếu chỉ in kết quả content-based, có thể dùng:

top_n = 10

Nhưng nếu dùng để ghép hybrid, nên lấy nhiều hơn:

top_n = 50

hoặc:

top_n = 100

Lý do là để có nhiều ứng viên phim hơn khi merge với Collaborative Filtering.

15. Input và output chuẩn của người 2
Input từ terminal
movie title
top_n

Ví dụ:

Toy Story
10
Input từ code của người 3
response = content_model.recommend("Toy Story", 50)
Output trả về
response["recommendations"]

DataFrame gồm:

movieId
title
genres
similarity

Đây là output chính mà người 3 cần dùng.

16. Ưu điểm của module
- Không cần thông tin cá nhân của user
- Có thể gợi ý phim chỉ từ một phim đầu vào
- Xử lý được cả genres và tags
- Dễ ghép với Collaborative Filtering
- Output có movieId nên dễ merge với các module khác
17. Hạn chế của module
- Nếu phim có ít tags, kết quả chủ yếu dựa vào genres
- Nếu user nhập tên phim sai quá nhiều, hệ thống có thể không tìm thấy
- Content-based thường gợi ý phim giống phim đầu vào, ít tạo sự bất ngờ
- Không tận dụng trực tiếp hành vi rating của user
18. Các phim nên dùng để test

Có thể test nhanh bằng các tên phim sau:

Toy Story
Matrix
Jumanji
Forrest Gump
Batman
Avengers
Titanic

Ví dụ:

response = content_model.recommend("Matrix", 10)
19. Troubleshooting
Lỗi không tìm thấy file

Thông báo có thể gặp:

FileNotFoundError: data/movies.csv

Cách sửa:

Kiểm tra lại thư mục data có chứa movies.csv và tags.csv chưa.

Cấu trúc đúng:

data/movies.csv
data/tags.csv
Lỗi thiếu thư viện

Thông báo có thể gặp:

ModuleNotFoundError: No module named 'pandas'

Cách sửa:

pip install pandas scikit-learn
Không tìm thấy phim

Nếu nhập:

Toy Stroy

hệ thống có thể báo không tìm thấy vì sai chính tả.

Nên nhập:

Toy Story

hoặc một phần tên phim:

Toy
20. Tóm tắt cho người 3

Người 3 chỉ cần nhớ:

from src.content_based import ContentBasedRecommender


content_model = ContentBasedRecommender(
    movies_path="data/movies.csv",
    tags_path="data/tags.csv"
)


response = content_model.recommend("Toy Story", 50)


if response["status"] == "success":
    content_result = response["recommendations"]

content_result là DataFrame chính để sử dụng tiếp.

Các cột quan trọng:

movieId
title
similarity

Khi làm hybrid:
movieId      → dùng để merge
similarity   → content_score