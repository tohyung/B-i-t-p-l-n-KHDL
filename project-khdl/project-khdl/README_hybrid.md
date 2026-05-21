# 1. Vai trò của module

Module này là hệ thống **Hybrid Recommendation** (Gợi ý Lai) được xây dựng dựa trên sự kết hợp giữa hai phương pháp:
- **Collaborative Filtering** (Sử dụng thuật toán phân rã ma trận SVD) do Người 3 phụ trách.
- **Content-based Recommendation** do Người 2 phụ trách.

Phần này do **Người 3** phát triển. Mục tiêu của module là mang lại kết quả gợi ý tốt nhất bằng cách kết hợp ưu điểm của cả hai hệ thống, vừa lấy được sở thích cá nhân của người dùng, vừa tìm được những bộ phim có nội dung tương đồng với phim yêu thích.

---

# 2. Các file mã nguồn mới

Để tôn trọng source code của các thành viên trước, **Người 3 tuyệt đối không chỉnh sửa bất kỳ file nào của Người 1 và Người 2**. Hệ thống được xây dựng hoàn toàn dựa trên các file mới sau:

1. `src/collaborative_filtering.py`: Chứa class `SVDRecommender` dùng thuật toán SVD (Singular Value Decomposition) để tái tạo ma trận User-Item và dự đoán rating.
2. `src/hybrid_recommender.py`: Chứa class `HybridRecommender` kết nối mô hình SVD và Content-based. Thực hiện quá trình chuẩn hóa (Min-Max Scaling) và kết hợp điểm.
3. `hybrid_main.py`: File chạy chương trình chính trên giao diện dòng lệnh (Terminal). File này kết nối trực tiếp vào MySQL Database để load dữ liệu thật lúc khởi động thay vì đọc file CSV, nhằm tích hợp hoàn hảo với hệ thống Database của Người 1.

---

# 3. Yêu cầu cài đặt thư viện

Ngoài các thư viện sẵn có của nhóm (`pandas`, `scikit-learn`, `mysql-connector-python`), module của Người 3 cần sử dụng thêm thuật toán toán học tối ưu từ thư viện `scipy`.

Chạy lệnh sau trên terminal để cài đặt (nếu chưa có):
```bash
pip install numpy scipy
```
> **Lưu ý:** Nếu bạn dùng VS Code và thấy gạch đỏ ở phần `import scipy`, đó chỉ là do VS Code đang chọn sai môi trường Python. Hãy bấm `Ctrl + Shift + P`, gõ `Python: Select Interpreter` và chọn đúng phiên bản Python mà bạn đang chạy trên terminal.

---

# 4. Nguyên lý hoạt động (Hybrid System)

Thay vì chạy độc lập, hệ thống Hybrid của Người 3 hoạt động theo quy trình nối tiếp như sau:

1. **Khởi tạo dữ liệu (DB Connection):** Chương trình tự động đọc các bảng `Movies`, `Tags`, `Ratings` từ MySQL vào bộ nhớ.
2. **Lọc ứng viên bằng Content-Based:** Khi user nhập một tên phim, hệ thống gọi `ContentBasedRecommender` của Người 2 để lấy ra **Top 100** phim có nội dung tương đồng nhất (Candidates).
3. **Chấm điểm cá nhân hóa bằng SVD:** Hệ thống gọi `SVDRecommender` dự đoán xem User hiện tại sẽ chấm bao nhiêu điểm (Predicted Rating) cho 100 bộ phim ứng viên kể trên.
4. **Chuẩn hóa (Normalization):** Điểm Similarity (khoảng 0-1) và điểm SVD (khoảng 0.5-5.0) được kéo về cùng hệ quy chiếu `[0.0, 1.0]` bằng Min-Max Scaling.
5. **Tính Final Score:** Áp dụng công thức trọng số kết hợp:
   `Final_Score = 0.7 * SVD_Score + 0.3 * Content_Score`
6. **Xếp hạng:** Sắp xếp danh sách theo `Final_Score` giảm dần và trả về Top N phim cho người dùng.

---

# 5. Cách chạy chương trình

Tại thư mục gốc của project (cùng cấp với `hybrid_main.py`), hãy đảm bảo MySQL Server đang chạy và bạn đã import dữ liệu ở bước của Người 1, sau đó chạy lệnh:

```bash
python hybrid_main.py
```

### 5.1. Quá trình nạp dữ liệu
Chương trình sẽ hiển thị log kết nối DB và huấn luyện mô hình (mất khoảng 2-4 giây):
```
[HỆ THỐNG] Đang kết nối database MySQL...
[HỆ THỐNG] Kết nối thành công! Đang tải dữ liệu để train mô hình...

  -> Tải dữ liệu Phim và Tags từ DB (Content-based)...
  -> Tải dữ liệu Ratings (SVD Collaborative Filtering)...
  -> Đang huấn luyện ma trận SVD...

[HỆ THỐNG] Huấn luyện hoàn tất!
```

### 5.2. Cách nhập Input
Menu chính sẽ hiện ra. Bấm `1` để chọn Recommend phim. 
Khác với Content-based (chỉ cần tên phim), mô hình Hybrid cần biết **User ID** là ai để dự đoán rating.

```
Select option: 1
Enter User ID (e.g. 1): 1
Enter Favorite Movie Title (e.g. Toy Story): Toy Story
Enter number of recommendations (default 10): 10
```

### 5.3. Output trả về
Hệ thống sẽ trả về một bảng trực quan chứa cả 3 loại điểm số để Người 1 và Người 2 dễ dàng theo dõi cách thuật toán Hybrid chấm điểm:

```
=========================================================================================================
HYBRID RECOMMENDATION (SVD Collaborative Filtering + Content-based)
Weight: 0.7 * SVD_Score + 0.3 * Content_Score
=========================================================================================================

User ID      : 1
Input Movie  : Toy Story (1995) (ID: 1)

Recommended Movies:

Rank  Movie ID  Title                                    SVD Score    Content      Final Score 
---------------------------------------------------------------------------------------------------------
1     3114      Toy Story 2 (1999)                         0.9250      0.7982         0.8869
2     2355      Bug's Life, A (1998)                       0.7850      0.8446         0.8028
3     2294      Antz (1998)                                0.6500      0.7546         0.6813
...
```

*Cảm ơn Người 1 và Người 2 đã chuẩn bị data và module Content-Based cực kỳ cẩn thận để Người 3 có thể dễ dàng lắp ráp hệ thống Hybrid này!*
