📌 Bài toán

Đây là bài toán phân cụm khách hàng (Customer Segmentation).

Mục tiêu: dựa trên dữ liệu chi tiêu và thu nhập của khách hàng, chia khách hàng thành các nhóm (cluster) có hành vi tiêu dùng tương tự nhau.

Ý nghĩa: giúp doanh nghiệp hiểu khách hàng, xây dựng chiến lược marketing riêng cho từng nhóm, và tối ưu doanh thu.

📌 Thuật toán sử dụng

K-Means Clustering (thuật toán phân cụm không giám sát – unsupervised learning).

Ý tưởng:

Chọn ngẫu nhiên tâm cụm ban đầu.

Gán mỗi điểm dữ liệu vào cụm gần nhất.

Cập nhật lại tâm cụm (centroid).

Lặp lại đến khi các tâm cụm ổn định.

Bạn cũng sử dụng thêm:

Elbow Method: chọn số cụm k tối ưu.

Silhouette Score: đánh giá chất lượng phân cụm.

📌 Công cụ & Thư viện

Python (ngôn ngữ chính).

Thư viện:

pandas → đọc & xử lý dữ liệu.

matplotlib → vẽ biểu đồ trực quan.

scikit-learn (KMeans, StandardScaler, silhouette_score) → thuật toán máy học.

Flask → xây dựng giao diện web truyền thống (hiển thị kết quả phân cụm).

<img width="1693" height="840" alt="image" src="https://github.com/user-attachments/assets/82cc5170-663d-4c24-85a7-9ac0438f372d" />
