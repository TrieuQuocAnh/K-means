from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# Load dữ liệu
df = pd.read_csv("k_means/Mall_Customers.csv")

# Đổi tên cột về chữ thường + bỏ khoảng trắng thừa để dễ xử lý
df.columns = df.columns.str.strip()

@app.route("/")
def index():
    # Mặc định hiển thị dataset gốc
    data = df.head(10).to_html(classes="table table-bordered", index=False)
    return render_template("index.html", tables=data, score=None, k=None)


@app.route("/cluster", methods=["POST"])
def cluster():
    # Lấy số cụm từ form
    k = int(request.form["k"])

    # Chọn đặc trưng để phân cụm (theo file của bạn)
    features = ["Annual Income ($)", "Spending Score (1-100)"]
    X = df[features]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Huấn luyện KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Tính Silhouette Score
    score = silhouette_score(X_scaled, df["Cluster"])

    # Lấy 10 dòng đầu để hiển thị
    data = df[["Gender", "Age", "Annual Income ($)", "Spending Score (1-100)", "Cluster"]].head(10).to_html(classes="table table-bordered", index=False)

    return render_template("index.html", tables=data, score=round(score, 3), k=k)


if __name__ == "__main__":
    app.run(debug=True)
