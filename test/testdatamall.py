import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================
# BƯỚC 1: Load và xử lý dữ liệu
# ============================

data = pd.read_csv("./Mall_Customers.csv")  # Đổi tên file nếu cần

# Chuyển 'Gender' thành số: Male = 1, Female = 0
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Chọn các đặc trưng để phân nhóm
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# =====================================
# BƯỚC 2: Chuẩn hóa dữ liệu với StandardScaler
# =====================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =================================
# BƯỚC 3: Elbow Method chọn số cụm
# =================================

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Số cụm (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method để chọn số cụm tối ưu')
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# ===============================
# BƯỚC 4: Áp dụng KMeans (giả sử k = 5)
# ===============================

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ==============================
# BƯỚC 5: Trực quan kết quả phân nhóm
# ==============================

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10')
plt.title('Phân nhóm khách hàng theo thu nhập và điểm chi tiêu')
plt.savefig("customer_segmentation.png")
plt.show()

# ==============================
# BƯỚC 6: Phân tích đặc điểm từng cụm
# ==============================

group_analysis = data.groupby('Cluster').mean(numeric_only=True)
print("Phân tích trung bình đặc điểm của từng cụm:")
print(group_analysis)

# Lưu kết quả ra file
data.to_csv("segmented_customers.csv", index=False)
