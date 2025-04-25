import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Đọc dữ liệu
df = pd.read_csv('data/Mall_Customers.csv')

# Chọn các features để phân cụm
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Chuẩn hóa dữ liệu - sử dụng RobustScaler để giảm ảnh hưởng của outliers
scaler = RobustScaler()
X = scaler.fit_transform(df[features])

# Lưu scaler
joblib.dump(scaler, 'scaler.pkl')

# Tìm số cụm tối ưu bằng phương pháp Elbow và Silhouette
inertias = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Vẽ đồ thị Elbow
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Số cụm (k)')
plt.ylabel('Inertia')
plt.title('Phương pháp Elbow')

# Vẽ đồ thị Silhouette
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('Số cụm (k)')
plt.ylabel('Silhouette Score')
plt.title('Phương pháp Silhouette')

plt.tight_layout()
plt.savefig('optimal_clusters.png')
plt.close()

# Xác định số cụm tối ưu dựa trên điểm gấp khúc của đồ thị Elbow
# Tìm điểm có độ dốc thay đổi lớn nhất
elbow_scores = np.diff(inertias) / np.diff(K)
optimal_k_elbow = K[np.argmin(elbow_scores) + 1]

# Xác định số cụm tối ưu dựa trên điểm cao nhất của đồ thị Silhouette
optimal_k_silhouette = K[np.argmax(silhouette_scores)]

print(f"Số cụm tối ưu theo phương pháp Elbow: {optimal_k_elbow}")
print(f"Số cụm tối ưu theo phương pháp Silhouette: {optimal_k_silhouette}")

# Sử dụng số cụm tối ưu từ phương pháp Elbow
optimal_k = optimal_k_elbow

# Thực hiện KMeans với số cụm tối ưu
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Lưu model
joblib.dump(kmeans, 'kmeans_model.pkl')

# Thử nghiệm DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Thử nghiệm Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(X)

# So sánh kết quả của các phương pháp
print("\nSo sánh kết quả phân cụm:")
print(f"KMeans - Số cụm: {len(np.unique(df['Cluster']))}")
print(f"DBSCAN - Số cụm: {len(np.unique(dbscan_labels))}")
print(f"Agglomerative - Số cụm: {len(np.unique(agg_labels))}")

# Phân tích các cụm
cluster_analysis = df.groupby('Cluster')[features].mean()
print("\nPhân tích trung bình của các cụm:")
print(cluster_analysis)

# Vẽ biểu đồ phân tán cho một số cặp features
plt.figure(figsize=(15, 10))

# Plot 1: Income vs Spending Score
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='deep', s=100, alpha=0.7)
plt.title('Thu nhập vs Điểm chi tiêu')

# Plot 2: Age vs Purchase Frequency
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='deep', s=100, alpha=0.7)
plt.title('Tuổi vs Điểm chi tiêu')

# Plot 3: Membership Years vs Last Purchase Amount
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Cluster', palette='deep', s=100, alpha=0.7)
plt.title('Tuổi vs Thu nhập')

# Plot 4: Age vs Income
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Cluster', palette='deep', s=100, alpha=0.7)
plt.title('Tuổi vs Thu nhập')

plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=300)
plt.close()

# Lưu kết quả phân cụm vào file CSV
df.to_csv('customer_segments.csv', index=False)

# In thông tin chi tiết về từng cụm
print("\nThông tin chi tiết về các cụm:")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCụm {cluster}:")
    print(f"Số lượng khách hàng: {len(cluster_data)}")
    print(f"Độ tuổi trung bình: {cluster_data['Age'].mean():.2f}")
    print(f"Thu nhập trung bình: {cluster_data['Annual Income (k$)'].mean():.2f}")
    print(f"Điểm chi tiêu trung bình: {cluster_data['Spending Score (1-100)'].mean():.2f}")
    
    # Phân tích giới tính
    gender_counts = cluster_data['Gender'].value_counts()
    print("Phân bố giới tính:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} ({count/len(cluster_data)*100:.1f}%)") 