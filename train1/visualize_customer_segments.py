import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Đọc dữ liệu
df = pd.read_csv('data/Mall_Customers.csv')

# Chọn các features để phân cụm
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Chuẩn hóa dữ liệu - sử dụng RobustScaler để giảm ảnh hưởng của outliers
scaler = RobustScaler()
X = scaler.fit_transform(df[features])

# Tìm số cụm tối ưu bằng phương pháp Elbow và Silhouette
inertias = []
silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Xác định số cụm tối ưu dựa trên điểm gấp khúc của đồ thị Elbow
# Tìm điểm có độ dốc thay đổi lớn nhất
elbow_scores = np.diff(inertias) / np.diff(K)
optimal_k = K[np.argmin(elbow_scores) + 1]
print(f"Số cụm tối ưu theo phương pháp Elbow: {optimal_k}")

# Thực hiện KMeans với số cụm tối ưu
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Thiết lập style cho biểu đồ


# Tạo bảng màu dựa trên số lượng cụm
if optimal_k <= 4:
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
elif optimal_k <= 6:
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
else:
    colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

# 1. Biểu đồ phân tán 2D cho các cặp features quan trọng
plt.figure(figsize=(20, 15))

# Plot 1: Income vs Spending Score
plt.subplot(2, 2, 1)
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
                c=[colors[i]], label=f'Cụm {i}', alpha=0.7, s=100)
plt.xlabel('Thu nhập')
plt.ylabel('Điểm chi tiêu')
plt.title('Phân bố Thu nhập và Điểm chi tiêu theo nhóm')
plt.legend()

# Plot 2: Age vs Purchase Frequency
plt.subplot(2, 2, 2)
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Spending Score (1-100)'], 
                c=[colors[i]], label=f'Cụm {i}', alpha=0.7, s=100)
plt.xlabel('Tuổi')
plt.ylabel('Điểm chi tiêu')
plt.title('Phân bố Tuổi và Điểm chi tiêu theo nhóm')
plt.legend()

# Plot 3: Membership Years vs Last Purchase Amount
plt.subplot(2, 2, 3)
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], 
                c=[colors[i]], label=f'Cụm {i}', alpha=0.7, s=100)
plt.xlabel('Thu nhập')
plt.ylabel('Điểm chi tiêu')
plt.title('Phân bố Thu nhập và Điểm chi tiêu theo nhóm')
plt.legend()

# Plot 4: Age vs Income
plt.subplot(2, 2, 4)
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], 
                c=[colors[i]], label=f'Cụm {i}', alpha=0.7, s=100)
plt.xlabel('Tuổi')
plt.ylabel('Thu nhập')
plt.title('Phân bố Tuổi và Thu nhập theo nhóm')
plt.legend()

plt.tight_layout()
plt.savefig('customer_segments_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Biểu đồ box plot cho các features
plt.figure(figsize=(20, 10))

for idx, feature in enumerate(features, 1):
    plt.subplot(2, 3, idx)
    sns.boxplot(x='Cluster', y=feature, data=df, palette=colors)
    plt.title(f'Phân bố {feature} theo nhóm')

plt.tight_layout()
plt.savefig('customer_segments_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Biểu đồ radar cho các nhóm
def make_spider_plot(cluster_data, cluster_num):
    # Tính giá trị trung bình cho mỗi feature
    means = cluster_data[features].mean()
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    means_scaled = scaler.fit_transform(means.values.reshape(1, -1))[0]
    
    # Tính góc cho mỗi feature
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    
    # Đóng đường radar
    values = np.concatenate((means_scaled, [means_scaled[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    return angles, values

# Vẽ biểu đồ radar
fig = plt.figure(figsize=(15, 15))
rows = (optimal_k + 2) // 3  # Làm tròn lên
cols = min(3, optimal_k)

for i in range(optimal_k):
    ax = fig.add_subplot(rows, cols, i+1, projection='polar')
    cluster_data = df[df['Cluster'] == i]
    angles, values = make_spider_plot(cluster_data, i)
    
    ax.plot(angles, values, color=colors[i], linewidth=2)
    ax.fill(angles, values, color=colors[i], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f'Đặc điểm cụm {i}')

plt.tight_layout()
plt.savefig('customer_segments_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Biểu đồ phân bố giới tính theo nhóm
plt.figure(figsize=(15, 8))
gender_dist = pd.crosstab(df['Cluster'], df['Gender'])
gender_dist_pct = gender_dist.div(gender_dist.sum(axis=1), axis=0) * 100

gender_dist_pct.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF'])
plt.title('Phân bố giới tính theo nhóm (%)')
plt.xlabel('Nhóm khách hàng')
plt.ylabel('Phần trăm')
plt.legend(title='Giới tính', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('customer_segments_gender.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Biểu đồ phân bố kích thước các cụm
plt.figure(figsize=(10, 6))
cluster_sizes = df['Cluster'].value_counts().sort_index()
plt.bar(cluster_sizes.index, cluster_sizes.values, color=colors)
plt.title('Phân bố kích thước các cụm')
plt.xlabel('Cụm')
plt.ylabel('Số lượng khách hàng')
for i, v in enumerate(cluster_sizes.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig('customer_segments_sizes.png', dpi=300, bbox_inches='tight')
plt.close()

print("Đã tạo xong các biểu đồ phân tích phân cụm khách hàng!")
print("Các file biểu đồ đã được lưu:")
print("1. customer_segments_2d.png - Biểu đồ phân tán 2D")
print("2. customer_segments_boxplot.png - Biểu đồ box plot")
print("3. customer_segments_radar.png - Biểu đồ radar")
print("4. customer_segments_gender.png - Biểu đồ phân bố giới tính")
print("5. customer_segments_sizes.png - Biểu đồ phân bố kích thước các cụm") 