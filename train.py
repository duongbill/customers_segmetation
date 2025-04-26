import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv('data/data_clean.csv')

# Sao chép dữ liệu để xử lý
df = data.copy()

# Mã hóa cột 'gender' và 'preferred_category'
label_encoders = {}
for col in ['gender', 'preferred_category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Các cột dùng cho phân cụm
features = [
    'age',
    'gender',
    'spending_score',
    'membership_years',
    'purchase_frequency',
    'preferred_category',
    'last_purchase_amount'
]

X = df[features]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tính Inertia cho các giá trị k khác nhau
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method để chọn số cụm tối ưu')
plt.xlabel('Số cụm (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
