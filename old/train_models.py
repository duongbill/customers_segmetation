
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from joblib import dump
import seaborn as sns

os.makedirs('models', exist_ok=True)
# Paths
data_path = 'data/data_clean.csv'
model_dir = 'models'

# Encoder and model paths
enc_gender_path = os.path.join(model_dir, 'gender_encoder.joblib')
enc_category_path = os.path.join(model_dir, 'category_encoder.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
kmeans_path = os.path.join(model_dir, 'kmeans.joblib')
rf_path = os.path.join(model_dir, 'rf_classifier.joblib')
features_path = os.path.join(model_dir, 'features.joblib')
cluster_info_path = os.path.join(model_dir, 'cluster_info.joblib')

# 1. Load and encode data
def load_and_prepare_data(path=data_path):
    print("Loading and preparing data...")
    df = pd.read_csv(path)
    
    # Label encoding
    gender_enc = LabelEncoder()
    category_enc = LabelEncoder()

    df['gender_encoded'] = gender_enc.fit_transform(df['gender'])
    df['category_encoded'] = category_enc.fit_transform(df['preferred_category'])
    
    # Save encoders
    os.makedirs(model_dir, exist_ok=True)
    dump(gender_enc, enc_gender_path)
    dump(category_enc, enc_category_path)
    
    return df

# 2. Train and save models
def train_and_save_models(df, n_clusters=4, rf_trees=100):
    print("Training models...")
    os.makedirs(model_dir, exist_ok=True)
    
    # Sử dụng tất cả các features quan trọng
    features = [
        'age', 
        'spending_score', 
        'membership_years', 
        'purchase_frequency', 
        'last_purchase_amount',
        'gender_encoded',
        'category_encoded'
    ]
    
    X = df[features]
    
    # Scale
    dscaler = StandardScaler().fit(X)
    X_scaled = dscaler.transform(X)
    dump(dscaler, scaler_path)
    
    # KMeans
    print(f"Training KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    df['Cluster'] = kmeans.predict(X_scaled)
    
    # Đánh giá chất lượng clustering
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # RandomForest
    print(f"Training RandomForest with {rf_trees} trees...")
    rf = RandomForestClassifier(n_estimators=rf_trees, random_state=42)
    rf.fit(X, df['Cluster'])
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature importance:")
    print(feature_importance)
    
    # Tạo thông tin chi tiết về các cluster
    cluster_stats = df.groupby('Cluster').agg({
        'age': ['mean', 'min', 'max', 'count'],
        'spending_score': ['mean', 'min', 'max'],
        'membership_years': ['mean', 'min', 'max'],
        'purchase_frequency': ['mean', 'min', 'max'],
        'last_purchase_amount': ['mean', 'min', 'max']
    }).round(2)
    
    # Tạo mô tả cho từng cluster dựa trên dữ liệu
    cluster_descriptions = {}
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        avg_spending = cluster_data['spending_score'].mean()
        avg_age = cluster_data['age'].mean()
        avg_frequency = cluster_data['purchase_frequency'].mean()
        
        if avg_spending > 70 and avg_frequency > 30:
            description = "Khách hàng VIP - Chi tiêu cao, mua sắm thường xuyên"
        elif avg_spending > 60 and avg_age < 35:
            description = "Khách hàng trẻ, chi tiêu cao"
        elif avg_spending < 40 and avg_age > 50:
            description = "Khách hàng lớn tuổi, chi tiêu thấp"
        elif avg_spending > 50 and avg_frequency < 20:
            description = "Khách hàng tiềm năng - Chi tiêu khá, mua sắm không thường xuyên"
        else:
            description = "Khách hàng thông thường - Chi tiêu trung bình"
        
        cluster_descriptions[cluster] = description
    
    # Lưu models và thông tin
    dump(kmeans, kmeans_path)
    dump(rf, rf_path)
    dump(features, features_path)
    dump((cluster_stats, cluster_descriptions), cluster_info_path)
    
    print(f"\nModels saved under '{model_dir}/'")
    
    # Vẽ biểu đồ phân cụm
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['age'], df['spending_score'], 
                         c=df['Cluster'], 
                         cmap='viridis', 
                         s=100,
                         alpha=0.6)
    
    # Thêm centroids
    centroids = dscaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', 
               marker='x', 
               s=200, 
               linewidths=3, 
               label='Centroids')
    
    plt.title('Phân cụm khách hàng theo tuổi và điểm chi tiêu', fontsize=14)
    plt.xlabel('Tuổi', fontsize=12)
    plt.ylabel('Điểm chi tiêu', fontsize=12)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(model_dir, 'cluster_visualization.png'))
    plt.show()
    
    # In thông tin về các cluster
    print("\nThông tin chi tiết về các cluster:")
    for cluster, description in cluster_descriptions.items():
        count = cluster_stats.loc[cluster][('age', 'count')]
        avg_age = cluster_stats.loc[cluster][('age', 'mean')]
        avg_spending = cluster_stats.loc[cluster][('spending_score', 'mean')]
        print(f"Cluster {cluster}: {description}")
        print(f"  - Số lượng khách hàng: {count}")
        print(f"  - Tuổi trung bình: {avg_age}")
        print(f"  - Điểm chi tiêu trung bình: {avg_spending}")
        print()
    
    return df

# 3. Hàm dự đoán cho khách hàng mới
def predict_customer_cluster(customer_data):
    """
    Dự đoán cluster cho khách hàng mới
    
    customer_data: dict chứa thông tin khách hàng
    """
    try:
        # Load models
        scaler = load(scaler_path)
        kmeans = load(kmeans_path)
        rf = load(rf_path)
        features = load(features_path)
        cluster_stats, cluster_descriptions = load(cluster_info_path)
        
        # Chuẩn bị dữ liệu
        X_new = np.array([[customer_data[feature] for feature in features]])
        
        # Chuẩn hóa dữ liệu
        X_new_scaled = scaler.transform(X_new)
        
        # Dự đoán cluster
        cluster = rf.predict(X_new)[0]
        
        return {
            'cluster': cluster,
            'description': cluster_descriptions[cluster],
            'stats': cluster_stats.loc[cluster]
        }
        
    except Exception as e:
        print(f"Error predicting cluster: {str(e)}")
        return None

if __name__ == '__main__':
    data = load_and_prepare_data()
    data_with_clusters = train_and_save_models(data)
    
    # Ví dụ dự đoán cho khách hàng mới
    print("\nDự đoán cho khách hàng mới:")
    new_customer = {
        'age': 30,
        'spending_score': 70,
        'membership_years': 2,
        'purchase_frequency': 15,
        'last_purchase_amount': 500,
        'gender_encoded': 0,  # Nam
        'category_encoded': 1  # Electronics
    }
    
    result = predict_customer_cluster(new_customer)
    if result:
        print(f"Khách hàng thuộc cluster: {result['cluster']}")
        print(f"Mô tả: {result['description']}")



