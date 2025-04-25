import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import os

def train_and_save_models(data_path='data/data_clean.csv'):
    """
    Huấn luyện và lưu các mô hình
    """
    try:
        # Tạo thư mục models nếu chưa tồn tại
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Đọc dữ liệu
        print("Đang đọc dữ liệu...")
        data = pd.read_csv(data_path)
        
        # Chuẩn hóa dữ liệu phân loại
        print("Đang xử lý dữ liệu...")
        gender_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        
        data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])
        data['category_encoded'] = category_encoder.fit_transform(data['preferred_category'])
        
        # Chọn features
        features = ['age', 'spending_score', 'membership_years', 
                   'purchase_frequency', 'last_purchase_amount',
                   'gender_encoded', 'category_encoded']
        
        X = data[features]
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Huấn luyện K-means
        print("Đang huấn luyện mô hình...")
        kmeans = KMeans(n_clusters=5, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Lưu các mô hình và encoder
        print("Đang lưu mô hình...")
        joblib.dump(kmeans, 'models/kmeans_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(gender_encoder, 'models/gender_encoder.pkl')
        joblib.dump(category_encoder, 'models/category_encoder.pkl')
        joblib.dump(features, 'models/features.pkl')
        
        print("Đã lưu xong các mô hình!")
        return True

    except Exception as e:
        print(f"Có lỗi xảy ra trong quá trình huấn luyện: {str(e)}")
        return False

def main():
    print("\n=== CHƯƠNG TRÌNH HUẤN LUYỆN MÔ HÌNH PHÂN KHÚC KHÁCH HÀNG ===")
    
    # Cho phép người dùng nhập đường dẫn file dữ liệu
    data_path = input("Nhập đường dẫn đến file dữ liệu (Enter để dùng mặc định 'data/data_clean.csv'): ")
    if not data_path:
        data_path = 'data/data_clean.csv'
    
    if train_and_save_models(data_path):
        print("\nHuấn luyện và lưu mô hình thành công!")
    else:
        print("\nHuấn luyện mô hình thất bại!")

if __name__ == "__main__":
    main()