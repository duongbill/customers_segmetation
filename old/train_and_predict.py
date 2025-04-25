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
    return data

def predict_customer_segment():
    """
    Dự đoán phân khúc cho khách hàng mới
    """
    try:
        # Load các mô hình
        kmeans = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        gender_encoder = joblib.load('models/gender_encoder.pkl')
        category_encoder = joblib.load('models/category_encoder.pkl')
        features = joblib.load('models/features.pkl')
        
        # Nhập thông tin khách hàng
        print("\nNhập thông tin khách hàng mới:")
        customer_data = {
            'age': float(input("Tuổi: ")),
            'gender': input("Giới tính (Nam/Nữ/Khác): "),
            'spending_score': float(input("Điểm chi tiêu (0-100): ")),
            'membership_years': float(input("Số năm là thành viên: ")),
            'purchase_frequency': float(input("Tần suất mua hàng (số lần/tháng): ")),
            'preferred_category': input("Danh mục ưa thích (Điện tử/Thời trang/Thực phẩm/Thể thao/Gia dụng): "),
            'last_purchase_amount': float(input("Số tiền mua hàng gần nhất (VNĐ): "))
        }
        
        # Chuẩn hóa giới tính và danh mục
        gender_mapping = {'Nam': 'Male', 'Nữ': 'Female', 'Khác': 'Other'}
        category_mapping = {
            'Điện tử': 'Electronics',
            'Thời trang': 'Clothing',
            'Thực phẩm': 'Groceries',
            'Thể thao': 'Sports',
            'Gia dụng': 'Home & Garden'
        }
        
        customer_data['gender'] = gender_mapping[customer_data['gender']]
        customer_data['preferred_category'] = category_mapping[customer_data['preferred_category']]
        
        # Mã hóa giới tính và danh mục
        customer_data['gender_encoded'] = gender_encoder.transform([customer_data['gender']])[0]
        customer_data['category_encoded'] = category_encoder.transform([customer_data['preferred_category']])[0]
        
        # Tạo feature vector
        X_new = np.array([[
            customer_data['age'],
            customer_data['spending_score'],
            customer_data['membership_years'],
            customer_data['purchase_frequency'],
            customer_data['last_purchase_amount'],
            customer_data['gender_encoded'],
            customer_data['category_encoded']
        ]])
        
        # Chuẩn hóa dữ liệu
        X_new_scaled = scaler.transform(X_new)
        
        # Dự đoán cluster
        cluster = kmeans.predict(X_new_scaled)[0]
        
        # Mô tả các cluster
        cluster_descriptions = {
            0: "Khách hàng tiết kiệm - Chi tiêu thấp, thu nhập trung bình",
            1: "Khách hàng tiềm năng - Chi tiêu trung bình, thu nhập cao",
            2: "Khách hàng VIP - Chi tiêu cao, thu nhập cao",
            3: "Khách hàng thông thường - Chi tiêu và thu nhập trung bình",
            4: "Khách hàng cần chú ý - Chi tiêu cao nhưng thu nhập thấp"
        }
        
        print(f"\nKết quả phân khúc khách hàng:")
        print(f"Nhóm khách hàng: {cluster}")
        print(f"Mô tả: {cluster_descriptions[cluster]}")
        
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

def main():
    while True:
        print("\n=== CHƯƠNG TRÌNH PHÂN KHÚC KHÁCH HÀNG ===")
        print("1. Huấn luyện lại mô hình")
        print("2. Dự đoán phân khúc khách hàng mới")
        print("3. Thoát")
        
        choice = input("\nVui lòng chọn chức năng (1-3): ")
        
        if choice == '1':
            train_and_save_models()
        elif choice == '2':
            predict_customer_segment()
        elif choice == '3':
            print("Cảm ơn bạn đã sử dụng chương trình!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại!")

if __name__ == "__main__":
    main()