import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

def load_and_prepare_data():
    # Đọc dữ liệu
    data = pd.read_csv('data/data_clean.csv')
    
    # Chuẩn hóa dữ liệu phân loại
    gender_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    
    # Fit và transform dữ liệu
    data['gender_encoded'] = gender_encoder.fit_transform(data['gender'])
    data['category_encoded'] = category_encoder.fit_transform(data['preferred_category'])
    
    return data, gender_encoder, category_encoder

def train_models(data):
    # Chọn features
    features = ['age', 'spending_score', 'membership_years', 
               'purchase_frequency', 'last_purchase_amount',
               'gender_encoded', 'category_encoded']
    
    X = data[features]
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Thực hiện K-means
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Huấn luyện Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, data['Cluster'])
    
    return kmeans, rf, scaler, features

def analyze_new_customer(customer_data, kmeans, rf, scaler, features, gender_encoder, category_encoder):
    # Chuẩn hóa dữ liệu đầu vào
    # Chuyển đổi giới tính thành định dạng chuẩn (chữ hoa đầu)
    gender = customer_data['gender'].capitalize()
    customer_data['gender_encoded'] = gender_encoder.transform([gender])[0]
    
    # Chuyển đổi danh mục thành định dạng chuẩn
    category = customer_data['preferred_category'].title()
    customer_data['category_encoded'] = category_encoder.transform([category])[0]
    
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
    cluster = rf.predict(X_new)[0]
    
    return cluster

def get_cluster_description(cluster, data):
    cluster_data = data[data['Cluster'] == cluster]
    
    description = {
        'Số lượng khách hàng tương tự': len(cluster_data),
        'Tuổi trung bình của nhóm': cluster_data['age'].mean(),
        'Điểm chi tiêu trung bình': cluster_data['spending_score'].mean(),
        'Năm thành viên trung bình': cluster_data['membership_years'].mean(),
        'Tần suất mua hàng trung bình': cluster_data['purchase_frequency'].mean(),
        'Số tiền mua hàng trung bình': cluster_data['last_purchase_amount'].mean(),
        'Giới tính phổ biến': cluster_data['gender'].mode().iloc[0] if not cluster_data['gender'].mode().empty else 'N/A',
        'Danh mục ưa thích phổ biến': cluster_data['preferred_category'].mode().iloc[0] if not cluster_data['preferred_category'].mode().empty else 'N/A'
    }
    
    return description

def get_cluster_name(cluster, data):
    cluster_data = data[data['Cluster'] == cluster]
    
    # Tính các chỉ số trung bình của cluster
    avg_spending = cluster_data['spending_score'].mean()
    avg_age = cluster_data['age'].mean()
    avg_frequency = cluster_data['purchase_frequency'].mean()
    
    # Phân loại cluster dựa trên các chỉ số
    if avg_spending > 70 and avg_frequency > 30:
        return "Khách hàng VIP"
    elif avg_spending > 60 and avg_age < 35:
        return "Khách hàng trẻ, chi tiêu cao"
    elif avg_spending < 40 and avg_age > 50:
        return "Khách hàng lớn tuổi, chi tiêu thấp"
    else:
        return "Khách hàng tiềm năng"

def main():
    # Load và chuẩn bị dữ liệu
    data, gender_encoder, category_encoder = load_and_prepare_data()
    
    # Huấn luyện models
    kmeans, rf, scaler, features = train_models(data)
    
    # Nhập thông tin khách hàng mới
    print("\nNhập thông tin khách hàng mới:")
    customer_data = {
        'age': float(input("Tuổi: ")),
        'gender': input("Giới tính (Male/Female/Other): "),
        'spending_score': float(input("Điểm chi tiêu (0-100): ")),
        'membership_years': int(input("Số năm thành viên: ")),
        'purchase_frequency': int(input("Tần suất mua hàng (số lần mua): ")),
        'preferred_category': input("Danh mục ưa thích (Electronics/Clothing/Groceries/Sports/Home & Garden): "),
        'last_purchase_amount': float(input("Số tiền mua hàng cuối cùng: "))
    }
    
    # Phân tích khách hàng
    cluster = analyze_new_customer(customer_data, kmeans, rf, scaler, features, gender_encoder, category_encoder)
    
    # Lấy tên cluster và mô tả
    cluster_name = get_cluster_name(cluster, data)
    cluster_description = get_cluster_description(cluster, data)
    
    # In kết quả
    print(f"\nKhách hàng này thuộc nhóm: {cluster_name}")
    print("\nThông tin chi tiết về nhóm khách hàng này:")
    for key, value in cluster_description.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # In thêm gợi ý marketing
    print("\nGợi ý marketing:")
    if cluster_name == "Khách hàng VIP":
        print("- Ưu đãi đặc biệt cho khách hàng VIP")
        print("- Chương trình tích điểm cao cấp")
        print("- Dịch vụ chăm sóc khách hàng ưu tiên")
    elif cluster_name == "Khách hàng trẻ, chi tiêu cao":
        print("- Chương trình khuyến mãi theo xu hướng")
        print("- Ưu đãi cho sản phẩm mới")
        print("- Tích hợp với mạng xã hội")
    elif cluster_name == "Khách hàng lớn tuổi, chi tiêu thấp":
        print("- Chương trình khuyến mãi theo mùa")
        print("- Ưu đãi cho sản phẩm thiết yếu")
        print("- Dịch vụ hỗ trợ đặc biệt")
    else:  # Khách hàng thường xuyên
        print("- Chương trình tích điểm")
        print("- Ưu đãi định kỳ")
        print("- Thông báo khuyến mãi thường xuyên")

if __name__ == "__main__":
    main()