import pandas as pd
import numpy as np
import os
from joblib import load

# Paths
data_path = 'data/data_clean.csv'
model_dir = 'models'
enc_gender_path = os.path.join(model_dir, 'gender_encoder.joblib')
enc_category_path = os.path.join(model_dir, 'category_encoder.joblib')
scaler_path = os.path.join(model_dir, 'scaler.joblib')
kmeans_path = os.path.join(model_dir, 'kmeans.joblib')
rf_path = os.path.join(model_dir, 'rf_classifier.joblib')

# Load models and data
scaler = load(scaler_path)
kmeans = load(kmeans_path)
rf = load(rf_path)
gender_enc = load(enc_gender_path)
category_enc = load(enc_category_path)

data = pd.read_csv(data_path)

# Gán cluster nếu chưa có trong data (cần cho thống kê)
if 'Cluster' not in data.columns:
    features = ['age', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount',
                'gender_encoded', 'category_encoded']
    if not {'gender_encoded', 'category_encoded'}.issubset(data.columns):
        data['gender_encoded'] = gender_enc.transform(data['gender'].str.capitalize())
        data['category_encoded'] = category_enc.transform(data['preferred_category'].str.title())
    X_all = data[features]
    data['Cluster'] = rf.predict(X_all)

def analyze_new(customer):
    customer['gender_encoded'] = gender_enc.transform([customer['gender'].capitalize()])[0]
    customer['category_encoded'] = category_enc.transform([customer['preferred_category'].title()])[0]
    X_new = np.array([[customer[k] for k in ['age','spending_score','membership_years','purchase_frequency','last_purchase_amount','gender_encoded','category_encoded']]])
    cluster = rf.predict(X_new)[0]
    return cluster

def get_cluster_name(cluster, data):
    cluster_data = data[data['Cluster'] == cluster]
    avg_spending = cluster_data['spending_score'].mean()
    avg_age = cluster_data['age'].mean()
    avg_frequency = cluster_data['purchase_frequency'].mean()

    if avg_spending > 70 and avg_frequency > 30:
        return "Khách hàng VIP"
    elif avg_spending > 60 and avg_age < 35:
        return "Khách hàng trẻ, chi tiêu cao"
    elif avg_spending < 40 and avg_age > 50:
        return "Khách hàng lớn tuổi, chi tiêu thấp"
    else:
        return "Khách hàng tiềm năng"

def get_cluster_description(cluster, data):
    cluster_data = data[data['Cluster'] == cluster]
    return {
        'Số lượng khách hàng tương tự': len(cluster_data),
        'Tuổi trung bình của nhóm': cluster_data['age'].mean(),
        'Điểm chi tiêu trung bình': cluster_data['spending_score'].mean(),
        'Năm thành viên trung bình': cluster_data['membership_years'].mean(),
        'Tần suất mua hàng trung bình': cluster_data['purchase_frequency'].mean(),
        'Số tiền mua hàng trung bình': cluster_data['last_purchase_amount'].mean(),
        'Giới tính phổ biến': cluster_data['gender'].mode().iloc[0] if not cluster_data['gender'].mode().empty else 'N/A',
        'Danh mục ưa thích phổ biến': cluster_data['preferred_category'].mode().iloc[0] if not cluster_data['preferred_category'].mode().empty else 'N/A'
    }

def main():
    print("\nNhập thông tin khách hàng mới:")
    customer_data = {
        'age': float(input("Tuổi: ")),
        'gender': input("Giới tính (Male/Female/Other): "),
        'spending_score': float(input("Điểm chi tiêu (0-100): ")),
        'membership_years': int(input("Số năm thành viên: ")),
        'purchase_frequency': int(input("Tần suất mua hàng: ")),
        'last_purchase_amount': float(input("Số tiền mua hàng cuối: ")),
        'preferred_category': input("Danh mục ưa thích: ")
    }

    cluster = analyze_new(customer_data)
    cluster_name = get_cluster_name(cluster, data)
    cluster_description = get_cluster_description(cluster, data)

    print(f"\nKhách hàng này thuộc nhóm: {cluster_name}")
    print("\nThông tin chi tiết về nhóm khách hàng này:")
    for key, value in cluster_description.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

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
    else:
        print("- Chương trình tích điểm")
        print("- Ưu đãi định kỳ")
        print("- Thông báo khuyến mãi thường xuyên")

if __name__ == '__main__':
    main()