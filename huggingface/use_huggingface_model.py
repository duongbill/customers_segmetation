import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
import tempfile
import os
from joblib import load

def load_models_from_hub(repo_id="duonggbill/dbill-customer-model"):
    """
    Tải các mô hình từ Hugging Face Hub
    
    Tham số:
    - repo_id: ID của repository trên Hugging Face Hub
    
    Trả về:
    - Dictionary chứa các mô hình đã tải
    """
    print(f"Đang tải các model từ Hugging Face Hub: {repo_id}...")
    
    # Tạo thư mục tạm để lưu các model tải về
    temp_dir = tempfile.mkdtemp()
    
    # Tải tất cả các file từ repository
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=temp_dir,
        ignore_patterns=["*.md", "*.json", ".git*"]
    )
    
    # Load các model từ thư mục tạm
    gender_enc = load(os.path.join(temp_dir, 'gender_encoder.joblib'))
    category_enc = load(os.path.join(temp_dir, 'category_encoder.joblib'))
    scaler = load(os.path.join(temp_dir, 'scaler.joblib'))
    kmeans = load(os.path.join(temp_dir, 'kmeans.joblib'))
    rf = load(os.path.join(temp_dir, 'rf_classifier.joblib'))
    features = load(os.path.join(temp_dir, 'features.joblib'))
    cluster_stats, cluster_descriptions = load(os.path.join(temp_dir, 'cluster_info.joblib'))
    
    print("Đã tải xong các model từ Hugging Face Hub")
    
    return {
        'gender_encoder': gender_enc,
        'category_encoder': category_enc,
        'scaler': scaler,
        'kmeans': kmeans,
        'rf_classifier': rf,
        'features': features,
        'cluster_stats': cluster_stats,
        'cluster_descriptions': cluster_descriptions
    }

def predict_customer_cluster_hub(customer_data, models):
    """
    Dự đoán cluster cho khách hàng mới sử dụng mô hình từ Hugging Face
    
    Tham số:
    - customer_data: dict chứa thông tin khách hàng
    - models: dict chứa các model đã tải từ Hugging Face
    
    Trả về:
    - dict chứa kết quả dự đoán
    """
    try:
        # Mã hóa các biến phân loại
        if 'gender' in customer_data:
            customer_data['gender_encoded'] = models['gender_encoder'].transform([customer_data['gender']])[0]
        
        if 'preferred_category' in customer_data:
            customer_data['category_encoded'] = models['category_encoder'].transform([customer_data['preferred_category']])[0]
        
        # Tạo feature vector
        features = models['features']
        X = np.array([[customer_data.get(f, 0) for f in features]])
        
        # Chuẩn hóa dữ liệu
        X_scaled = models['scaler'].transform(X)
        
        # Dự đoán cluster
        cluster = models['rf_classifier'].predict(X)[0]
        
        # Lấy thông tin về cluster
        description = models['cluster_descriptions'][cluster]
        stats = models['cluster_stats'].loc[cluster]
        
        return {
            'cluster': int(cluster),
            'description': description,
            'stats': stats
        }
    
    except Exception as e:
        print(f"Lỗi khi dự đoán cluster: {str(e)}")
        return None

if __name__ == "__main__":
    # Ví dụ sử dụng
    # Tải mô hình từ Hugging Face Hub
    models = load_models_from_hub("duonggbill/dbill-customer-model")
    
    # Dự đoán cho một khách hàng mới
    new_customer = {
        'age': 30,
        'gender': 'Female',
        'spending_score': 75,
        'membership_years': 3,
        'purchase_frequency': 25,
        'preferred_category': 'Clothing',
        'last_purchase_amount': 350.0
    }
    
    result = predict_customer_cluster_hub(new_customer, models)
    
    if result:
        print(f"Khách hàng thuộc cụm: {result['cluster']}")
        print(f"Mô tả: {result['description']}")
        print(f"Thống kê cụm:")
        print(f"- Số lượng khách hàng: {result['stats'][('age', 'count')]}")
        print(f"- Tuổi trung bình: {result['stats'][('age', 'mean')]:.1f}")
        print(f"- Điểm chi tiêu trung bình: {result['stats'][('spending_score', 'mean')]:.1f}")