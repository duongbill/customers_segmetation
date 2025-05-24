import os
import pandas as pd
import numpy as np
from joblib import load
import tempfile
from huggingface_hub import hf_hub_download, snapshot_download

# Đường dẫn đến thư mục chứa model
MODEL_DIR = 'models'

def load_models(use_huggingface=False, repo_id="duonggbill/dbill-customer-model"):
    """
    Load các model và thành phần cần thiết từ thư mục models hoặc từ Hugging Face Hub
    
    Tham số:
    - use_huggingface: Boolean, nếu True sẽ tải model từ Hugging Face Hub
    - repo_id: String, ID của repository trên Hugging Face Hub
    """
    if use_huggingface:
        print(f"Đang tải các model từ Hugging Face Hub: {repo_id}...")
        try:
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
            
        except Exception as e:
            print(f"Lỗi khi tải model từ Hugging Face: {str(e)}")
            print("Chuyển sang sử dụng model cục bộ...")
            return load_models(use_huggingface=False)
    else:
        print("Đang load các model từ thư mục cục bộ...")
        
        # Load encoders
        gender_enc = load(os.path.join(MODEL_DIR, 'gender_encoder.joblib'))
        category_enc = load(os.path.join(MODEL_DIR, 'category_encoder.joblib'))
        
        # Load scaler
        scaler = load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        
        # Load models
        kmeans = load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
        rf = load(os.path.join(MODEL_DIR, 'rf_classifier.joblib'))
        
        # Load features và thông tin cluster
        features = load(os.path.join(MODEL_DIR, 'features.joblib'))
        cluster_stats, cluster_descriptions = load(os.path.join(MODEL_DIR, 'cluster_info.joblib'))
        
        print("Đã load xong các model từ thư mục cục bộ")
    
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