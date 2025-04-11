import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

def load_model_and_scaler():
    """Load the trained KMeans model and scaler"""
    try:
        model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("Không tìm thấy model đã huấn luyện. Vui lòng chạy customer_segmentation.py trước.")
        return None, None

def predict_customer_segment(customer_data):
    """Predict the segment for a new customer"""
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return None
    
    # Chuẩn hóa dữ liệu khách hàng mới
    features = ['age', 'income', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']
    customer_features = np.array([customer_data[feature] for feature in features]).reshape(1, -1)
    customer_scaled = scaler.transform(customer_features)
    
    # Dự đoán nhóm
    segment = model.predict(customer_scaled)[0]
    
    return segment

def get_segment_description(segment):
    """Get description of the segment based on the cluster analysis"""
    descriptions = {
        0: "Khách hàng có thu nhập trung bình, chi tiêu vừa phải",
        1: "Khách hàng có thu nhập cao, chi tiêu cao",
        2: "Khách hàng có thu nhập thấp, chi tiêu thấp",
        3: "Khách hàng có thu nhập cao, chi tiêu thấp"
    }
    return descriptions.get(segment, "Không có mô tả cho nhóm này")

def main():
    print("Nhập thông tin khách hàng:")
    try:
        customer_data = {
            'age': float(input("Tuổi: ")),
            'income': float(input("Thu nhập: ")),
            'spending_score': float(input("Điểm chi tiêu (0-100): ")),
            'membership_years': float(input("Số năm thành viên: ")),
            'purchase_frequency': float(input("Tần suất mua hàng: ")),
            'last_purchase_amount': float(input("Giá trị giao dịch cuối cùng: "))
        }
        
        segment = predict_customer_segment(customer_data)
        if segment is not None:
            print(f"\nKhách hàng thuộc nhóm: {segment}")
            print(f"Mô tả nhóm: {get_segment_description(segment)}")
            
            # Hiển thị thông tin so sánh với trung bình của nhóm
            df = pd.read_csv('data/data_customer.csv')
            cluster_data = df[df['Cluster'] == segment]
            
            print("\nSo sánh với trung bình của nhóm:")
            for feature in customer_data.keys():
                group_mean = cluster_data[feature].mean()
                customer_value = customer_data[feature]
                diff_percent = ((customer_value - group_mean) / group_mean) * 100
                print(f"{feature}: {customer_value:.2f} (Nhóm trung bình: {group_mean:.2f}, Chênh lệch: {diff_percent:.1f}%)")
    
    except ValueError:
        print("Vui lòng nhập số hợp lệ cho tất cả các trường.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main() 