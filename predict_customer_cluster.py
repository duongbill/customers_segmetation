# Import thư viện os để thao tác với hệ điều hành (đường dẫn file, folder)
import os
# Import pandas để xử lý dữ liệu dạng bảng
import pandas as pd
# Import numpy để tính toán số học và xử lý mảng
import numpy as np
# Import joblib để load các mô hình machine learning đã được lưu
from joblib import load

# Định nghĩa đường dẫn đến thư mục chứa các mô hình đã được huấn luyện
MODEL_DIR = 'models'

def load_models():
    """
    Hàm load tất cả các mô hình và thành phần cần thiết từ thư mục models

    Trả về:
    - dict: Dictionary chứa tất cả các mô hình và encoder đã được load
    """
    # In thông báo bắt đầu quá trình load
    print("Đang load các model...")

    # Load các encoder để mã hóa dữ liệu phân loại
    # Load encoder cho giới tính (Male/Female/Other -> số)
    gender_enc = load(os.path.join(MODEL_DIR, 'gender_encoder.joblib'))
    # Load encoder cho danh mục sản phẩm ưa thích (Electronics/Clothing/... -> số)
    category_enc = load(os.path.join(MODEL_DIR, 'category_encoder.joblib'))

    # Load scaler để chuẩn hóa dữ liệu số (đưa về cùng thang đo)
    scaler = load(os.path.join(MODEL_DIR, 'scaler.joblib'))

    # Load các mô hình machine learning
    # Load mô hình K-Means để phân cụm
    kmeans = load(os.path.join(MODEL_DIR, 'kmeans.joblib'))
    # Load mô hình Random Forest để phân loại/dự đoán cụm
    rf = load(os.path.join(MODEL_DIR, 'rf_classifier.joblib'))

    # Load danh sách các đặc trưng (features) được sử dụng trong mô hình
    features = load(os.path.join(MODEL_DIR, 'features.joblib'))
    # Load thống kê và mô tả của từng cụm khách hàng
    cluster_stats, cluster_descriptions = load(os.path.join(MODEL_DIR, 'cluster_info.joblib'))

    # In thông báo hoàn thành quá trình load
    print("Đã load xong các model")

    # Trả về dictionary chứa tất cả các thành phần đã load
    return {
        'gender_encoder': gender_enc,        # Encoder cho giới tính
        'category_encoder': category_enc,    # Encoder cho danh mục sản phẩm
        'scaler': scaler,                   # Scaler để chuẩn hóa dữ liệu
        'kmeans': kmeans,                   # Mô hình K-Means
        'rf_classifier': rf,                # Mô hình Random Forest
        'features': features,               # Danh sách các đặc trưng
        'cluster_stats': cluster_stats,     # Thống kê của từng cụm
        'cluster_descriptions': cluster_descriptions  # Mô tả của từng cụm
    }

def predict_customer_cluster(customer_data, models):
    """
    Hàm dự đoán cụm khách hàng cho một khách hàng mới

    Tham số:
    - customer_data: dict chứa thông tin khách hàng (tuổi, giới tính, điểm chi tiêu, v.v.)
    - models: dict chứa các mô hình và encoder đã được load

    Trả về:
    - dict chứa kết quả dự đoán (cụm, mô tả, thống kê) hoặc None nếu có lỗi
    """
    # Bắt đầu quá trình chuẩn bị dữ liệu và dự đoán
    try:
        # Mã hóa các biến phân loại thành số để mô hình có thể xử lý
        # Kiểm tra và mã hóa giới tính nếu có trong dữ liệu
        if 'gender' in customer_data:
            # Chuyển đổi giới tính từ text (Male/Female/Other) thành số
            customer_data['gender_encoded'] = models['gender_encoder'].transform([customer_data['gender']])[0]

        # Kiểm tra và mã hóa danh mục sản phẩm ưa thích nếu có trong dữ liệu
        if 'preferred_category' in customer_data:
            # Chuyển đổi danh mục từ text (Electronics/Clothing/...) thành số
            customer_data['category_encoded'] = models['category_encoder'].transform([customer_data['preferred_category']])[0]

        # Tạo vector đặc trưng từ dữ liệu khách hàng
        features = models['features']  # Lấy danh sách các đặc trưng cần thiết
        # Tạo mảng numpy 2D chứa giá trị của từng đặc trưng (dùng 0 nếu thiếu dữ liệu)
        X = np.array([[customer_data.get(f, 0) for f in features]])

        # Chuẩn hóa dữ liệu để đưa về cùng thang đo (tùy chọn, có thể không dùng)
        X_scaled = models['scaler'].transform(X)

        # Sử dụng mô hình Random Forest để dự đoán cụm
        # Lấy kết quả dự đoán đầu tiên (vì chỉ có 1 khách hàng)
        cluster = models['rf_classifier'].predict(X)[0]

        # Lấy thông tin chi tiết về cụm đã dự đoán
        description = models['cluster_descriptions'][cluster]  # Mô tả cụm
        stats = models['cluster_stats'].loc[cluster]  # Thống kê của cụm

        # Trả về kết quả dự đoán dưới dạng dictionary
        return {
            'cluster': int(cluster),      # Số hiệu cụm (chuyển về int)
            'description': description,   # Mô tả chi tiết cụm
            'stats': stats               # Thống kê của cụm (tuổi TB, chi tiêu TB, v.v.)
        }

    # Xử lý lỗi nếu có vấn đề trong quá trình dự đoán
    except Exception as e:
        print(f"Lỗi khi dự đoán cluster: {str(e)}")  # In thông báo lỗi
        return None  # Trả về None để báo hiệu có lỗi

def analyze_customer_file(file_path, models):
    """
    Hàm phân tích file CSV chứa nhiều khách hàng và dự đoán cụm cho từng người

    Tham số:
    - file_path: đường dẫn đến file CSV chứa dữ liệu khách hàng
    - models: dict chứa các mô hình và encoder đã được load

    Trả về:
    - DataFrame chứa dữ liệu khách hàng gốc + cột Cluster và Cluster_Description
    - None nếu có lỗi xảy ra
    """
    # Bắt đầu quá trình phân tích file
    try:
        # Đọc dữ liệu từ file CSV
        df = pd.read_csv(file_path)
        # In thông báo số lượng khách hàng đã đọc được
        print(f"Đã đọc {len(df)} khách hàng từ file {file_path}")

        # Mã hóa các biến phân loại thành số để mô hình có thể xử lý
        # Kiểm tra và mã hóa cột giới tính nếu tồn tại
        if 'gender' in df.columns:
            # Chuyển đổi tất cả giá trị giới tính từ text thành số
            df['gender_encoded'] = models['gender_encoder'].transform(df['gender'])

        # Kiểm tra và mã hóa cột danh mục sản phẩm ưa thích nếu tồn tại
        if 'preferred_category' in df.columns:
            # Chuyển đổi tất cả giá trị danh mục từ text thành số
            df['category_encoded'] = models['category_encoder'].transform(df['preferred_category'])

        # Chuẩn bị dữ liệu đặc trưng cho việc dự đoán
        features = models['features']  # Lấy danh sách các đặc trưng cần thiết
        # Trích xuất giá trị của các đặc trưng thành mảng numpy 2D
        X = df[features].values

        # Sử dụng mô hình Random Forest để dự đoán cụm cho tất cả khách hàng
        df['Cluster'] = models['rf_classifier'].predict(X)

        # Thêm cột mô tả cụm bằng cách map từ số cụm sang mô tả
        df['Cluster_Description'] = df['Cluster'].map(models['cluster_descriptions'])

        # Trả về DataFrame đã được bổ sung thông tin cụm
        return df

    # Xử lý lỗi nếu có vấn đề trong quá trình phân tích
    except Exception as e:
        print(f"Lỗi khi phân tích file: {str(e)}")  # In thông báo lỗi
        return None  # Trả về None để báo hiệu có lỗi

def main():
    """
    Hàm chính để demo và test các chức năng của module
    """
    # Load tất cả các mô hình cần thiết
    models = load_models()

    # Tạo dữ liệu mẫu: thông tin của một khách hàng mới để test
    new_customer = {
        'age': 30,                          # Tuổi: 30
        'gender': 'Female',                 # Giới tính: Nữ
        'spending_score': 75,               # Điểm chi tiêu: 75/100
        'membership_years': 3,              # Số năm thành viên: 3 năm
        'purchase_frequency': 25,           # Tần suất mua hàng: 25 lần/năm
        'preferred_category': 'Clothing',   # Danh mục ưa thích: Quần áo
        'last_purchase_amount': 350.0       # Số tiền mua hàng gần nhất: $350
    }

    # Gọi hàm dự đoán cụm cho khách hàng mẫu
    result = predict_customer_cluster(new_customer, models)

    # Kiểm tra và hiển thị kết quả dự đoán
    if result:
        print("\nKết quả dự đoán:")  # In tiêu đề kết quả
        print(f"Cluster: {result['cluster']}")  # In số hiệu cụm
        print(f"Mô tả: {result['description']}")  # In mô tả cụm
        print("\nThống kê về nhóm khách hàng này:")  # In tiêu đề thống kê
        # In số lượng khách hàng trong cụm
        print(f"- Số lượng khách hàng tương tự: {result['stats'][('age', 'count')]}")
        # In tuổi trung bình của cụm
        print(f"- Tuổi trung bình: {result['stats'][('age', 'mean')]}")
        # In điểm chi tiêu trung bình của cụm
        print(f"- Điểm chi tiêu trung bình: {result['stats'][('spending_score', 'mean')]}")
        # In tần suất mua hàng trung bình của cụm
        print(f"- Tần suất mua hàng trung bình: {result['stats'][('purchase_frequency', 'mean')]}")

    # Ví dụ về cách sử dụng hàm phân tích file (hiện tại bị comment)
    # Để sử dụng, bỏ comment các dòng dưới đây và chuẩn bị file CSV phù hợp
    # result_df = analyze_customer_file('data/new_customers.csv', models)
    # if result_df is not None:
    #     # Lưu kết quả phân tích vào file CSV mới
    #     result_df.to_csv('data/new_customers_with_clusters.csv', index=False)
    #     print(f"Đã lưu kết quả phân tích vào file 'data/new_customers_with_clusters.csv'")

# Kiểm tra nếu file này được chạy trực tiếp (không phải import)
if __name__ == "__main__":
    main()  # Gọi hàm main để chạy demo
