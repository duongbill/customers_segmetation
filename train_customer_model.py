# Import thư viện os để thao tác với hệ điều hành (tạo thư mục, đường dẫn file)
import os
# Import pandas để xử lý dữ liệu dạng bảng (DataFrame)
import pandas as pd
# Import numpy để tính toán số học và xử lý mảng
import numpy as np
# Import matplotlib để vẽ biểu đồ và trực quan hóa dữ liệu
import matplotlib.pyplot as plt
# Import các công cụ tiền xử lý từ scikit-learn
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Import thuật toán phân cụm K-Means
from sklearn.cluster import KMeans
# Import thuật toán Random Forest để phân loại
from sklearn.ensemble import RandomForestClassifier
# Import metric để đánh giá chất lượng phân cụm
from sklearn.metrics import silhouette_score
# Import joblib để lưu và load các mô hình machine learning
from joblib import dump

# Định nghĩa các tham số cấu hình và đường dẫn file
DATA_PATH = 'data/data_clean.csv'  # Đường dẫn đến file dữ liệu đã được làm sạch
MODEL_DIR = 'models'  # Thư mục để lưu trữ các mô hình đã huấn luyện
RESULTS_PATH = 'models/customer_clusters.csv'  # Đường dẫn file lưu kết quả phân cụm
N_CLUSTERS = 4  # Số lượng cụm khách hàng mong muốn (có thể điều chỉnh)
RF_TREES = 100  # Số lượng cây quyết định trong mô hình Random Forest
SAVE_PLOT = True  # Biến boolean quyết định có lưu biểu đồ trực quan hay không

# Tạo thư mục models nếu chưa tồn tại (exist_ok=True để không báo lỗi nếu đã có)
os.makedirs(MODEL_DIR, exist_ok=True)

def train_customer_model():
    """
    Train mô hình phân cụm khách hàng và lưu các thành phần cần thiết

    Quy trình:
    1. Đọc dữ liệu
    2. Tiền xử lý dữ liệu (mã hóa biến phân loại)
    3. Chọn đặc trưng (features) cho mô hình
    4. Chuẩn hóa dữ liệu
    5. Huấn luyện mô hình KMeans để phân cụm
    6. Huấn luyện mô hình RandomForest để dự đoán cụm
    7. Phân tích thông tin các cụm
    8. Trực quan hóa kết quả phân cụm
    9. Lưu kết quả
    """
    # In thông báo bắt đầu quá trình huấn luyện mô hình
    print("=== BẮT ĐẦU TRAIN MÔ HÌNH PHÂN CỤM KHÁCH HÀNG ===")

    # BƯỚC 1: Đọc dữ liệu từ file CSV
    print("Đọc dữ liệu từ", DATA_PATH)  # In thông báo đường dẫn file
    df = pd.read_csv(DATA_PATH)  # Đọc dữ liệu từ file CSV vào DataFrame
    print(f"Đã đọc {len(df)} bản ghi")  # In số lượng bản ghi đã đọc được

    # BƯỚC 2: Tiền xử lý dữ liệu - Mã hóa các biến phân loại thành dạng số
    print("Chuẩn bị dữ liệu...")  # In thông báo bắt đầu chuẩn bị dữ liệu

    # Tạo các bộ mã hóa (encoder) cho các biến phân loại
    gender_enc = LabelEncoder()  # Bộ mã hóa cho giới tính (Male/Female/Other -> 0/1/2)
    category_enc = LabelEncoder()  # Bộ mã hóa cho danh mục sản phẩm ưa thích

    # Áp dụng bộ mã hóa để chuyển đổi dữ liệu text thành số
    df['gender_encoded'] = gender_enc.fit_transform(df['gender'])  # Mã hóa cột giới tính
    df['category_encoded'] = category_enc.fit_transform(df['preferred_category'])  # Mã hóa cột danh mục

    # Lưu các bộ mã hóa vào file để sử dụng khi dự đoán cho khách hàng mới
    dump(gender_enc, os.path.join(MODEL_DIR, 'gender_encoder.joblib'))  # Lưu encoder giới tính
    dump(category_enc, os.path.join(MODEL_DIR, 'category_encoder.joblib'))  # Lưu encoder danh mục

    # BƯỚC 3: Chọn các đặc trưng (features) quan trọng cho việc phân cụm
    features = [
        'age',                  # Tuổi khách hàng (số nguyên)
        'spending_score',       # Điểm chi tiêu từ 0-100 (số thực)
        'membership_years',     # Số năm là thành viên (số nguyên)
        'purchase_frequency',   # Tần suất mua hàng trong năm (số nguyên)
        'last_purchase_amount', # Số tiền mua hàng gần nhất (số thực)
        'gender_encoded',       # Giới tính đã được mã hóa thành số (0/1/2)
        'category_encoded'      # Danh mục ưa thích đã được mã hóa thành số
    ]

    # Lưu danh sách các đặc trưng để sử dụng khi dự đoán
    dump(features, os.path.join(MODEL_DIR, 'features.joblib'))

    # BƯỚC 4: Chuẩn hóa dữ liệu để đưa tất cả đặc trưng về cùng thang đo
    X = df[features]  # Trích xuất ma trận đặc trưng từ DataFrame
    scaler = StandardScaler()  # Tạo bộ chuẩn hóa (mean=0, std=1)
    X_scaled = scaler.fit_transform(X)  # Học và áp dụng chuẩn hóa cho dữ liệu
    dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))  # Lưu bộ chuẩn hóa để dùng sau

    # BƯỚC 5: Huấn luyện mô hình K-Means để phân cụm khách hàng
    print(f"Training KMeans với {N_CLUSTERS} clusters...")  # In thông báo bắt đầu huấn luyện
    # Tạo mô hình K-Means với các tham số:
    # - n_clusters: số cụm mong muốn
    # - random_state: seed để kết quả có thể tái tạo
    # - n_init: số lần chạy thuật toán với centroid khởi tạo khác nhau
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_scaled)  # Huấn luyện mô hình trên dữ liệu đã chuẩn hóa
    df['Cluster'] = kmeans.predict(X_scaled)  # Dự đoán và gán nhãn cụm cho từng khách hàng

    # Đánh giá chất lượng phân cụm bằng chỉ số Silhouette Score
    # Silhouette Score có giá trị từ -1 đến 1:
    # - Gần 1: phân cụm tốt (các điểm trong cụm gần nhau, các cụm xa nhau)
    # - Gần 0: phân cụm trung bình
    # - Gần -1: phân cụm kém
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score: {silhouette_avg:.4f}")  # In điểm đánh giá với 4 chữ số thập phân

    # Lưu mô hình K-Means đã huấn luyện vào file
    dump(kmeans, os.path.join(MODEL_DIR, 'kmeans.joblib'))

    # BƯỚC 6: Huấn luyện mô hình Random Forest để dự đoán cụm cho khách hàng mới
    # Random Forest được sử dụng để dự đoán cụm mà không cần chuẩn hóa dữ liệu mỗi lần
    print(f"Training RandomForest với {RF_TREES} cây...")  # In thông báo bắt đầu huấn luyện
    # Tạo mô hình Random Forest với:
    # - n_estimators: số cây quyết định trong rừng
    # - random_state: seed để kết quả có thể tái tạo
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=42)
    rf.fit(X, df['Cluster'])  # Huấn luyện mô hình trên dữ liệu gốc (chưa chuẩn hóa)

    # Lưu mô hình Random Forest đã huấn luyện vào file
    dump(rf, os.path.join(MODEL_DIR, 'rf_classifier.joblib'))

    # BƯỚC 7: Tính toán thông tin thống kê về các cụm khách hàng
    print("Tính toán thông tin về các cluster...")  # In thông báo bắt đầu phân tích

    # Tính toán thống kê cơ bản cho từng cụm (trung bình, min, max, số lượng)
    cluster_stats = df.groupby('Cluster').agg({
        'age': ['mean', 'min', 'max', 'count'],  # Thống kê về tuổi: TB, min, max, số lượng
        'spending_score': ['mean', 'min', 'max'],  # Thống kê về điểm chi tiêu: TB, min, max
        'membership_years': ['mean', 'min', 'max'],  # Thống kê về năm thành viên: TB, min, max
        'purchase_frequency': ['mean', 'min', 'max'],  # Thống kê về tần suất mua: TB, min, max
        'last_purchase_amount': ['mean', 'min', 'max']  # Thống kê về số tiền mua: TB, min, max
    }).round(2)  # Làm tròn kết quả đến 2 chữ số thập phân

    # Tạo mô tả chi tiết cho từng cụm dựa trên đặc điểm của cụm đó
    cluster_descriptions = {}  # Dictionary để lưu mô tả của từng cụm
    # Duyệt qua từng cụm để phân tích và tạo mô tả
    for cluster in range(N_CLUSTERS):
        # Lọc dữ liệu của cụm hiện tại
        cluster_data = df[df['Cluster'] == cluster]
        # Tính các giá trị trung bình của cụm để phân loại
        avg_spending = cluster_data['spending_score'].mean()  # Điểm chi tiêu trung bình
        avg_age = cluster_data['age'].mean()  # Tuổi trung bình
        avg_frequency = cluster_data['purchase_frequency'].mean()  # Tần suất mua hàng trung bình
        avg_membership = cluster_data['membership_years'].mean()  # Số năm thành viên trung bình (dự phòng)
        avg_purchase = cluster_data['last_purchase_amount'].mean()  # Số tiền mua hàng trung bình (dự phòng)

        # Phân loại cụm dựa trên độ tuổi, điểm chi tiêu và tần suất mua hàng
        # Sử dụng các ngưỡng để xác định đặc điểm của từng cụm khách hàng
        if avg_age <= 40 and avg_spending < 50 and avg_frequency > 30:
            # Khách hàng trẻ, chi tiêu thấp nhưng mua hàng thường xuyên -> có tiềm năng
            description = "Khách hàng trẻ - tiềm năng cao"
        elif avg_age <= 40 and avg_spending < 50:
            # Khách hàng trẻ, chi tiêu thấp, mua hàng ít -> tiết kiệm
            description = "Khách hàng trẻ tiết kiệm"
        elif avg_age > 40 and avg_age <= 50 and avg_spending >= 50 and avg_frequency > 30:
            # Khách hàng trung niên, chi tiêu cao, mua hàng thường xuyên -> khách hàng VIP
            description = "Khách hàng trung niên chi tiêu cao"
        elif avg_age > 50 and avg_spending >= 50:
            # Khách hàng lớn tuổi, chi tiêu cao -> khách hàng có giá trị
            description = "Khách hàng lớn tuổi chi tiêu cao"
        elif avg_frequency < 20:
            # Khách hàng mua hàng ít -> cần kích thích
            description = "Khách hàng tần suất mua hàng ít, chi tiêu thấp"
        else:
            # Trường hợp không thuộc các nhóm trên -> mô tả tổng quát
            description = f"Khách hàng {int(avg_age)} tuổi - Chi tiêu {int(avg_spending)} điểm"

        # Xử lý trường hợp các cụm có mô tả trùng lặp
        # Lấy danh sách các mô tả đã được sử dụng
        used_descriptions = [desc for desc in cluster_descriptions.values()]
        # Kiểm tra nếu mô tả hiện tại đã tồn tại
        if description in used_descriptions:
            # Nếu mô tả đã tồn tại, thêm thông tin chi tiết để phân biệt
            if "trung niên chi tiêu cao" in description:
                # Phân biệt dựa trên tần suất mua hàng
                if avg_frequency > 30:
                    description = "Khách hàng trung niên chi tiêu cao - tần suất mua hàng lớn"
                else:
                    description = "Khách hàng trung niên chi tiêu cao"
            elif "lớn tuổi chi tiêu cao" in description:
                # Giữ nguyên mô tả cho khách hàng lớn tuổi
                description = "Khách hàng lớn tuổi chi tiêu cao"
            elif "trẻ - tiềm năng cao" in description:
                # Giữ nguyên mô tả cho khách hàng trẻ tiềm năng cao
                description = "Khách hàng trẻ - tiềm năng cao"
            elif "trẻ tiết kiệm" in description:
                # Giữ nguyên mô tả cho khách hàng trẻ tiết kiệm
                description = "Khách hàng trẻ tiết kiệm"
            elif "tần suất mua hàng ít, chi tiêu thấp" in description:
                # Giữ nguyên mô tả cho khách hàng mua hàng ít
                description = "Khách hàng tần suất mua hàng ít, chi tiêu thấp"
            else:
                # Thêm thông tin về tần suất mua hàng cho các trường hợp khác
                if avg_frequency > 30:
                    description += " - tần suất mua hàng lớn"
                elif avg_frequency < 20:
                    description += " - tần suất mua hàng ít"
        else:
            # Nếu mô tả chưa tồn tại, kiểm tra trường hợp đặc biệt
            if "trung niên chi tiêu cao" in description:
                # Thêm chi tiết về tần suất mua hàng nếu cao
                if avg_frequency > 30:
                    description = "Khách hàng trung niên chi tiêu cao - tần suất mua hàng lớn"

        # Lưu mô tả vào dictionary với key là số hiệu cụm
        cluster_descriptions[cluster] = description

    # Lưu thông tin thống kê và mô tả các cụm vào file
    dump((cluster_stats, cluster_descriptions), os.path.join(MODEL_DIR, 'cluster_info.joblib'))

    # BƯỚC 8: Vẽ biểu đồ phân cụm để trực quan hóa kết quả
    if SAVE_PLOT:  # Chỉ vẽ biểu đồ nếu SAVE_PLOT = True
        print("Vẽ biểu đồ phân cụm...")  # In thông báo bắt đầu vẽ biểu đồ
        plt.figure(figsize=(12, 8))  # Tạo figure với kích thước 12x8 inch

        # Vẽ scatter plot với màu sắc tương ứng với từng cụm
        scatter = plt.scatter(df['age'], df['spending_score'],  # Trục x: tuổi, trục y: điểm chi tiêu
                             c=df['Cluster'],  # Màu sắc theo nhãn cụm
                             cmap='viridis',  # Bảng màu viridis (xanh-vàng)
                             s=100,  # Kích thước điểm = 100
                             alpha=0.6)  # Độ trong suốt 60%

        # Thêm centroids (tâm của các cụm) vào biểu đồ
        # Chuyển đổi centroids từ dữ liệu chuẩn hóa về dữ liệu gốc
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(centroids[:, 0], centroids[:, 1],  # Tọa độ x, y của centroids (tuổi, điểm chi tiêu)
                   c='red',  # Màu đỏ nổi bật cho centroids
                   marker='x',  # Sử dụng dấu X
                   s=200,  # Kích thước lớn hơn các điểm dữ liệu
                   linewidths=3,  # Độ dày đường viền = 3
                   label='Centroids')  # Nhãn hiển thị trong legend

        # Thêm tiêu đề và nhãn cho các trục
        plt.title('Phân cụm khách hàng theo tuổi và điểm chi tiêu', fontsize=14)  # Tiêu đề với font size 14
        plt.xlabel('Tuổi', fontsize=12)  # Nhãn trục x với font size 12
        plt.ylabel('Điểm chi tiêu', fontsize=12)  # Nhãn trục y với font size 12
        plt.colorbar(scatter, label='Cluster')  # Thêm thanh màu với nhãn 'Cluster'
        plt.legend()  # Hiển thị chú thích (legend)
        plt.grid(True, alpha=0.3)  # Thêm lưới với độ trong suốt 30%
        plt.savefig(os.path.join(MODEL_DIR, 'cluster_visualization.png'))  # Lưu biểu đồ vào file PNG
        plt.close()  # Đóng figure để giải phóng bộ nhớ

    # BƯỚC 9: In thông tin chi tiết về các cụm ra console
    print("\nThông tin chi tiết về các cluster:")  # In tiêu đề với dòng trống phía trước
    # Duyệt qua từng cụm và in thông tin chi tiết
    for cluster, description in cluster_descriptions.items():
        # Lấy các thông tin thống kê từ cluster_stats
        count = cluster_stats.loc[cluster][('age', 'count')]  # Số lượng khách hàng trong cụm
        avg_age = cluster_stats.loc[cluster][('age', 'mean')]  # Tuổi trung bình
        avg_spending = cluster_stats.loc[cluster][('spending_score', 'mean')]  # Điểm chi tiêu trung bình
        # In thông tin của từng cụm
        print(f"Cluster {cluster}: {description}")  # In số hiệu và mô tả cụm
        print(f"  - Số lượng khách hàng: {count}")  # In số lượng khách hàng
        print(f"  - Tuổi trung bình: {avg_age}")  # In tuổi trung bình
        print(f"  - Điểm chi tiêu trung bình: {avg_spending}")  # In điểm chi tiêu trung bình
        print()  # In dòng trống để phân cách các cụm

    # BƯỚC 10: Thêm mô tả cụm vào DataFrame gốc
    df['Cluster_Description'] = df['Cluster'].map(cluster_descriptions)  # Map số cụm thành mô tả

    # BƯỚC 11: Lưu kết quả phân cụm vào file CSV
    print(f"Lưu kết quả phân cụm vào {RESULTS_PATH}...")  # In thông báo bắt đầu lưu file

    # Tạo DataFrame kết quả chỉ chứa các cột quan trọng
    results_df = df[['age', 'gender', 'spending_score', 'membership_years',
                     'purchase_frequency', 'preferred_category', 'last_purchase_amount',
                     'Cluster', 'Cluster_Description']]  # Chọn các cột cần thiết

    # Lưu DataFrame kết quả vào file CSV
    results_df.to_csv(RESULTS_PATH, index=False)  # index=False để không lưu chỉ số dòng
    print(f"Đã lưu kết quả phân cụm cho {len(results_df)} khách hàng")  # In thông báo hoàn thành

    # BƯỚC 12: Lưu kết quả thống kê chi tiết dưới dạng các file CSV
    stats_dir = 'data/cluster_statistics'  # Thư mục lưu các file thống kê
    os.makedirs(stats_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    # Tạo và lưu file thống kê tổng quan
    overview_df = pd.DataFrame({
        'Cluster': list(cluster_descriptions.keys()),  # Danh sách số hiệu cụm
        'Description': list(cluster_descriptions.values()),  # Danh sách mô tả cụm
        'Count': [cluster_stats.loc[c][('age', 'count')] for c in range(N_CLUSTERS)]  # Số lượng khách hàng mỗi cụm
    })
    overview_df.to_csv(f"{stats_dir}/overview.csv", index=False)  # Lưu file tổng quan
    print(f"Đã lưu thống kê tổng quan vào {stats_dir}/overview.csv")

    # Lưu thống kê chi tiết cho từng đặc trưng
    for feature in ['age', 'spending_score', 'membership_years', 'purchase_frequency', 'last_purchase_amount']:
        feature_stats = pd.DataFrame()  # Tạo DataFrame mới cho mỗi đặc trưng
        # Duyệt qua các loại thống kê (trung bình, min, max)
        for stat in ['mean', 'min', 'max']:
            # Kiểm tra nếu thống kê này tồn tại cho đặc trưng hiện tại
            if stat in cluster_stats[feature].columns:
                # Lấy giá trị thống kê cho tất cả các cụm
                feature_stats[stat] = [cluster_stats.loc[c][(feature, stat)] for c in range(N_CLUSTERS)]
        # Thêm thông tin cụm và mô tả
        feature_stats['Cluster'] = list(range(N_CLUSTERS))  # Số hiệu cụm
        feature_stats['Description'] = list(cluster_descriptions.values())  # Mô tả cụm
        # Lưu file thống kê cho đặc trưng này
        feature_stats.to_csv(f"{stats_dir}/{feature.capitalize()}.csv", index=False)
        print(f"Đã lưu thống kê chi tiết cho {feature} vào {stats_dir}/{feature.capitalize()}.csv")

    print(f"Đã lưu thống kê chi tiết vào thư mục {stats_dir}")  # In thông báo hoàn thành

    # In thông báo kết thúc quá trình huấn luyện
    print("=== HOÀN THÀNH TRAIN MÔ HÌNH ===")
    print(f"Các model đã được lưu trong thư mục: {os.path.abspath(MODEL_DIR)}")  # In đường dẫn tuyệt đối

    return df  # Trả về DataFrame đã được phân cụm

# Kiểm tra nếu file này được chạy trực tiếp (không phải import)
if __name__ == "__main__":
    train_customer_model()  # Gọi hàm huấn luyện mô hình







