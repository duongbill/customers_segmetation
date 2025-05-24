# Import thư viện Streamlit để tạo giao diện web
import streamlit as st
# Import pandas để xử lý dữ liệu dạng bảng
import pandas as pd
# Import numpy để tính toán số học
import numpy as np
# Import os để thao tác với hệ điều hành (file, folder)
import os
# Import matplotlib để vẽ biểu đồ
import matplotlib.pyplot as plt
# Import seaborn để vẽ biểu đồ thống kê đẹp hơn
import seaborn as sns
# Import joblib để load các mô hình đã được lưu
from joblib import load
# Import plotly express để tạo biểu đồ tương tác
import plotly.express as px
# Import plotly graph objects để tạo biểu đồ phức tạp hơn
import plotly.graph_objects as go
# Import các hàm từ module predict_customer_cluster để xử lý phân cụm
from predict_customer_cluster import load_models, predict_customer_cluster, analyze_customer_file

# Cấu hình trang Streamlit với các thông số cơ bản
st.set_page_config(
    page_title="Phân cụm khách hàng",  # Tiêu đề hiển thị trên tab trình duyệt
    page_icon="👥",  # Icon hiển thị trên tab trình duyệt
    layout="centered",  # Bố cục trang ở giữa
    initial_sidebar_state="collapsed"  # Sidebar ban đầu được thu gọn
)

# CSS để tối ưu giao diện cho laptop 13 inch
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    .stSelectbox > div > div {
        font-size: 14px;
    }
    .stNumberInput > div > div > input {
        font-size: 14px;
    }
    h1 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 0.6rem !important;
    }
</style>
""", unsafe_allow_html=True)  # Cho phép HTML không an toàn để áp dụng CSS

# Hiển thị tiêu đề chính của ứng dụng
st.title("🔍 Phân cụm khách hàng")
# Hiển thị mô tả ngắn về chức năng của ứng dụng
st.markdown("Phân loại khách hàng thành các nhóm có đặc điểm tương tự.")

# Tạo hàm load models với cache để tối ưu hiệu suất
@st.cache_resource  # Decorator để cache kết quả, tránh load lại mỗi lần refresh
def get_models():
    """Cache models để tránh load lại mỗi lần refresh trang"""
    return load_models()  # Gọi hàm load_models từ module predict_customer_cluster

# Tạo header cho sidebar
st.sidebar.header("Chọn chức năng")
# Tạo radio button để người dùng chọn chức năng
option = st.sidebar.radio(
    "Bạn muốn làm gì?",  # Câu hỏi hiển thị
    ["Dự đoán cho khách hàng mới", "Phân tích file khách hàng", "Thông tin về các cụm"]  # Các lựa chọn
)

# Thử load các mô hình và xử lý lỗi nếu có
try:
    models = get_models()  # Gọi hàm get_models để load các mô hình
    st.sidebar.success("Đã tải xong các mô hình!")  # Hiển thị thông báo thành công
except Exception as e:  # Bắt lỗi nếu có
    st.sidebar.error(f"Lỗi khi tải mô hình: {str(e)}")  # Hiển thị thông báo lỗi
    st.stop()  # Dừng ứng dụng nếu không load được mô hình

# Kiểm tra nếu người dùng chọn chức năng "Thông tin về các cụm"
if option == "Thông tin về các cụm":
    # Hiển thị tiêu đề chính cho phần thông tin cụm
    st.header("Thông tin về các cụm khách hàng")

    # Lấy thông tin thống kê và mô tả các cụm từ models đã load
    cluster_stats = models['cluster_stats']  # Thống kê số liệu của từng cụm
    cluster_descriptions = models['cluster_descriptions']  # Mô tả chi tiết từng cụm

    # Hiển thị phần mô tả các cụm
    st.subheader("Mô tả các cụm")
    # Duyệt qua từng cụm và hiển thị mô tả
    for cluster, description in cluster_descriptions.items():
        st.markdown(f"**Cụm {cluster}**: {description}")  # Hiển thị tên cụm và mô tả

    # Hiển thị phần thống kê số liệu các cụm
    st.subheader("Thống kê về các cụm")

    # Tạo DataFrame rỗng để chứa thống kê tổng hợp
    stats_df = pd.DataFrame()
    # Duyệt qua từng cụm để lấy thống kê chi tiết
    for cluster in cluster_descriptions.keys():
        # Tạo dictionary chứa thông tin thống kê của từng cụm
        cluster_data = {
            'Cụm': cluster,  # Tên cụm
            'Mô tả': cluster_descriptions[cluster],  # Mô tả cụm
            'Số lượng khách hàng': cluster_stats.loc[cluster][('age', 'count')],  # Số lượng khách hàng trong cụm
            'Tuổi trung bình': cluster_stats.loc[cluster][('age', 'mean')],  # Tuổi trung bình
            'Điểm chi tiêu trung bình': cluster_stats.loc[cluster][('spending_score', 'mean')],  # Điểm chi tiêu TB
            'Năm thành viên trung bình': cluster_stats.loc[cluster][('membership_years', 'mean')],  # Số năm thành viên TB
            'Tần suất mua hàng trung bình': cluster_stats.loc[cluster][('purchase_frequency', 'mean')],  # Tần suất mua TB
            'Số tiền mua hàng trung bình': cluster_stats.loc[cluster][('last_purchase_amount', 'mean')]  # Giá trị đơn hàng TB
        }
        # Thêm dữ liệu cụm vào DataFrame tổng hợp
        stats_df = pd.concat([stats_df, pd.DataFrame([cluster_data])], ignore_index=True)

    # Hiển thị bảng thống kê dưới dạng dataframe tương tác
    st.dataframe(stats_df)

    # Tạo phần so sánh các cụm bằng biểu đồ
    st.subheader("So sánh các cụm")

    # Tạo widget multiselect để người dùng chọn đặc trưng muốn so sánh
    features_to_compare = st.multiselect(
        "Chọn các đặc trưng để so sánh",  # Label của widget
        ['Tuổi trung bình', 'Điểm chi tiêu trung bình', 'Năm thành viên trung bình',
         'Tần suất mua hàng trung bình', 'Số tiền mua hàng trung bình'],  # Danh sách các lựa chọn
        default=['Tuổi trung bình', 'Điểm chi tiêu trung bình']  # Giá trị mặc định được chọn
    )

    # Kiểm tra nếu người dùng đã chọn ít nhất một đặc trưng
    if features_to_compare:
        # Tạo biểu đồ cột nhóm để so sánh các đặc trưng giữa các cụm
        fig = px.bar(
            stats_df,  # Dữ liệu nguồn
            x='Cụm',  # Trục x hiển thị tên cụm
            y=features_to_compare,  # Trục y hiển thị các đặc trưng được chọn
            barmode='group',  # Chế độ hiển thị cột nhóm
            title="So sánh các đặc trưng giữa các cụm",  # Tiêu đề biểu đồ
            color_discrete_sequence=px.colors.qualitative.Plotly  # Bảng màu cho biểu đồ
        )
        # Hiển thị biểu đồ với chiều rộng tự động theo container
        st.plotly_chart(fig, use_container_width=True)

    # Tạo biểu đồ tròn để hiển thị phân bố khách hàng theo cụm
    fig_pie = px.pie(
        stats_df,  # Dữ liệu nguồn
        values='Số lượng khách hàng',  # Giá trị để tính tỷ lệ
        names='Mô tả',  # Nhãn hiển thị trên biểu đồ
        title="Phân bố khách hàng theo cụm"  # Tiêu đề biểu đồ
    )
    # Hiển thị biểu đồ tròn với chiều rộng tự động theo container
    st.plotly_chart(fig_pie, use_container_width=True)

# Kiểm tra nếu người dùng chọn chức năng "Dự đoán cho khách hàng mới"
elif option == "Dự đoán cho khách hàng mới":
    # Hiển thị tiêu đề chính cho phần dự đoán
    st.header("Dự đoán cụm khách hàng")

    # Hiển thị tiêu đề phụ cho phần nhập thông tin
    st.subheader("Thông tin khách hàng")

    # Tạo 2 cột để bố trí các input field một cách gọn gàng
    col1, col2 = st.columns(2)

    # Cột 1: Chứa các input field đầu tiên
    with col1:
        # Input số cho tuổi khách hàng
        age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
        # Selectbox cho giới tính
        gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
        # Slider cho điểm chi tiêu
        spending_score = st.slider("Điểm chi tiêu (0-100)", 0, 100, 50)
        # Input số cho số năm thành viên
        membership_years = st.number_input("Số năm thành viên", min_value=0, max_value=50, value=2)

    # Cột 2: Chứa các input field còn lại
    with col2:
        # Input số cho tần suất mua hàng
        purchase_frequency = st.number_input("Tần suất mua hàng (số lần/năm)", min_value=0, max_value=365, value=12)
        # Selectbox cho danh mục sản phẩm ưa thích
        preferred_category = st.selectbox(
            "Danh mục ưa thích",
            ["Electronics", "Clothing", "Groceries", "Sports", "Home & Garden", "Beauty", "Books", "Other"]
        )
        # Input số cho số tiền mua hàng gần nhất
        last_purchase_amount = st.number_input("Số tiền mua hàng gần nhất ($)", min_value=0.0, value=100.0)

    # Tạo dictionary để lưu trữ tất cả thông tin khách hàng đã nhập
    customer_data = {
        'age': age,  # Tuổi
        'gender': gender,  # Giới tính
        'spending_score': spending_score,  # Điểm chi tiêu
        'membership_years': membership_years,  # Số năm thành viên
        'purchase_frequency': purchase_frequency,  # Tần suất mua hàng
        'preferred_category': preferred_category,  # Danh mục ưa thích
        'last_purchase_amount': last_purchase_amount  # Số tiền mua hàng gần nhất
    }

    # Tạo nút phân tích với kiểu primary (màu nổi bật)
    if st.button("🔍 Phân tích khách hàng", type="primary"):
        # Gọi hàm dự đoán cụm khi người dùng nhấn nút
        result = predict_customer_cluster(customer_data, models)
    else:
        # Nếu chưa nhấn nút thì result = None
        result = None

    # Kiểm tra nếu có kết quả dự đoán
    if result:
        # Hiển thị kết quả dự đoán với thông báo thành công (màu xanh)
        st.success(f"**Cụm {result['cluster']}**: {result['description']}")

        # Tạo 2 cột có kích thước bằng nhau để hiển thị thông tin
        col1, col2 = st.columns([1, 1])

        # Cột 1: Hiển thị thống kê của cụm
        with col1:
            st.subheader("Thống kê cụm")  # Tiêu đề phụ
            # Hiển thị số lượng khách hàng trong cụm
            st.write(f"**Số lượng:** {result['stats'][('age', 'count')]}")
            # Hiển thị tuổi trung bình với 1 chữ số thập phân
            st.write(f"**Tuổi TB:** {result['stats'][('age', 'mean')]:.1f}")
            # Hiển thị điểm chi tiêu trung bình với 1 chữ số thập phân
            st.write(f"**Chi tiêu TB:** {result['stats'][('spending_score', 'mean')]:.1f}")
            # Hiển thị tần suất mua hàng trung bình với 1 chữ số thập phân
            st.write(f"**Tần suất mua:** {result['stats'][('purchase_frequency', 'mean')]:.1f}")
            # Hiển thị giá trị đơn hàng trung bình làm tròn thành số nguyên
            st.write(f"**Giá trị đơn hàng:** ${result['stats'][('last_purchase_amount', 'mean')]:.0f}")

        # Cột 2: Hiển thị gợi ý marketing
        with col2:
            st.subheader("Gợi ý marketing")  # Tiêu đề phụ

            # Phân tích mô tả cụm để đưa ra gợi ý marketing phù hợp
            # Kiểm tra nếu là cụm khách hàng trung niên chi tiêu cao
            if "trung niên chi tiêu cao" in result['description'].lower():
                st.markdown("""
                **Chiến lược marketing:**
                - 🎁 Ưu đãi đặc biệt cho khách VIP
                - 🛎️ Dịch vụ chăm sóc ưu tiên
                - 💌 Thông báo sản phẩm mới
                - 🏆 Chương trình tích điểm cao cấp
                """)
            # Kiểm tra nếu là cụm khách hàng trẻ tiềm năng cao
            elif "trẻ - tiềm năng cao" in result['description'].lower():
                st.markdown("""
                **Chiến lược marketing:**
                - 📱 Marketing qua mạng xã hội
                - 🔄 Chương trình giới thiệu bạn bè
                - 🎮 Gamification mua sắm
                - 🌟 Ưu đãi sản phẩm xu hướng
                """)
            # Kiểm tra nếu là cụm khách hàng lớn tuổi chi tiêu cao
            elif "lớn tuổi chi tiêu cao" in result['description'].lower():
                st.markdown("""
                **Chiến lược marketing:**
                - 📞 Hỗ trợ qua điện thoại
                - 🏷️ Giảm giá sản phẩm cao cấp
                - 📆 Ưu đãi theo mùa
                - 🔍 Hướng dẫn chi tiết
                """)
            # Kiểm tra nếu là cụm khách hàng có tần suất mua hàng ít
            elif "tần suất mua hàng ít" in result['description'].lower():
                st.markdown("""
                **Chiến lược marketing:**
                - 📊 Khảo sát nhu cầu
                - 🎯 Marketing có mục tiêu
                - 🔔 Nhắc nhở sản phẩm
                - 🎁 Ưu đãi khuyến khích
                """)
            # Gợi ý marketing mặc định cho các cụm khác
            else:
                st.markdown("""
                **Chiến lược marketing:**
                - 📧 Email marketing định kỳ
                - 💰 Chương trình tích điểm
                - 🏷️ Mã giảm giá
                - 📱 Thông báo khuyến mãi
                """)

        # Tạo phần so sánh khách hàng với cụm bằng biểu đồ
        st.subheader("So sánh với cụm")

        # Tạo dữ liệu để so sánh khách hàng hiện tại với trung bình cụm
        comparison_data = {
            'Đặc trưng': ['Tuổi', 'Chi tiêu', 'Tần suất mua'],  # Các đặc trưng để so sánh
            'Khách hàng': [customer_data['age'], customer_data['spending_score'], customer_data['purchase_frequency']],  # Giá trị của khách hàng
            'Trung bình cụm': [result['stats'][('age', 'mean')], result['stats'][('spending_score', 'mean')], result['stats'][('purchase_frequency', 'mean')]]  # Giá trị trung bình của cụm
        }

        # Import pandas để tạo DataFrame (có thể trùng với import ở đầu file)
        import pandas as pd
        # Tạo DataFrame từ dữ liệu so sánh
        df_comparison = pd.DataFrame(comparison_data)

        # Tạo biểu đồ cột nhóm để so sánh
        fig = px.bar(
            df_comparison.melt(id_vars='Đặc trưng', var_name='Loại', value_name='Giá trị'),  # Chuyển đổi dữ liệu từ wide sang long format
            x='Đặc trưng',  # Trục x hiển thị các đặc trưng
            y='Giá trị',  # Trục y hiển thị giá trị
            color='Loại',  # Màu sắc phân biệt giữa khách hàng và trung bình cụm
            barmode='group',  # Chế độ hiển thị cột nhóm
            height=300  # Chiều cao biểu đồ
        )
        # Hiển thị biểu đồ với chiều rộng tự động theo container
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Hiển thị thông báo hướng dẫn khi người dùng chưa nhấn nút phân tích
        st.info("👆 Vui lòng nhập thông tin khách hàng và nhấn nút **'🔍 Phân tích khách hàng'** để xem kết quả phân cụm.")

# Kiểm tra nếu người dùng chọn chức năng "Phân tích file khách hàng"
elif option == "Phân tích file khách hàng":
    # Hiển thị tiêu đề chính cho phần phân tích file
    st.header("Phân tích file khách hàng")

    # Hiển thị hướng dẫn định dạng file cho người dùng
    st.info("""
    **Hướng dẫn:** Tải lên file CSV chứa dữ liệu khách hàng. File phải có các cột sau:
    - age: Tuổi khách hàng
    - gender: Giới tính (Male/Female/Other)
    - spending_score: Điểm chi tiêu (0-100)
    - membership_years: Số năm thành viên
    - purchase_frequency: Tần suất mua hàng
    - preferred_category: Danh mục ưa thích
    - last_purchase_amount: Số tiền mua hàng gần nhất
    """)

    # Tạo widget upload file chỉ chấp nhận file CSV
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    # Kiểm tra nếu người dùng đã upload file
    if uploaded_file is not None:
        # Tạo đường dẫn file tạm thời để lưu file upload
        temp_file_path = "temp_customer_data.csv"
        # Mở file tạm thời ở chế độ ghi binary
        with open(temp_file_path, "wb") as f:
            # Ghi nội dung file upload vào file tạm thời
            f.write(uploaded_file.getbuffer())

        # Đọc và hiển thị dữ liệu gốc từ file CSV
        df_original = pd.read_csv(temp_file_path)
        st.subheader("Dữ liệu khách hàng gốc")  # Tiêu đề phụ
        st.dataframe(df_original)  # Hiển thị dataframe tương tác

        # Tạo nút để bắt đầu quá trình phân tích
        if st.button("Phân tích và phân cụm"):
            # Hiển thị spinner trong quá trình xử lý
            with st.spinner("Đang phân tích dữ liệu..."):
                # Gọi hàm phân tích file khách hàng
                result_df = analyze_customer_file(temp_file_path, models)

                # Kiểm tra nếu phân tích thành công
                if result_df is not None:
                    # Tạo đường dẫn file để lưu kết quả
                    result_file_path = "customer_clusters_result.csv"
                    # Lưu kết quả phân cụm vào file CSV
                    result_df.to_csv(result_file_path, index=False)

                    # Hiển thị kết quả phân cụm
                    st.subheader("Kết quả phân cụm")
                    st.dataframe(result_df)  # Hiển thị dataframe kết quả

                    # Tạo phần hiển thị biểu đồ phân bố cụm
                    st.subheader("Phân bố khách hàng theo cụm")

                    # Đếm số lượng khách hàng trong mỗi cụm
                    cluster_counts = result_df['Cluster'].value_counts().reset_index()
                    # Đặt tên cột cho dataframe đếm
                    cluster_counts.columns = ['Cluster', 'Count']

                    # Thêm cột mô tả cụm bằng cách map từ cluster_descriptions
                    cluster_counts['Description'] = cluster_counts['Cluster'].map(models['cluster_descriptions'])

                    # Tạo biểu đồ tròn để hiển thị phân bố
                    fig = px.pie(
                        cluster_counts,  # Dữ liệu nguồn
                        values='Count',  # Giá trị để tính tỷ lệ
                        names='Description',  # Nhãn hiển thị
                        title="Phân bố khách hàng theo cụm"  # Tiêu đề biểu đồ
                    )
                    # Hiển thị biểu đồ với chiều rộng tự động
                    st.plotly_chart(fig, use_container_width=True)

                    # Tạo nút tải xuống kết quả phân cụm
                    with open(result_file_path, "rb") as file:  # Mở file ở chế độ đọc binary
                        st.download_button(
                            label="Tải xuống kết quả phân cụm",  # Nhãn nút
                            data=file,  # Dữ liệu file
                            file_name="customer_clusters_result.csv",  # Tên file khi tải xuống
                            mime="text/csv"  # Loại MIME của file
                        )
                else:
                    # Hiển thị thông báo lỗi nếu phân tích thất bại
                    st.error("Có lỗi xảy ra khi phân tích file.")

            # Xóa file tạm thời sau khi xử lý xong
            if os.path.exists(temp_file_path):  # Kiểm tra file có tồn tại không
                os.remove(temp_file_path)  # Xóa file tạm thời

# Tạo footer cho ứng dụng
st.markdown("---")  # Tạo đường kẻ ngang
st.markdown("📊 **Ứng dụng phân cụm khách hàng**")  # Hiển thị tên ứng dụng









