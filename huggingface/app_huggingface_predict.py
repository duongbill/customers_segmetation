import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from huggingface.use_huggingface_model import load_models_from_hub, predict_customer_cluster_hub

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân cụm khách hàng - Hugging Face",
    page_icon="🤗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS để tối ưu cho laptop 13 inch
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
""", unsafe_allow_html=True)

# Tiêu đề ứng dụng
st.title("🤗 Phân cụm khách hàng - Hugging Face")
st.markdown("Phân loại khách hàng sử dụng mô hình từ Hugging Face Hub.")

# Load models từ Hugging Face Hub
@st.cache_resource
def get_models():
    """Cache models để tránh load lại mỗi lần refresh trang"""
    return load_models_from_hub("duonggbill/dbill-customer-model")

# Tạo sidebar
st.sidebar.header("Chọn chức năng")
option = st.sidebar.radio(
    "Bạn muốn làm gì?",
    ["Dự đoán cho khách hàng mới", "Phân tích file khách hàng", "Thông tin về các cụm"]
)

# Load models
try:
    with st.spinner("Đang tải mô hình từ Hugging Face Hub..."):
        models = get_models()
    st.sidebar.success("Đã tải xong các mô hình từ Hugging Face!")
except Exception as e:
    st.sidebar.error(f"Lỗi khi tải mô hình: {str(e)}")
    st.stop()

# Hiển thị thông tin về các cụm
if option == "Thông tin về các cụm":
    st.header("Thông tin về các cụm khách hàng")

    # Lấy thông tin về các cụm
    cluster_stats = models['cluster_stats']
    cluster_descriptions = models['cluster_descriptions']

    # Hiển thị mô tả các cụm
    st.subheader("Mô tả các cụm")
    for cluster, description in cluster_descriptions.items():
        st.markdown(f"**Cụm {cluster}**: {description}")

    # Tạo DataFrame để hiển thị thống kê
    stats_data = []
    for cluster in cluster_descriptions.keys():
        stats = cluster_stats.loc[cluster]
        stats_data.append({
            'Cụm': cluster,
            'Mô tả': cluster_descriptions[cluster],
            'Số lượng khách hàng': int(stats[('age', 'count')]),
            'Tuổi trung bình': round(stats[('age', 'mean')], 1),
            'Điểm chi tiêu TB': round(stats[('spending_score', 'mean')], 1),
            'Tần suất mua hàng TB': round(stats[('purchase_frequency', 'mean')], 1),
            'Giá trị đơn hàng TB': round(stats[('last_purchase_amount', 'mean')], 0)
        })

    stats_df = pd.DataFrame(stats_data)
    
    # Hiển thị bảng thống kê
    st.subheader("Thống kê chi tiết")
    st.dataframe(stats_df, use_container_width=True)

    # Hiển thị biểu đồ phân bố khách hàng theo cụm
    fig_pie = px.pie(
        stats_df,
        values='Số lượng khách hàng',
        names='Mô tả',
        title="Phân bố khách hàng theo cụm"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Dự đoán cho khách hàng mới
elif option == "Dự đoán cho khách hàng mới":
    st.header("Dự đoán cụm khách hàng")

    # Tạo các input fields
    st.subheader("Thông tin khách hàng")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Giới tính", ["Male", "Female", "Other"])
        spending_score = st.slider("Điểm chi tiêu (0-100)", 0, 100, 50)
        membership_years = st.number_input("Số năm thành viên", min_value=0, max_value=50, value=2)

    with col2:
        purchase_frequency = st.number_input("Tần suất mua hàng (số lần/năm)", min_value=0, max_value=365, value=12)
        preferred_category = st.selectbox(
            "Danh mục ưa thích",
            ["Electronics", "Clothing", "Groceries", "Sports", "Home & Garden", "Beauty", "Books", "Other"]
        )
        last_purchase_amount = st.number_input("Số tiền mua hàng gần nhất ($)", min_value=0.0, value=100.0)

    # Tạo dictionary chứa thông tin khách hàng
    customer_data = {
        'age': age,
        'gender': gender,
        'spending_score': spending_score,
        'membership_years': membership_years,
        'purchase_frequency': purchase_frequency,
        'preferred_category': preferred_category,
        'last_purchase_amount': last_purchase_amount
    }

    # Thêm nút phân tích
    if st.button("🔍 Phân tích khách hàng", type="primary"):
        # Dự đoán cụm khi nhấn nút
        result = predict_customer_cluster_hub(customer_data, models)
    else:
        result = None

    if result:
        # Hiển thị kết quả
        st.success(f"**Cụm {result['cluster']}**: {result['description']}")

        # Tạo 2 cột để hiển thị thông tin
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Thống kê cụm")
            st.write(f"**Số lượng:** {result['stats'][('age', 'count')]}")
            st.write(f"**Tuổi TB:** {result['stats'][('age', 'mean')]:.1f}")
            st.write(f"**Chi tiêu TB:** {result['stats'][('spending_score', 'mean')]:.1f}")
            st.write(f"**Tần suất mua:** {result['stats'][('purchase_frequency', 'mean')]:.1f}")
            st.write(f"**Giá trị đơn hàng:** ${result['stats'][('last_purchase_amount', 'mean')]:.0f}")

        with col2:
            st.subheader("Gợi ý marketing")
            cluster = result['cluster']
            
            # Gợi ý marketing dựa trên cluster
            if cluster == 0:
                st.write("🎯 **Chiến lược:** Khách hàng VIP")
                st.write("• Chương trình loyalty cao cấp")
                st.write("• Sản phẩm premium")
                st.write("• Dịch vụ cá nhân hóa")
            elif cluster == 1:
                st.write("🎯 **Chiến lược:** Phát triển tiềm năng")
                st.write("• Khuyến mãi hấp dẫn")
                st.write("• Giới thiệu sản phẩm mới")
                st.write("• Tăng tần suất tương tác")
            elif cluster == 2:
                st.write("🎯 **Chiến lược:** Khách hàng trung thành")
                st.write("• Duy trì chất lượng dịch vụ")
                st.write("• Cross-selling")
                st.write("• Chương trình giới thiệu bạn bè")
            else:
                st.write("🎯 **Chiến lược:** Kích hoạt lại")
                st.write("• Khuyến mãi đặc biệt")
                st.write("• Email marketing")
                st.write("• Khảo sát nhu cầu")

# Phân tích file khách hàng
elif option == "Phân tích file khách hàng":
    st.header("Phân tích file khách hàng")

    # Hướng dẫn định dạng file
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

    # Upload file
    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

    if uploaded_file is not None:
        # Lưu file tạm thời
        temp_file_path = "temp_customer_data.csv"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Hiển thị dữ liệu gốc
        df_original = pd.read_csv(temp_file_path)
        st.subheader("Dữ liệu khách hàng gốc")
        st.dataframe(df_original)

        # Phân tích file
        if st.button("Phân tích và phân cụm"):
            with st.spinner("Đang phân tích dữ liệu..."):
                # Phân tích từng khách hàng
                results = []
                for _, row in df_original.iterrows():
                    customer_data = row.to_dict()
                    result = predict_customer_cluster_hub(customer_data, models)
                    if result:
                        results.append({
                            **customer_data,
                            'Cluster': result['cluster'],
                            'Cluster_Description': result['description']
                        })

                if results:
                    result_df = pd.DataFrame(results)
                    
                    # Lưu kết quả
                    result_file_path = "customer_clusters_result.csv"
                    result_df.to_csv(result_file_path, index=False)

                    # Hiển thị kết quả
                    st.subheader("Kết quả phân cụm")
                    st.dataframe(result_df)

                    # Tạo biểu đồ phân bố cụm
                    st.subheader("Phân bố khách hàng theo cụm")

                    # Đếm số lượng khách hàng trong mỗi cụm
                    cluster_counts = result_df['Cluster'].value_counts().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']

                    # Thêm mô tả cụm
                    cluster_counts['Description'] = cluster_counts['Cluster'].map(models['cluster_descriptions'])

                    # Vẽ biểu đồ
                    fig = px.pie(
                        cluster_counts,
                        values='Count',
                        names='Description',
                        title="Phân bố khách hàng theo cụm"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Tạo nút tải xuống kết quả
                    with open(result_file_path, "rb") as file:
                        st.download_button(
                            label="Tải xuống kết quả phân cụm",
                            data=file,
                            file_name="customer_clusters_result.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("Có lỗi xảy ra khi phân tích file.")

        # Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Footer
st.markdown("---")
st.markdown("🤗 **Ứng dụng phân cụm khách hàng - Powered by Hugging Face Hub**")
