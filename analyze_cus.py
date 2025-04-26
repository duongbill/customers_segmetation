import pandas as pd
import numpy as np
import os
from joblib import load
import streamlit as st

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

def analyze_customer_types(data):
    # Phân tích các nhóm khách hàng dựa trên dữ liệu
    customer_types = {
        'VIP': {
            'criteria': {
                'spending_score': {'min': 70},
                'purchase_frequency': {'min': 30}
            },
            'description': 'Khách hàng có điểm chi tiêu cao và tần suất mua hàng cao',
            'icon': '🎯',
            'color': '#FF6B6B'
        },
        'Young High Spenders': {
            'criteria': {
                'age': {'max': 35},
                'spending_score': {'min': 60}
            },
            'description': 'Khách hàng trẻ tuổi có điểm chi tiêu cao',
            'icon': '👤',
            'color': '#4ECDC4'
        },
        'Old Low Spenders': {
            'criteria': {
                'age': {'min': 50},
                'spending_score': {'max': 40}
            },
            'description': 'Khách hàng lớn tuổi có điểm chi tiêu thấp',
            'icon': '📊',
            'color': '#FFE66D'
        },
        'Frequent Buyers': {
            'criteria': {
                'purchase_frequency': {'min': 20},
                'spending_score': {'min': 50, 'max': 70}
            },
            'description': 'Khách hàng mua hàng thường xuyên với mức chi tiêu trung bình',
            'icon': '💰',
            'color': '#95E1D3'
        },
        'Average Customers': {
            'criteria': {
                'age': {'min': 35, 'max': 50},
                'spending_score': {'min': 40, 'max': 60},
                'purchase_frequency': {'min': 15, 'max': 25}
            },
            'description': 'Khách hàng có đặc điểm trung bình về tuổi, chi tiêu và tần suất mua hàng',
            'icon': '📝',
            'color': '#A8E6CF'
        },
        'New Customers': {
            'criteria': {
                'membership_years': {'max': 3},
                'purchase_frequency': {'max': 15}
            },
            'description': 'Khách hàng mới tham gia với tần suất mua hàng thấp',
            'icon': '💳',
            'color': '#DCEDC1'
        }
    }
    
    # Phân loại khách hàng
    for type_name, type_info in customer_types.items():
        mask = pd.Series(True, index=data.index)
        for feature, criteria in type_info['criteria'].items():
            if 'min' in criteria:
                mask &= (data[feature] >= criteria['min'])
            if 'max' in criteria:
                mask &= (data[feature] <= criteria['max'])
        customer_types[type_name]['customers'] = data[mask]
        customer_types[type_name]['count'] = len(data[mask])
    
    return customer_types

def get_customer_type(customer, customer_types):
    # Xác định loại khách hàng dựa trên thông tin đầu vào
    for type_name, type_info in customer_types.items():
        is_match = True
        for feature, criteria in type_info['criteria'].items():
            if 'min' in criteria and customer[feature] < criteria['min']:
                is_match = False
                break
            if 'max' in criteria and customer[feature] > criteria['max']:
                is_match = False
                break
        if is_match:
            return type_name, type_info
    return "Other", {
        "description": "Khách hàng không thuộc các nhóm đã định nghĩa",
        "icon": "❓",
        "color": "#B8B8B8"
    }

def get_marketing_suggestions(customer_type, customer_data):
    suggestions = {
        'VIP': [
            "🎁 Gửi quà tặng độc quyền và ưu đãi đặc biệt",
            "💎 Mời tham gia chương trình khách hàng thân thiết cao cấp",
            "🌟 Ưu tiên tiếp cận sản phẩm mới trước",
            "📱 Dịch vụ chăm sóc khách hàng VIP"
        ],
        'Young High Spenders': [
            "📱 Tập trung marketing qua mạng xã hội",
            "🎮 Tạo trải nghiệm mua sắm tương tác",
            "🎁 Chương trình giới thiệu bạn bè",
            "⚡ Flash sales và ưu đãi giới hạn"
        ],
        'Old Low Spenders': [
            "📞 Hỗ trợ mua hàng qua điện thoại",
            "📝 Gửi catalog và brochure truyền thống",
            "💰 Chương trình tích điểm đổi quà",
            "🎯 Ưu đãi cho sản phẩm phù hợp lứa tuổi"
        ],
        'Frequent Buyers': [
            "🎯 Chương trình khách hàng thân thiết",
            "📦 Miễn phí vận chuyển",
            "💝 Quà tặng cho đơn hàng thường xuyên",
            "📅 Lịch mua sắm được cá nhân hóa"
        ],
        'Average Customers': [
            "📊 Khảo sát và nghiên cứu thị trường",
            "🎁 Ưu đãi theo mùa",
            "📱 Thông báo khuyến mãi định kỳ",
            "💡 Gợi ý sản phẩm phù hợp"
        ],
        'New Customers': [
            "🎉 Ưu đãi chào mừng hấp dẫn",
            "📱 Hướng dẫn sử dụng app/website",
            "💝 Quà tặng cho đơn hàng đầu tiên",
            "📞 Hỗ trợ tư vấn tận tình"
        ],
        'Other': [
            "📊 Phân tích hành vi mua hàng",
            "🎯 Tạo chương trình marketing phù hợp",
            "📱 Gửi thông tin sản phẩm mới",
            "💡 Đề xuất sản phẩm theo xu hướng"
        ]
    }
    
    category_suggestions = {
        'Groceries': [
            "🛒 Giảm giá cho mua sắm định kỳ",
            "📦 Combo sản phẩm tiết kiệm",
            "🎁 Quà tặng thực phẩm bổ sung"
        ],
        'Sports': [
            "🏃‍♂️ Tư vấn sản phẩm theo môn thể thao",
            "👕 Ưu đãi cho bộ sưu tập mới",
            "🎽 Quà tặng phụ kiện thể thao"
        ],
        'Clothing': [
            "👔 Tư vấn phong cách cá nhân",
            "👗 Thông báo bộ sưu tập mới",
            "🎁 Quà tặng phụ kiện thời trang"
        ],
        'Home & Garden': [
            "🏠 Tư vấn trang trí nhà cửa",
            "🌺 Ưu đãi theo mùa",
            "🛠️ Combo sản phẩm tiện ích"
        ],
        'Electronics': [
            "📱 Bảo hành và dịch vụ ưu đãi",
            "💻 Tư vấn sản phẩm công nghệ",
            "🎮 Quà tặng phụ kiện điện tử"
        ]
    }
    
    base_suggestions = suggestions[customer_type]
    category_specific = category_suggestions[customer_data['preferred_category']]
    
    return base_suggestions, category_specific

def main():
    # Streamlit interface
    st.set_page_config(
        page_title="Phân tích khách hàng",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Main theme */
    :root {
        --background: #0B0F19;
        --card-bg: #1A1F29;
        --accent: #FF4B4B;
        --text: #FFFFFF;
        --text-secondary: #A1A1AA;
        --border: #2D3748;
    }

    .stApp {
        background-color: var(--background);
    }

    /* Container styles */
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
    }

    /* Common input styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div {
        background-color: #1E1E1E !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 4px !important;
        color: var(--text) !important;
        height: 35px !important;
        min-height: 35px !important;
        width: 100% !important;
        font-size: 0.875rem !important;
    }

    /* Number input specific */
    .stNumberInput > div > div > input {
        padding: 0.5rem !important;
    }

    /* Select box specific */
    .stSelectbox > div > div {
        height: 35px !important;
        line-height: 35px !important;
        padding: 0 0.5rem !important;
    }

    .stSelectbox [data-baseweb="select"] {
        height: 35px !important;
    }

    /* Labels */
    .stTextInput label, 
    .stNumberInput label, 
    .stSelectbox label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.25rem !important;
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0 2rem 0 !important;
    }

    .stSlider > div > div > div > div {
        background-color: var(--accent) !important;
    }

    .stSlider > div > div > div {
        background-color: #2D2D2D !important;
    }

    .stSlider > div > div {
        font-size: 0.875rem !important;
    }

    /* Button */
    .stButton > button {
        background-color: var(--accent) !important;
        color: white !important;
        border: none !important;
        height: 35px !important;
        font-size: 0.875rem !important;
        border-radius: 4px !important;
        width: 100% !important;
        margin-top: 1.5rem !important;
        cursor: pointer !important;
    }

    /* Section titles */
    .section-title {
        color: var(--text);
        font-size: 1rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Remove spinners from number inputs */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield;
    }

    /* Custom minus/plus buttons */
    .stNumberInput > div > div > button {
        background-color: #2D2D2D !important;
        border: 1px solid #3D3D3D !important;
        color: white !important;
        width: 25px !important;
        height: 35px !important;
        padding: 0 !important;
        font-size: 1rem !important;
    }

    .stNumberInput > div > div > button:hover {
        background-color: #3D3D3D !important;
    }

    /* Container for form sections */
    .form-section {
        background: #1A1F29;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div class="section-title" style="font-size: 1.25rem; margin-bottom: 2rem;">
        🎯 Phân tích khách hàng
    </div>
    """, unsafe_allow_html=True)

    # Input sections
    st.markdown("""
    <div class="section-title">
        👤 Nhập thông tin khách hàng
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="form-section">
                <div class="section-title" style="font-size: 0.9rem; margin-bottom: 1rem;">
                    📝 Thông tin cơ bản
                </div>
            """, unsafe_allow_html=True)
            
            age = st.number_input('Tuổi', min_value=18, max_value=100, value=25)
            gender = st.selectbox('Giới tính', ['Male', 'Female', 'Other'])
            spending_score = st.slider('Điểm chi tiêu', 0, 100, 50)
            membership_years = st.number_input('Số năm thành viên', 1, 10, 2)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="form-section">
                <div class="section-title" style="font-size: 0.9rem; margin-bottom: 1rem;">
                    💰 Thông tin mua hàng
                </div>
            """, unsafe_allow_html=True)
            
            purchase_frequency = st.number_input('Tần suất mua hàng', 1, 50, 10)
            last_purchase_amount = st.number_input('Số tiền mua hàng cuối', 0, 1000, 100)
            preferred_category = st.selectbox('Danh mục ưa thích', 
                                           ['Groceries', 'Sports', 'Clothing', 'Home & Garden', 'Electronics'])
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    if st.button('🔍 Phân tích khách hàng'):
        customer_data = {
            'age': age,
            'gender': gender,
            'spending_score': spending_score,
            'membership_years': membership_years,
            'purchase_frequency': purchase_frequency,
            'last_purchase_amount': last_purchase_amount,
            'preferred_category': preferred_category
        }
        
        # Get customer type and suggestions
        customer_type, type_info = get_customer_type(customer_data, analyze_customer_types(data))
        base_suggestions, category_suggestions = get_marketing_suggestions(customer_type, customer_data)
        
        # Results container
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Tabs
        tab1, tab2 = st.tabs(["📊 Thông tin khách hàng", "🎯 Chiến lược Marketing"])
        
        with tab1:
            st.markdown("""
            <div class="section-header">
                <h2>📊 Kết quả phân tích</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Customer type display
            st.markdown(f"""
            <h3 style="color: {type_info['color']}; font-size: 1.8em; margin: 1rem 0;">
                {type_info['icon']} {customer_type}
            </h3>
            <p style="color: var(--text-gray); font-size: 1.2em; margin-bottom: 2rem;">
                {type_info['description']}
            </p>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            metric_style = """
            <div class="metric-card">
                <h4>{label}</h4>
                <p>{value}</p>
            </div>
            """
            
            with col1:
                st.markdown(metric_style.format(label="Tuổi", value=age), unsafe_allow_html=True)
                st.markdown(metric_style.format(label="Giới tính", value=gender), unsafe_allow_html=True)
            
            with col2:
                st.markdown(metric_style.format(label="Điểm chi tiêu", value=spending_score), unsafe_allow_html=True)
                st.markdown(metric_style.format(label="Năm thành viên", value=membership_years), unsafe_allow_html=True)
            
            with col3:
                st.markdown(metric_style.format(label="Tần suất mua", value=purchase_frequency), unsafe_allow_html=True)
                st.markdown(metric_style.format(label="Số tiền mua cuối", value=f"{last_purchase_amount:,.0f}"), unsafe_allow_html=True)
            
            with col4:
                st.markdown(metric_style.format(label="Danh mục ưa thích", value=preferred_category), unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="section-header">
                <h2>🎯 Gợi ý chiến lược Marketing</h2>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="marketing-card">
                    <h3 style="color: var(--primary-color); margin-bottom: 1rem;">
                        🎯 Chiến lược chung
                    </h3>
                    <ul class="suggestion-list">
                """, unsafe_allow_html=True)
                
                for suggestion in base_suggestions:
                    st.markdown(f"""
                    <li class="suggestion-item">
                        {suggestion}
                    </li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="marketing-card">
                    <h3 style="color: var(--secondary-color); margin-bottom: 1rem;">
                        🛍️ Gợi ý cho {customer_data['preferred_category']}
                    </h3>
                    <ul class="suggestion-list">
                """, unsafe_allow_html=True)
                
                for suggestion in category_suggestions:
                    st.markdown(f"""
                    <li class="suggestion-item">
                        {suggestion}
                    </li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()