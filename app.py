import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="FocusGuard AI - Trợ lý Sức khỏe số", 
    page_icon="🛡️",
    layout="centered"
)

# --- 2. GIAO DIỆN TIÊU ĐỀ ---
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🛡️ FocusGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Tối ưu hóa sự tập trung & Cân bằng cuộc sống kỹ thuật số</p>", unsafe_allow_html=True)
st.divider()

# --- 3. KIỂM TRA FILE MÔ HÌNH ---
model_path = 'focus_model.pkl'
model_ready = False

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        model_ready = True
    except:
        st.error("⚠️ Lỗi: Không thể nạp file mô hình. Hãy kiểm tra lại file .pkl.")
else:
    st.warning("ℹ️ Chế độ mô phỏng: Hiện tại không tìm thấy file 'focus_model.pkl'. Hệ thống đang chờ con tải file lên GitHub/Thư mục.")

# --- 4. NHẬP LIỆU TỪ NGƯỜI DÙNG ---
st.subheader("📝 Nhập thông số sử dụng thiết bị")
col1, col2 = st.columns(2)

with col1:
    st_val = st.slider("Tổng Screen Time (Phút/ngày):", 0, 700, 250, help="Tổng thời gian bạn nhìn vào màn hình điện thoại/máy tính.")
    pk_val = st.number_input("Số lần mở máy (Pickups/ngày):", min_value=0, max_value=200, value=50)

with col2:
    ft_val = st.slider("Thời gian học tập trung (Phút/ngày):", 0, 500, 180, help="Thời gian bạn thực sự tập trung học mà không chạm vào điện thoại.")
    stress = st.select_slider("Mức độ căng thẳng cảm nhận:", options=["Thấp", "Trung bình", "Cao"])

# --- 5. LOGIC PHÂN LOẠI & KHUYẾN NGHỊ ---
if st.button("🚀 PHÂN TÍCH HÀNH VI"):
    if model_ready:
        # Sử dụng mô hình AI thực tế
        input_data = np.array([[st_val, pk_val, ft_val]])
        cluster = model.predict(input_data)[0]
    else:
        # Logic dự phòng nếu chưa có file .pkl (Dựa trên ngưỡng Screen Time)
        if st_val < 150: cluster = 0
        elif st_val < 250: cluster = 1
        elif st_val < 400: cluster = 2
        elif st_val < 550: cluster = 3
        else: cluster = 4

    # Hệ thống chuyên gia (Expert System Mapping)
    groups = {
        0: ("Deep Flow", "🧘‍♂️", "green", "Chiến binh Tập trung", 
            "Trạng thái đỉnh cao! Hãy duy trì kỷ luật này để đạt hiệu suất học tập tốt nhất.", 
            "Duy trì phương pháp Pomodoro 50/10. Chia sẻ bí quyết tập trung của bạn lên cộng đồng."),
        1: ("Smart Pulse", "✨", "blue", "Người dùng Thông thái", 
            "Hiệu suất rất tốt! Bạn đang làm chủ thiết bị, nhưng hãy cẩn thận với các thông báo rác.", 
            "Kiểm tra và tắt thông báo từ các ứng dụng không cần thiết (Social Media)."),
        2: ("Steady Mode", "⚖️", "orange", "Trạng thái Thăng bằng", 
            "Cần thêm sự ổn định. Bạn đang ở ranh giới, hãy đặt giới hạn thời gian cho các ứng dụng giải trí.", 
            "Thiết lập App Limit (giới hạn thời gian) 30 phút cho các ứng dụng xao nhãng nhất."),
        3: ("Wandering Mind", "⚠️", "orange", "Tâm trí Lang thang", 
            "Cảnh báo xao nhãng! Tâm trí bạn đang bắt đầu mất tập trung do thói quen mở máy vô thức.", 
            "Thực hiện ngay bài tập thở 4-7-8 trong 2 phút. Để điện thoại cách xa tầm tay ít nhất 2 mét."),
        4: ("Digital Fog", "🚨", "red", "Màn sương Kỹ thuật số", 
            "Báo động đỏ! Bạn đang bị quá tải kỹ thuật số. Hãy rời khỏi màn hình ngay lập tức.", 
            "Kích hoạt chế độ Focus Mode ngay. Đứng dậy vươn vai và nhìn xa 20 mét trong 20 giây để thư giãn mắt.")
    }

    name, emoji, color, status, advice, action = groups[cluster]

    # HIỂN THỊ KẾT QUẢ
    st.divider()
    st.markdown(f"### Nhóm của bạn: :{color}[{name} {emoji}]")
    st.info(f"**Phân loại:** {status}")
    
    st.markdown(f"💡 **Khuyến nghị từ AI:** *{advice}*")
    st.success(f"🎯 **Hành động cụ thể:** {action}")
    
    # GIẢI THÍCH CHUYÊN SÂU (DÀNH CHO BÁO CÁO)
    with st.expander("Xem giải thích về mô hình AI"):
        st.write("""
        Mô hình AI của FocusGuard không chỉ đưa ra một kết luận chung chung mà sử dụng **Hệ thống chuyên gia (Expert System)** để đối chiếu mã nhóm (từ thuật toán K-Means Clustering) với bộ quy tắc hành động. 
        Điều này giúp ứng dụng không chỉ là một công cụ phân tích mà còn là một **Người huấn luyện (Coach)** thực thụ, 
        giúp học sinh cải thiện sức khỏe số (Digital Wellbeing) theo từng bước nhỏ (Micro-habits).
        """)

# --- 6. CHÂN TRANG ---
st.sidebar.markdown("---")
st.sidebar.write("📊 **Dữ liệu huấn luyện:** 50 mẫu học sinh")
st.sidebar.write("🤖 **Thuật toán:** K-Means Clustering")
st.sidebar.write("🌿 **Dự án:** Green Digital Insights")
