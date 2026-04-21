import streamlit as st
import joblib
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="FocusGuard AI", page_icon="🧘‍♂️")

# Nạp mô hình đã huấn luyện
try:
    model = joblib.load('focus_model.pkl')
except:
    st.error("Không tìm thấy file mô hình! Hãy chạy file train.py trước.")

st.title("🛡️ FocusGuard AI")
st.write("### Nhập thông số sử dụng thiết bị")

col1, col2 = st.columns(2)
with col1:
    st_val = st.slider("Tổng Screen Time (phút):", 0, 700, 250)
    pk_val = st.number_input("Số lần mở máy (Pickups):", 0, 200, 50)
with col2:
    ft_val = st.slider("Thời gian tập trung (phút):", 0, 500, 180)

if st.button("🚀 PHÂN TÍCH"):
    # AI dự đoán nhóm dựa trên dữ liệu nhập vào
    input_data = np.array([[st_val, pk_val, ft_val]])
    cluster = model.predict(input_data)[0]
    
    # Hệ thống chuyên gia đối chiếu (Mapping)
    # Lưu ý: Thứ tự cụm 0-4 phụ thuộc vào dữ liệu, đây là logic mẫu:
    groups = {
        0: ("Deep Flow", "🧘‍♂️", "green", "Chiến binh Tập trung", "Trạng thái đỉnh cao! Duy trì kỷ luật.", "Duy trì Pomodoro 50/10."),
        1: ("Smart Pulse", "✨", "blue", "Người dùng Thông thái", "Hiệu suất rất tốt!", "Tắt thông báo không cần thiết."),
        2: ("Steady Mode", "⚖️", "yellow", "Trạng thái Thăng bằng", "Cần thêm sự ổn định.", "Đặt giới hạn 30p cho MXH."),
        3: ("Wandering Mind", "⚠️", "orange", "Tâm trí Lang thang", "Cảnh báo xao nhãng!", "Thực hiện bài tập thở 2 phút."),
        4: ("Digital Fog", "🚨", "red", "Màn sương Kỹ thuật số", "Báo động đỏ! Rời màn hình ngay.", "Nhìn xa 20m trong 20 giây.")
    }
    
    name, emoji, color, status, advice, action = groups[cluster]
    
    st.divider()
    st.markdown(f"### Kết quả: :{color}[{name} {emoji}]")
    st.info(f"**Trạng thái:** {status}")
    st.write(f"💡 **Khuyến nghị:** {advice}")
    st.success(f"🎯 **Hành động:** {action}")
    
    st.divider()
    st.caption("🔍 **Hệ thống chuyên gia (Expert System):** Ứng dụng sử dụng mô hình K-Means để phân cụm, sau đó đối chiếu với bộ quy tắc hành động để đưa ra hướng dẫn cá nhân hóa (Micro-habits).")