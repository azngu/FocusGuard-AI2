import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

# --- 1. TẠO DỮ LIỆU MẪU (Dành cho báo cáo 50 mẫu) ---
# Nếu con đã có file data_50_samples.csv thì có thể dùng lệnh pd.read_csv
# Ở đây thầy tự động tạo dữ liệu để đảm bảo code chạy thông suốt 100%
np.random.seed(42)

# Giả lập 3 chỉ số chính của dự án: Screen Time, Pickups, Focus Time
data = {
    'Screen_Time': np.random.randint(50, 700, 50),
    'Pickups': np.random.randint(10, 200, 50),
    'Focus_Time': np.random.randint(20, 450, 50)
}

df = pd.DataFrame(data)

# --- 2. HUẤN LUYỆN MÔ HÌNH AI (K-Means Clustering) ---
# Chọn 3 cột dữ liệu để AI học
X = df[['Screen_Time', 'Pickups', 'Focus_Time']]

# Khởi tạo thuật toán K-Means với 5 cụm (tương ứng 5 nhóm đã đặt tên)
# n_init=10 để đảm bảo thuật toán chạy ổn định
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# Tiến hành "cho AI học"
model.fit(X)

# --- 3. ĐÓNG GÓI MÔ HÌNH (Xuất file .pkl) ---
# Đây chính là bước quan trọng nhất để tạo ra "bộ não" cho app.py
joblib.dump(model, 'focus_model.pkl')

# --- 4. THÔNG BÁO KẾT QUẢ ---
print("="*50)
print("✅ QUY TRÌNH HUẤN LUYỆN HOÀN TẤT!")
print(f"1. Đã xử lý: {len(df)} mẫu dữ liệu học sinh.")
print(f"2. Thuật toán sử dụng: K-Means Clustering (K=5).")
print(f"3. FILE XUẤT RA: focus_model.pkl")
print("="*50)
print("Bây giờ con hãy copy file 'focus_model.pkl' vừa xuất hiện")
print("vào cùng thư mục với file 'app.py' để chạy ứng dụng nhé!")
