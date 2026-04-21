import pandas as pd
from sklearn.cluster import KMeans
import joblib
import numpy as np

# 1. Tạo dữ liệu giả lập dựa trên thống kê của dự án (50 mẫu)
# Các cột: [Screen Time, Pickups, Focus Time]
np.random.seed(42)
data = np.random.randint(50, 600, size=(50, 3))

# 2. Khởi tạo mô hình K-Means với 5 cụm (tương ứng 5 nhóm con đã đặt tên)
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# 3. Cho AI "học" (Fit) dữ liệu này
model.fit(data)

# 4. "Đóng gói" AI này vào file .pkl
joblib.dump(model, 'focus_model.pkl')

print("✅ Đã tạo thành công file focus_model.pkl!")
print("Bây giờ con có thể dùng file này để upload lên Streamlit rồi đó.")