import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

# 1. Giả lập dữ liệu huấn luyện (giống với 50 mẫu của con)
# Các cột: [Screen Time, Pickups, Focus Time]
np.random.seed(42)
data = np.random.randint(50, 650, size=(50, 3))

# 2. Khởi tạo mô hình K-Means với 5 cụm
# Đây chính là "bộ não" phân loại học sinh thành 5 nhóm
model = KMeans(n_clusters=5, random_state=42, n_init=10)

# 3. Cho AI học dữ liệu
model.fit(data)

# 4. XUẤT FILE .PKL (Đóng gói bộ não)
joblib.dump(model, 'focus_model.pkl')

print("--------------------------------------------------")
print("✅ THÀNH CÔNG RỒI!")
print("Thầy đã tạo ra file 'focus_model.pkl' cho con.")
print("Bây giờ con hãy tìm file này trong thư mục và upload lên GitHub nhé.")
print("--------------------------------------------------")
