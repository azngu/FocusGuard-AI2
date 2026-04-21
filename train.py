import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

# 1. Tạo dữ liệu giả lập 50 mẫu (Screen Time, Pickups, Focus Time)
np.random.seed(42)
data = {
    'screen_time': np.random.randint(50, 700, 50),
    'pickups': np.random.randint(10, 150, 50),
    'focus_time': np.random.randint(30, 450, 50)
}
df = pd.DataFrame(data)

# 2. Huấn luyện mô hình K-Means với 5 cụm (tương ứng 5 nhóm đã đặt tên)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(df)

# 3. Lưu mô hình vào file 'focus_model.pkl' để app.py sử dụng
joblib.dump(kmeans, 'focus_model.pkl')

print("✅ Đã huấn luyện xong và lưu mô hình vào file focus_model.pkl!")