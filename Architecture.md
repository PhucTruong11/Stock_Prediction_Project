# Kiến Trúc Hệ Thống (System Architecture)

Tài liệu này mô tả chi tiết luồng xử lý dữ liệu và kiến trúc huấn luyện mô hình của dự án **Dự đoán Giá Cổ Phiếu**. Mục tiêu cốt lõi là tối ưu hóa quá trình trích xuất đặc trưng (Feature Engineering) và huấn luyện mô hình (Training Process) để đạt được kết quả dự đoán sát với xu hướng thực tế.

---

## 1. Luồng Xử Lý Dữ Liệu (Data Pipeline)

Hệ thống hoạt động qua 4 bước chính:
**`Raw Data` ➔ `Feature Engineering` ➔ `Sliding Window Processing` ➔ `Model Training`**

### Bước 1: Thu thập & Làm sạch dữ liệu (Data Ingestion)
- **Nguồn:** `yfinance` (cho cổ phiếu Mỹ như NQ, AAPL) hoặc `vnstock` (cho cổ phiếu Việt Nam như HPG, VNINDEX).
- **Timeframe:** 3-5 năm gần nhất.
- **Xử lý thô:** Loại bỏ các ngày nghỉ lễ (không có giao dịch), điền khuyết dữ liệu (Missing Values) bằng phương pháp `Forward Fill` (ffill).

### Bước 2: Tinh chế Dữ liệu & Kỹ thuật Định lượng (Feature Engineering)
Thay vì chỉ dùng dữ liệu thô (OHLCV), chúng ta áp dụng các thuật toán định lượng để tạo ra các đặc trưng có giá trị cao giúp mô hình học dễ hơn:
1. **Trend Indicators (Chỉ báo xu hướng):** 
   - *SMA 10, SMA 20, SMA 50* (Đường trung bình động đơn giản).
   - *MACD* (Phân kỳ hội tụ trung bình động).
2. **Momentum Indicators (Chỉ báo động lượng):**
   - *RSI 14* (Chỉ số sức mạnh tương đối) - Đo lường mức độ quá mua/quá bán.
3. **Volatility (Độ biến động):**
   - *Bollinger Bands (Upper/Lower)* - Đo lường biên độ dao động giá.

### Bước 3: Đóng gói Chuỗi Thời Gian (Sliding Window)
Mô hình Deep Learning (LSTM) không nhận từng dòng dữ liệu đơn lẻ, nó nhận dữ liệu theo dạng **Cửa sổ trượt (Sliding Window)**.
- **Window Size ($W$):** Ví dụ 60 ngày. (Lấy 60 ngày quá khứ làm 1 mẫu).
- **Target Size ($N$):** Ví dụ 5 ngày. (Dự đoán 5 ngày tiếp theo).
- **Scale Dữ liệu:** BẮT BUỘC dùng `MinMaxScaler(0, 1)` trước khi chia cửa sổ để đồng bộ hóa đơn vị đo lường (Giá có thể lên tới 10,000, trong khi RSI chỉ từ 0-100).
  *(Lưu ý: Chỉ fit scaler trên tập Training để tránh Data Leakage).*

---

## 2. Kiến Trúc Mô Hình (Model Architecture)

Chúng ta sử dụng mạng nơ-ron hồi quy **LSTM (Long Short-Term Memory)** hoặc **GRU (Gated Recurrent Unit)**. Đây là các kiến trúc mạnh nhất trong việc xử lý dữ liệu chuỗi (Sequential Data).

```text
[Input Layer]
  - Shape: (Batch_Size, 60, Số_lượng_Features)
       ↓
[LSTM Layer 1] 
  - 128 units, return_sequences=True
  - Activation: tanh
       ↓
[Dropout Layer]
  - Rate: 0.2 (Chống Overfitting)
       ↓
[LSTM Layer 2]
  - 64 units, return_sequences=False
       ↓
[Dense Layer]
  - 32 units, Activation: ReLU (Học kết hợp các đặc trưng bậc cao)
       ↓
[Output Layer]
  - N units (N là số ngày cần dự đoán, ví dụ: 5)
  - Activation: Linear (Vì đây là bài toán Hồi quy)
```

---

## 3. Quy Trình Huấn Luyện (Training Process)

Để đưa ra kết quả tương đối chính xác và tránh bị Overfitting, quy trình huấn luyện được cấu hình chặt chẽ:

### Loss Function & Optimizer
- **Loss Function:** Sử dụng **MSE (Mean Squared Error)** hoặc **Huber Loss** (Huber Loss chống nhiễu (outliers) tốt hơn khi thị trường có những ngày biến động giá bất thường).
- **Optimizer:** `Adam` (với tốc độ học mặc định 0.001).

### Callbacks (Tinh chỉnh tự động)
Trong quá trình train trên Kaggle/Colab, ta dùng 2 công cụ để tinh chỉnh:
1. **Early Stopping:** Dừng train ngay lập tức nếu hàm Loss trên tập Validation không giảm trong $X$ vòng (patience = 10) -> Giúp tiết kiệm GPU và chống Overfitting.
2. **ReduceLROnPlateau:** Tự động giảm tốc độ học (Learning Rate) xuống nếu mô hình bắt đầu bão hòa -> Giúp mô hình hội tụ sâu hơn vào điểm tối ưu.

### Đánh Giá Mô Hình (Evaluation Metrics)
Sau khi Inverse Transform (đưa dữ liệu từ scale 0-1 về lại giá trị tiền tệ thực tế), ta đo lường bằng:
- **RMSE (Root Mean Squared Error):** Cho biết sai số trung bình là bao nhiêu giá (VD: lệch 1.5 điểm).
- **MAPE (Mean Absolute Percentage Error):** Tính phần trăm sai số, dễ hình dung cho báo cáo (VD: Mô hình dự đoán lệch 2% so với thực tế).
