# Dự Án Phân Tích & Dự Đoán Giá Cổ Phiếu (Stock Price Prediction)

## Tổng Quan Dự Án (Overview)
Dự án này tập trung vào việc xây dựng một hệ thống Machine Learning/Deep Learning để dự đoán giá cổ phiếu trong tương lai dựa trên dữ liệu lịch sử. 
Đây là bài toán **Hồi quy chuỗi thời gian đa biến, đa đầu ra (Multivariate, Multi-output Time-Series Regression)**, trong đó:
- **Đầu vào (Input):** Chuỗi dữ liệu quá khứ (ví dụ 3 năm - 5 năm) bao gồm Giá, Khối lượng giao dịch và các chỉ báo phân tích kỹ thuật (Technical Indicators).
- **Đầu ra (Output):** Dự đoán giá đóng cửa (Close price) cho $N$ ngày tiếp theo.

## Mục Tiêu Cốt Lõi
Trọng tâm của dự án không nằm ở việc tạo ra "chén thánh" bách phát bách trúng (điều bất thi trong thị trường tài chính), mà tập trung vào:
1. **Xử lý Dữ liệu (Data Processing & Feature Engineering):** Áp dụng các thuật toán định lượng (SMA, RSI, MACD, Bollinger Bands) để tinh chế dữ liệu thô thành các đặc trưng có độ trích xuất thông tin cao.
2. **Quy trình Huấn luyện (Training Process):** Tối ưu hóa các mô hình chuỗi thời gian (như LSTM, GRU) với cấu trúc Sliding Window để mô hình học được chu kỳ và xu hướng, từ đó đưa ra kết quả tương đối, bám sát xu hướng thị trường.

## Cấu Trúc Thư Mục (Directory Structure)
Dự án được tổ chức theo chuẩn **Cookiecutter Data Science / ML Template** để đảm bảo tính module hóa và dễ dàng tái tạo các thực nghiệm (experiments).

```text
Stock_Prediction_Project/
├── data/                   # Thư mục chứa dữ liệu
│   ├── raw/                # Dữ liệu gốc tải từ yfinance/vnstock
│   ├── interim/            # Dữ liệu trung gian sau khi làm sạch
│   └── processed/          # Dữ liệu cuối cùng (đã chuẩn hóa, tạo sliding windows)
├── models/                 # Chứa các model đã được huấn luyện (saved weights: .h5, .pkl)
├── notebooks/              # Jupyter notebooks dùng để EDA và thử nghiệm
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experimentation.ipynb
├── src/                    # Source code chính của dự án
│   ├── data/               # Code tải dữ liệu (yfinance, vnstock)
│   ├── features/           # Code tính toán indicators (RSI, MACD...) & Sliding Window
│   ├── models/             # Định nghĩa mạng neural (LSTM, GRU), training loop
│   └── visualization/      # Code vẽ biểu đồ đánh giá
├── reports/                # Báo cáo kết quả huấn luyện (biểu đồ loss, metrics)
├── README.md               # Tổng quan dự án
├── Architecture.md         # Chi tiết kiến trúc hệ thống và luồng xử lý
└── requirements.txt        # Các thư viện phụ thuộc
```

## Hướng Dẫn Cài Đặt (Cho các thành viên trong nhóm)
1. Clone dự án về máy:
   ```bash
   git clone <URL_GITHUB_CỦA_DỰ_ÁN>
   cd Stock_Prediction_Project
   ```
2. Tạo và kích hoạt môi trường ảo (Virtual Environment):
   ```bash
   # Windows (Mở Terminal / PowerShell)
   python -m venv venv
   .\venv\Scripts\activate
   
   # MacOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
4. Chạy luồng chuẩn bị dữ liệu (Data Pipeline):
   ```bash
   # Bước 1: Tải dữ liệu thô (Mất khoảng vài phút tùy mạng)
   python src/data/download_dataset.py
   
   # Bước 2: Tính toán các chỉ báo định lượng (RSI, MACD, v.v...)
   python src/features/build_features.py
   ```
   *Lưu ý: Dữ liệu (file CSV, NPY) sẽ KHÔNG được đẩy lên Github để tránh làm nặng repo. Mỗi máy tính tự chạy lệnh trên để sinh ra dữ liệu nhé!*
