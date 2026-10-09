import yfinance as yf
import pandas as pd
import os
from pathlib import Path

# Xác định đường dẫn tuyệt đối đến thư mục data/raw
# File này đang ở src/data/, nên ta phải lùi lại 2 cấp để về thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def download_stock_data(ticker, start_date, end_date):
    print(f"Đang tải dữ liệu cho mã {ticker} từ {start_date} đến {end_date}...")
    
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print(f"Không tìm thấy dữ liệu cho {ticker}.")
        return None

    # Reset index để cột 'Date' trở thành một cột bình thường thay vì index
    df.reset_index(inplace=True)
    
    # Làm gọn tên cột (nếu bị MultiIndex do yfinance mới update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Định dạng lại cột Date để dễ nhìn
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    # Lưu ra file CSV vào thư mục data/raw
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_raw.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Đã lưu thành công: {file_path}")
    print(f"Số lượng dòng (ngày giao dịch): {len(df)}")
    print("-" * 50)
    
    return file_path

if __name__ == "__main__":
    TICKERS = ['FPT.VN', 'HPG.VN', 'VCB.VN', 'SSI.VN', 'VNM.VN', 'MWG.VN']
    START_DATE = '2018-01-01'
    END_DATE = '2025-01-01'
    
    # Đảm bảo thư mục raw tồn tại
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    for ticker in TICKERS:
        download_stock_data(ticker, START_DATE, END_DATE)
    
    print("HOÀN THÀNH QUÁ TRÌNH TẢI RAW DATA!")
