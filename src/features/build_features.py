import pandas as pd
import numpy as np
import ta
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
INTERIM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'interim')

def add_technical_indicators(df):
    # Thêm các chỉ báo phân tích kỹ thuật (Technical Indicators) vào dữ liệu.
    # Sử dụng thư viện 'ta'.
    # Xử lý giá trị bị thiếu (nếu có) bằng cách lấy giá ngày hôm trước đắp vào
    df = df.ffill().bfill()
    
    # Sắp xếp đúng thứ tự thời gian
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Trend Indicators (Chỉ báo xu hướng)
    # Simple Moving Average (SMA)
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Momentum Indicators (Chỉ báo động lượng)
    # RSI (Relative Strength Index)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Volatility Indicators (Độ biến động)
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()
    
    # ATR (Average True Range)
    df['ATR'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'])
    
    # Price Action (Hành vi giá)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Spread'] = df['High'] - df['Low']
    
    # LƯU Ý QUAN TRỌNG:
    # SMA_50 cần 50 ngày đầu tiên để lấy trung bình, nên 49 ngày đầu sẽ bị NaN (không có data).
    # Chúng ta bắt buộc phải drop (xóa) những ngày NaN này đi để không làm hỏng model.
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
    
    # Tìm tất cả các file trong thư mục raw
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('_raw.csv')]
    
    print(f"Tìm thấy {len(raw_files)} mã cổ phiếu. Đang bắt đầu tính toán kỹ thuật định lượng...")
    
    for file_name in raw_files:
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        df = pd.read_csv(file_path)
        
        # Thêm features bằng thuật toán định lượng
        df_featured = add_technical_indicators(df)
        
        # Lưu vào thư mục interim
        ticker = file_name.split('_')[0]
        output_name = f"{ticker}_features.csv"
        output_path = os.path.join(INTERIM_DATA_DIR, output_name)
        
        df_featured.to_csv(output_path, index=False)
        print(f"Đã xử lý xong: {ticker} -> Lưu tại {output_path} | Dữ liệu còn: {len(df_featured)} ngày")
        
    print("HOÀN THÀNH QUÁ TRÌNH FEATURE ENGINEERING!")
