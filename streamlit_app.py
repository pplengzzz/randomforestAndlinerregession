import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px

# ฟังก์ชันสำหรับการโหลดและทำความสะอาดข้อมูล
def load_and_clean_data(file):
    try:
        data = pd.read_csv(file)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        data = data[data['wl_up'] >= 100]  # ลบข้อมูลที่มีค่า wl_up ต่ำกว่า 100
        return data
    except Exception as e:
        st.error(f"ไม่สามารถโหลดข้อมูลได้: {e}")
        return None

# ------------------------ Linear Regression: ใช้สถานีตัวเอง (Colab ข้อ 1) ------------------------
def forecast_with_self_station(data, forecast_start_date):
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    if training_data_start < data.index.min():
        st.error("ข้อมูลไม่เพียงพอสำหรับการเทรน")
        return pd.DataFrame()

    # สร้างฟีเจอร์ lag
    training_data = data.loc[training_data_start:training_data_end].copy()
    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    training_data.dropna(inplace=True)

    if training_data.empty:
        st.error("ข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    X_train = training_data[[f'lag_{lag}' for lag in lags]]
    y_train = training_data['wl_up']

    model = LinearRegression()
    model.fit(X_train, y_train)

    # พยากรณ์ล่วงหน้า 1 วัน (96 ช่วงเวลา, 15 นาทีต่อครั้ง)
    forecast_periods = 96
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
            else:
                lag_value = np.nan
            lag_features[f'lag_{lag}'] = lag_value

        X_pred = pd.DataFrame([lag_features])
        if not X_pred.isnull().values.any():
            forecast_value = model.predict(X_pred)[0]
            forecasted_data.at[idx, 'wl_up'] = forecast_value

    forecasted_data.dropna(inplace=True)
    return forecasted_data

# ------------------------ Linear Regression: ใช้สถานีข้างบน (Colab ข้อ 2) ------------------------
def forecast_with_upstream_station(data, upstream_data, forecast_start_date, delay_hours):
    upstream_data = shift_upstream_data(upstream_data, delay_hours)

    # เทรนโมเดลด้วยข้อมูลย้อนหลัง 3 วัน
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    if training_data_start < data.index.min():
        st.error("ข้อมูลไม่เพียงพอสำหรับการเทรน")
        return pd.DataFrame()

    # รวมข้อมูลสถานีข้างบนและเลื่อนเวลา
    training_data = data.loc[training_data_start:training_data_end].copy()
    training_data = training_data.join(upstream_data, rsuffix='_upstream')

    # สร้างฟีเจอร์ lag
    lags = [1, 4, 96, 192]
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)

    training_data.dropna(inplace=True)

    if training_data.empty:
        st.error("ข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    X_train = training_data[[f'lag_{lag}' for lag in lags] + [f'lag_{lag}_upstream' for lag in lags]]
    y_train = training_data['wl_up']

    model = LinearRegression()
    model.fit(X_train, y_train)

    # พยากรณ์ล่วงหน้า 1 วัน (96 ช่วงเวลา, 15 นาทีต่อครั้ง)
    forecast_periods = 96
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
                lag_value_upstream = upstream_data.at[lag_time, 'wl_up_upstream'] if lag_time in upstream_data.index else np.nan
            else:
                lag_value = np.nan
                lag_value_upstream = np.nan

            lag_features[f'lag_{lag}'] = lag_value
            lag_features[f'lag_{lag}_upstream'] = lag_value_upstream

        X_pred = pd.DataFrame([lag_features])
        if not X_pred.isnull().values.any():
            forecast_value = model.predict(X_pred)[0]
            forecasted_data.at[idx, 'wl_up'] = forecast_value

    forecasted_data.dropna(inplace=True)
    return forecasted_data

# ฟังก์ชันเลื่อนข้อมูลสถานีข้างบนตามชั่วโมงที่กำหนด
def shift_upstream_data(upstream_data, delay_hours):
    upstream_data_shifted = upstream_data.copy()
    upstream_data_shifted.index = upstream_data_shifted.index + pd.Timedelta(hours=delay_hours)
    return upstream_data_shifted

# ฟังก์ชันแสดงผลกราฟ
def plot_data(data, forecasted=None, title="กราฟแสดงระดับน้ำ"):
    fig = px.line(data, x=data.index, y='wl_up', title=title, labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    st.plotly_chart(fig)

# ---------------------------------- Main Streamlit App ----------------------------------

st.title("การพยากรณ์ระดับน้ำด้วย Linear Regression")

# Sidebar สำหรับอัปโหลดไฟล์และเลือกการทำงาน
uploaded_target_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV สำหรับสถานีที่ต้องการทำนาย", type="csv")
use_upstream = st.sidebar.checkbox("ใช้ข้อมูลจากสถานีข้างบน", value=False)

if use_upstream:
    uploaded_upstream_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV สำหรับสถานีข้างบน", type="csv")
    delay_hours = st.sidebar.slider("ระบุเวลาล่าช้า (ชั่วโมง)", 0, 48, 2)

if uploaded_target_file:
    target_data = load_and_clean_data(uploaded_target_file)

    if use_upstream and uploaded_upstream_file:
        upstream_data = load_and_clean_data(uploaded_upstream_file)
    else:
        upstream_data = None

    # เลือกช่วงวันที่สำหรับพยากรณ์
    st.sidebar.header("เลือกช่วงวันที่สำหรับพยากรณ์")
    start_date = st.sidebar.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-06-01"))

    if st.sidebar.button("ประมวลผลข้อมูล"):
        forecast_start_date = pd.to_datetime(start_date) + pd.Timedelta(hours=24)

        if use_upstream and upstream_data is not None:
            forecasted_data = forecast_with_upstream_station(target_data, upstream_data, forecast_start_date, delay_hours)
            plot_data(target_data, forecasted=forecasted_data, title="การพยากรณ์ระดับน้ำโดยใช้ข้อมูลจากสถานีข้างบน")
        else:
            forecasted_data = forecast_with_self_station(target_data, forecast_start_date)
            plot_data(target_data, forecasted=forecasted_data, title="การพยากรณ์ระดับน้ำโดยใช้ข้อมูลจากสถานีตัวเอง")

