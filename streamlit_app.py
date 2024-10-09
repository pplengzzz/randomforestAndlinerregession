import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
import plotly.express as px

# ------------------------ ฟังก์ชันสำหรับ Random Forest ------------------------

def load_data(file):
    message_placeholder = st.empty()  # สร้างตำแหน่งที่ว่างสำหรับข้อความแจ้งเตือน
    if file is None:
        st.error("ไม่มีไฟล์ที่อัปโหลด กรุณาอัปโหลดไฟล์ CSV")
        return None
    
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("ไฟล์ CSV ว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
            return None
        message_placeholder.success("ไฟล์ถูกโหลดเรียบร้อยแล้ว")  # แสดงข้อความในตำแหน่งที่ว่าง
        return df
    except pd.errors.EmptyDataError:
        st.error("ไม่สามารถอ่านข้อมูลจากไฟล์ได้ ไฟล์อาจว่างเปล่าหรือไม่ใช่ไฟล์ CSV ที่ถูกต้อง")
        return None
    except pd.errors.ParserError:
        st.error("เกิดข้อผิดพลาดในการแยกวิเคราะห์ไฟล์ CSV กรุณาตรวจสอบรูปแบบของไฟล์")
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")
        return None
    finally:
        message_placeholder.empty()  # ลบข้อความแจ้งเตือนเมื่อเสร็จสิ้นการโหลดไฟล์

def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= 100) & (data_clean['wl_up'] <= 450)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    return data_clean

def create_time_features(data_clean):
    if not pd.api.types.is_datetime64_any_dtype(data_clean['datetime']):
        data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')

    data_clean['year'] = data_clean['datetime'].dt.year
    data_clean['month'] = data_clean['datetime'].dt.month
    data_clean['day'] = data_clean['datetime'].dt.day
    data_clean['hour'] = data_clean['datetime'].dt.hour
    data_clean['minute'] = data_clean['datetime'].dt.minute
    data_clean['day_of_week'] = data_clean['datetime'].dt.dayofweek
    data_clean['day_of_year'] = data_clean['datetime'].dt.dayofyear
    data_clean['week_of_year'] = data_clean['datetime'].dt.isocalendar().week
    data_clean['days_in_month'] = data_clean['datetime'].dt.days_in_month

    return data_clean

def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

def train_and_evaluate_model(X, y, model_type='random_forest'):
    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ฝึกโมเดลด้วยชุดฝึก
    if model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
    elif model_type == 'linear_regression':
        model = train_linear_regression_model(X_train, y_train)
    else:
        st.error("โมเดลที่เลือกไม่ถูกต้อง")
        return None

    # ตรวจสอบว่าฝึกโมเดลสำเร็จหรือไม่
    if model is None:
        st.error("การฝึกโมเดลล้มเหลว")
        return None
    return model

def train_random_forest(X_train, y_train):
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring='neg_mean_absolute_error'
    )
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

# ------------------------ ฟังก์ชันสำหรับ Linear Regression ------------------------

# ฟังก์ชันสำหรับการฝึก Linear Regression
def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

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

# ฟังก์ชัน Linear Regression สำหรับสถานีตัวเอง
def forecast_with_self_station(data, start_date, end_date):
    training_data_end = pd.to_datetime(end_date)
    training_data_start = pd.to_datetime(start_date)

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

    # พยากรณ์ล่วงหน้า
    forecast_periods = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() // 900
    forecast_index = pd.date_range(start=start_date, periods=int(forecast_periods), freq='15T')
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

# ฟังก์ชัน Linear Regression สำหรับสถานีข้างบน
def forecast_with_upstream_station(data, upstream_data, start_date, end_date, delay_hours):
    upstream_data = shift_upstream_data(upstream_data, delay_hours)

    training_data_end = pd.to_datetime(end_date)
    training_data_start = pd.to_datetime(start_date)

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

    # พยากรณ์ล่วงหน้า
    forecast_periods = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() // 900
    forecast_index = pd.date_range(start=start_date, periods=int(forecast_periods), freq='15T')
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

st.title("การพยากรณ์ระดับน้ำด้วย Random Forest และ Linear Regression")

# Sidebar สำหรับอัปโหลดไฟล์และเลือกโมเดล
model_choice = st.sidebar.radio("เลือกโมเดล", ("Random Forest", "Linear Regression"))
st.sidebar.header("อัปโหลดข้อมูล")

# ------------------ ส่วนของ Random Forest ------------------
if model_choice == "Random Forest":
    uploaded_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV", type="csv")
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("ข้อมูลที่โหลดมา:")
            st.write(df.head())
            df_clean = clean_data(df)
            X, y = prepare_features(df_clean)
            start_date = st.sidebar.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-06-01"))
            end_date = st.sidebar.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-06-30"))
            
            model_rf = train_random_forest(X, y)
            st.success("Random Forest เทรนเรียบร้อยแล้ว")

# ------------------ ส่วนของ Linear Regression ------------------
elif model_choice == "Linear Regression":
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
        start_date = st.sidebar.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-06-01"))
        end_date = st.sidebar.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-06-30"))

        if st.sidebar.button("ประมวลผลข้อมูล"):
            if use_upstream and upstream_data is not None:
                forecasted_data = forecast_with_upstream_station(target_data, upstream_data, start_date, end_date, delay_hours)
                plot_data(target_data, forecasted=forecasted_data, title="การพยากรณ์ระดับน้ำโดยใช้ข้อมูลจากสถานีข้างบน")
            else:
                forecasted_data = forecast_with_self_station(target_data, start_date, end_date)
                plot_data(target_data, forecasted=forecasted_data, title="การพยากรณ์ระดับน้ำโดยใช้ข้อมูลจากสถานีตัวเอง")



