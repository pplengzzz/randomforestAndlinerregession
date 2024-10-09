import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# -------------------------------
# ฟังก์ชันสำหรับการทำความสะอาดข้อมูล
# -------------------------------
def clean_data(df):
    data_clean = df.copy()
    if 'datetime' not in data_clean.columns:
        st.error("ไม่มีคอลัมน์ 'datetime' ในไฟล์ CSV")
        return pd.DataFrame()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    if 'wl_up' not in data_clean.columns:
        st.error("ไม่มีคอลัมน์ 'wl_up' ในไฟล์ CSV")
        return pd.DataFrame()
    data_clean = data_clean[(data_clean['wl_up'] >= 100) & (data_clean['wl_up'] <= 450)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    return data_clean

# -------------------------------
# ฟังก์ชันสำหรับการสร้างฟีเจอร์เวลา
# -------------------------------
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

# -------------------------------
# ฟังก์ชันสำหรับการเตรียมฟีเจอร์
# -------------------------------
def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

# -------------------------------
# ฟังก์ชันสำหรับการฝึกและประเมินโมเดล
# -------------------------------
def train_and_evaluate_model(X, y, model_type='linear_regression'):
    # ใช้ TimeSeriesSplit สำหรับข้อมูลลำดับเวลา
    tscv = TimeSeriesSplit(n_splits=5)
    
    if model_type == 'linear_regression':
        model = LinearRegression()
    else:
        st.error("โมเดลที่เลือกไม่ถูกต้อง")
        return None
    
    # ฝึกโมเดลด้วยการ cross-validation
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        mae = mean_absolute_error(y_test_cv, y_pred_cv)
        scores.append(mae)
    
    st.write(f"Mean Absolute Error (MAE) จาก Cross-Validation: {np.mean(scores):.2f}")
    
    # ฝึกโมเดลด้วยข้อมูลทั้งหมด
    model.fit(X, y)
    return model

# -------------------------------
# ฟังก์ชันสำหรับการสร้างช่วงวันที่ครบถ้วน
# -------------------------------
def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    data_with_all_dates['datetime'] = pd.to_datetime(data_with_all_dates['datetime'], errors='coerce').dt.tz_localize(None)
    return data_with_all_dates

# -------------------------------
# ฟังก์ชันสำหรับการเติมคอลัมน์ 'code'
# -------------------------------
def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

# -------------------------------
# ฟังก์ชันสำหรับการจัดการกับค่าที่หายไป
# -------------------------------
def handle_missing_values(data_clean, model, feature_cols):
    data = data_clean.copy()
    data['wl_forecast'] = np.nan
    forecasted_indices = data[data['wl_up'].isnull()].index

    for idx in forecasted_indices:
        row = data.loc[idx, feature_cols].values.reshape(1, -1)
        predicted_value = model.predict(row)[0]
        data.at[idx, 'wl_forecast'] = predicted_value
        data.at[idx, 'wl_up'] = predicted_value  # เติมค่าพยากรณ์ลงใน 'wl_up' เพื่อใช้ในการพยากรณ์ครั้งถัดไป

    data['wl_up2'] = data['wl_up'].combine_first(data['wl_forecast'])
    return data

# -------------------------------
# ฟังก์ชันสำหรับการลบข้อมูลตามช่วงวันที่
# -------------------------------
def delete_data_by_date_range(data, delete_start_date, delete_end_date):
    delete_start_date = pd.to_datetime(delete_start_date)
    delete_end_date = pd.to_datetime(delete_end_date)

    data_to_delete = data[(data['datetime'] >= delete_start_date) & (data['datetime'] <= delete_end_date)]

    if len(data_to_delete) == 0:
        st.warning(f"ไม่พบข้อมูลระหว่าง {delete_start_date} และ {delete_end_date}.")
    elif len(data_to_delete) > (0.3 * len(data)):
        st.warning("คำเตือน: มีข้อมูลมากเกินไปที่จะลบ การดำเนินการลบถูกยกเลิก")
    else:
        data.loc[data_to_delete.index, 'wl_up'] = np.nan

    return data

# -------------------------------
# ฟังก์ชันสำหรับการคำนวณค่าความแม่นยำ
# -------------------------------
def calculate_error_metrics(original, forecasted):
    """
    คำนวณค่า MAE และ RMSE ระหว่างข้อมูลจริงและข้อมูลที่พยากรณ์
    Args:
        original (pd.DataFrame): ข้อมูลจริงที่มีคอลัมน์ 'datetime' และ 'wl_up'
        forecasted (pd.DataFrame): ข้อมูลพยากรณ์ที่มี index เป็น 'datetime' และคอลัมน์ 'wl_up'
    Returns:
        mae (float): ค่า Mean Absolute Error
        rmse (float): ค่า Root Mean Squared Error
        actual_forecasted_data (pd.DataFrame): ข้อมูลจริงที่ตรงกับช่วงเวลาการพยากรณ์
    """
    # เปลี่ยน index ของ forecasted เป็น 'datetime' หากยังไม่ใช่
    if forecasted.index.name != 'datetime':
        forecasted = forecasted.reset_index().rename(columns={'index': 'datetime'})

    # รวมข้อมูลจริงและพยากรณ์
    merged = pd.merge(original[['datetime', 'wl_up']], forecasted[['datetime', 'wl_up']], on='datetime', how='inner', suffixes=('_actual', '_forecasted'))

    # ลบแถวที่มีค่า NaN
    merged = merged.dropna(subset=['wl_up_actual', 'wl_up_forecasted'])

    if merged.empty:
        st.warning("ไม่มีข้อมูลที่ตรงกันสำหรับการคำนวณค่าความแม่นยำ")
        return None, None, None

    # คำนวณ MAE และ RMSE
    mae = mean_absolute_error(merged['wl_up_actual'], merged['wl_up_forecasted'])
    rmse = mean_squared_error(merged['wl_up_actual'], merged['wl_up_forecasted'], squared=False)

    # คืนค่าข้อมูลจริงที่ใช้ในการเปรียบเทียบ
    actual_forecasted_data = merged[['datetime', 'wl_up_actual', 'wl_up_forecasted']].copy()
    actual_forecasted_data.rename(columns={'wl_up_actual': 'Actual', 'wl_up_forecasted': 'Forecasted'}, inplace=True)

    return mae, rmse, actual_forecasted_data

# -------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression (สถานีเดียว)
# -------------------------------
def forecast_with_linear_regression_single(data, forecast_start_date):
    # ใช้ข้อมูลย้อนหลัง 3 วันในการเทรนโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # สร้างชุดข้อมูลสำหรับการเทรน
    training_data = data.loc[training_data_start:training_data_end].copy()

    # สร้างฟีเจอร์ lag
    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag
    training_data.dropna(inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้างฟีเจอร์ lag
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # สร้าง DataFrame สำหรับการพยากรณ์
    forecast_periods = 96  # พยากรณ์ 1 วัน (96 ช่วงเวลา 15 นาที)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # สร้างชุดข้อมูลสำหรับการพยากรณ์
    combined_data = data.copy()

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_value = combined_data.at[lag_time, 'wl_up']
            else:
                # ถ้าไม่มีค่า lag ให้ใช้ค่าเฉลี่ยของ y_train
                lag_value = y_train.mean()
            lag_features[f'lag_{lag}'] = lag_value

        X_pred = pd.DataFrame([lag_features])
        forecast_value = model.predict(X_pred)[0]
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value

    return forecasted_data

# -------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression (สองสถานี)
# -------------------------------
def forecast_with_linear_regression_two(data, upstream_data, forecast_start_date, delay_hours):
    # เตรียมข้อมูลจาก upstream_data
    if not upstream_data.empty:
        upstream_data = upstream_data.copy()
        if delay_hours > 0:
            upstream_data.index = upstream_data.index + pd.Timedelta(hours=delay_hours)

    # ใช้ข้อมูล 3 วันจากทั้งสองสถานีในการเทรนโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # สร้างชุดข้อมูลสำหรับการเทรน
    training_data = data.loc[training_data_start:training_data_end].copy()
    if not upstream_data.empty:
        training_data = training_data.join(upstream_data[['wl_up']], rsuffix='_upstream')

    # สร้างฟีเจอร์ lag
    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        if not upstream_data.empty:
            training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag
    if not upstream_data.empty:
        training_data.dropna(subset=[f'lag_{lag}' for lag in lags] + [f'lag_{lag}_upstream' for lag in lags], inplace=True)
    else:
        training_data.dropna(subset=[f'lag_{lag}' for lag in lags], inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้างฟีเจอร์ lag
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    if not upstream_data.empty:
        feature_cols = [f'lag_{lag}' for lag in lags] + [f'lag_{lag}_upstream' for lag in lags]
    else:
        feature_cols = [f'lag_{lag}' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # สร้าง DataFrame สำหรับการพยากรณ์
    forecast_periods = 96  # พยากรณ์ 1 วัน (96 ช่วงเวลา 15 นาที)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # สร้างชุดข้อมูลสำหรับการพยากรณ์
    combined_data = data.copy()
    if not upstream_data.empty:
        combined_upstream = upstream_data.copy()

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ค่า lag ของสถานีหลัก
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_value = combined_data.at[lag_time, 'wl_up']
            else:
                # ถ้าไม่มีค่า lag ให้ใช้ค่าเฉลี่ยของ y_train
                lag_value = y_train.mean()
            lag_features[f'lag_{lag}'] = lag_value

            # ค่า lag ของ upstream
            if not upstream_data.empty:
                if lag_time in combined_upstream.index and not pd.isnull(combined_upstream.at[lag_time, 'wl_up']):
                    lag_value_upstream = combined_upstream.at[lag_time, 'wl_up']
                else:
                    lag_value_upstream = y_train.mean()
                lag_features[f'lag_{lag}_upstream'] = lag_value_upstream

        X_pred = pd.DataFrame([lag_features])
        forecast_value = model.predict(X_pred)[0]
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' และ 'combined_upstream' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value
        if not upstream_data.empty:
            combined_upstream.at[idx, 'wl_up'] = lag_features.get(f'lag_{lag}_upstream', y_train.mean())

    return forecasted_data

# -------------------------------
# ฟังก์ชันสำหรับการแสดงกราฟข้อมูลพร้อมการพยากรณ์
# -------------------------------
def plot_data_combined(original_data, forecasted=None, label='ระดับน้ำ'):
    fig = go.Figure()

    # เพิ่มกราฟค่าจริง
    fig.add_trace(go.Scatter(
        x=original_data['datetime'],
        y=original_data['wl_up'],
        mode='lines',
        name='ค่าจริง',
        line=dict(color='blue')
    ))

    # เพิ่มกราฟค่าที่พยากรณ์
    if forecasted is not None and not forecasted.empty:
        fig.add_trace(go.Scatter(
            x=forecasted.index,
            y=forecasted['wl_up'],
            mode='lines',
            name='ค่าที่พยากรณ์',
            line=dict(color='red')
        ))

    fig.update_layout(
        title=f'ระดับน้ำที่สถานี {label}',
        xaxis_title="วันที่",
        yaxis_title="ระดับน้ำ (wl_up)",
        legend_title="ประเภทข้อมูล",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="การพยากรณ์ระดับน้ำ",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
# การพยากรณ์ระดับน้ำ

แอป Streamlit สำหรับจัดการข้อมูลระดับน้ำ โดยใช้โมเดล **Linear Regression** เพื่อเติมค่าที่ขาดหายไปและพยากรณ์ข้อมูล
ข้อมูลถูกประมวลผลและแสดงผลผ่านกราฟและการวัดค่าความแม่นยำ ผู้ใช้สามารถเลือกอัปโหลดไฟล์, 
กำหนดช่วงเวลาลบข้อมูล และเลือกวิธีการพยากรณ์ได้
""")
st.markdown("---")

# Sidebar: Upload files and choose date ranges
with st.sidebar:

    st.sidebar.title("เลือกวิธีการพยากรณ์")
    with st.sidebar.expander("ตั้งค่าโมเดล", expanded=True):
        model_choice = st.sidebar.radio("", ("Linear Regression",))

    st.sidebar.title("ตั้งค่าข้อมูล")

    if model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_upstream = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)

            if use_upstream:
                uploaded_up_file = st.file_uploader("อัปโหลดไฟล์ CSV ของสถานีข้างบน (upstream)", type="csv", key="uploader_up_lr")
                delay_hours = st.number_input("ระบุชั่วโมงหน่วงเวลาสำหรับการเชื่อมโยงข้อมูลจากสถานี upstream", value=0, min_value=0)

            uploaded_fill_file = st.file_uploader("อัปโหลดไฟล์ CSV สำหรับเติมข้อมูล", type="csv", key="uploader_fill_lr")

        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับพยากรณ์", expanded=False):
            forecast_start_date = st.date_input("วันที่เริ่มต้นพยากรณ์", value=pd.to_datetime("2024-06-01"), key='forecast_start_lr')
            forecast_start_time = st.time_input("เวลาเริ่มต้นพยากรณ์", value=pd.Timestamp("00:00:00").time(), key='forecast_start_time_lr')
            forecast_end_date = st.date_input("วันที่สิ้นสุดพยากรณ์", value=pd.to_datetime("2024-06-02"), key='forecast_end_lr')
            forecast_end_time = st.time_input("เวลาสิ้นสุดพยากรณ์", value=pd.Timestamp("23:45:00").time(), key='forecast_end_time_lr')

        process_button2 = st.button("ประมวลผล", type="primary")

# -------------------------------
# Main content: Display results after file uploads and date selection
# -------------------------------
if model_choice == "Linear Regression":
    if uploaded_fill_file:
        # โหลดข้อมูลของสถานีที่ต้องการทำนาย
        try:
            target_df = pd.read_csv(uploaded_fill_file)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
            target_df = pd.DataFrame()

        if target_df.empty:
            st.error("ไฟล์ CSV สำหรับเติมข้อมูลว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
        else:
            target_df = clean_data(target_df)
            if target_df.empty:
                st.error("หลังจากการทำความสะอาดข้อมูลแล้วไม่มีข้อมูลที่เหลือ")
            else:
                target_df = generate_missing_dates(target_df)
                target_df['datetime'] = pd.to_datetime(target_df['datetime'], errors='coerce').dt.tz_localize(None)  # แปลงเป็น timezone-naive
                target_df = create_time_features(target_df)
                target_df['wl_up_prev'] = target_df['wl_up'].shift(1)
                target_df['wl_up_prev'] = target_df['wl_up_prev'].interpolate(method='linear')

                # โหลดข้อมูลสถานีใกล้เคียงถ้าเลือกใช้
                if use_upstream and uploaded_up_file:
                    try:
                        upstream_df = pd.read_csv(uploaded_up_file)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์สถานีข้างบน: {e}")
                        upstream_df = pd.DataFrame()

                    if upstream_df.empty:
                        st.error("ไฟล์ CSV สถานีใกล้เคียงว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
                        upstream_df = pd.DataFrame()
                    else:
                        upstream_df = clean_data(upstream_df)
                        if upstream_df.empty:
                            st.error("หลังจากการทำความสะอาดข้อมูลสถานีใกล้เคียงแล้วไม่มีข้อมูลที่เหลือ")
                            upstream_df = pd.DataFrame()
                        else:
                            upstream_df = generate_missing_dates(upstream_df)
                            upstream_df['datetime'] = pd.to_datetime(upstream_df['datetime'], errors='coerce').dt.tz_localize(None)  # แปลงเป็น timezone-naive
                            upstream_df = create_time_features(upstream_df)
                            upstream_df['wl_up_prev'] = upstream_df['wl_up'].shift(1)
                            upstream_df['wl_up_prev'] = upstream_df['wl_up_prev'].interpolate(method='linear')
                else:
                    upstream_df = None

                # แสดงกราฟข้อมูล
                st.subheader('กราฟข้อมูลระดับน้ำ')
                plot_data_combined(target_df, label='สถานีที่ต้องการทำนาย')
                if upstream_df is not None and not upstream_df.empty:
                    plot_data_combined(upstream_df, label='สถานีใกล้เคียง (up)')
                else:
                    st.info("ไม่มีข้อมูลสถานีใกล้เคียง")

                if process_button2:
                    with st.spinner("กำลังพยากรณ์..."):
                        start_datetime = pd.Timestamp.combine(forecast_start_date, forecast_start_time)
                        end_datetime = pd.Timestamp.combine(forecast_end_date, forecast_end_time)

                        if start_datetime > end_datetime:
                            st.error("วันและเวลาที่เริ่มต้นต้องไม่เกินวันและเวลาสิ้นสุด")
                        else:
                            selected_data = target_df[(target_df['datetime'] >= start_datetime) & (target_df['datetime'] <= end_datetime)].copy()

                            if selected_data.empty:
                                st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
                            else:
                                forecast_start_date_actual = selected_data['datetime'].max() + pd.Timedelta(minutes=15)

                                # เตรียมข้อมูลสำหรับการพยากรณ์
                                X, y = prepare_features(target_df)

                                # ฝึกโมเดล Linear Regression ด้วยข้อมูลจริง
                                model = train_and_evaluate_model(X, y, model_type='linear_regression')

                                if model is not None:
                                    # พยากรณ์ด้วย Linear Regression (สถานีเดียว)
                                    if not use_upstream or upstream_df.empty:
                                        forecasted_data = forecast_with_linear_regression_single(
                                            data=target_df.set_index('datetime'),
                                            forecast_start_date=forecast_start_date_actual
                                        )
                                    else:
                                        # พยากรณ์ด้วย Linear Regression (สองสถานี)
                                        forecasted_data = forecast_with_linear_regression_two(
                                            data=target_df.set_index('datetime'),
                                            upstream_data=upstream_df.set_index('datetime'),
                                            forecast_start_date=forecast_start_date_actual,
                                            delay_hours=delay_hours
                                        )

                                    if not forecasted_data.empty:
                                        # แสดงกราฟข้อมูลพร้อมการพยากรณ์
                                        st.subheader('กราฟข้อมูลพร้อมการพยากรณ์')
                                        plot_data_combined(
                                            original_data=target_df,
                                            forecasted=forecasted_data,
                                            label='สถานีที่ต้องการทำนาย'
                                        )

                                        # ตรวจสอบและคำนวณค่าความแม่นยำ
                                        mae, rmse, actual_forecasted_data = calculate_error_metrics(
                                            original=target_df,
                                            forecasted=forecasted_data
                                        )

                                        if actual_forecasted_data is not None:
                                            st.subheader('ตารางข้อมูลเปรียบเทียบ')
                                            comparison_table = pd.DataFrame({
                                                'Datetime': actual_forecasted_data['datetime'],
                                                'ค่าจริง': actual_forecasted_data['Actual'],
                                                'ค่าที่พยากรณ์': actual_forecasted_data['Forecasted']
                                            })
                                            st.dataframe(comparison_table)

                                            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                                        else:
                                            st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                                    else:
                                        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
                                else:
                                    st.error("ไม่สามารถฝึกโมเดลได้ กรุณาตรวจสอบข้อมูล")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV สำหรับเติมข้อมูล เพื่อเริ่มต้นการพยากรณ์")











