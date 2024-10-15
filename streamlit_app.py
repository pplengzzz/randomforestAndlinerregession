import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
import altair as alt
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io

# ฟังก์ชันสำหรับรวมข้อมูล (ถ้าไม่มี การสร้างคอลัมน์ 'wl_up_prev' จาก 'wl_up' ของ df1)
def merge_data(df1, df2=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_prev'))
    else:
        # ถ้าไม่มี df2 ให้สร้างคอลัมน์ 'wl_up_prev' จาก 'wl_up' ของ df1 (shifted by 1)
        df1['wl_up_prev'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันสำหรับโหลดข้อมูล
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

# ฟังก์ชันสำหรับทำความสะอาดข้อมูล
def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    
    return data_clean

# ฟังก์ชันสำหรับสร้างฟีเจอร์เวลา
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

# ฟังก์ชันสำหรับเตรียมฟีเจอร์และตัวแปรเป้าหมาย
def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

# ฟังก์ชันสำหรับฝึกและประเมินโมเดล
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

# ฟังก์ชันสำหรับฝึกโมเดล Random Forest
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

# ฟังก์ชันสำหรับฝึกโมเดล Linear Regression
def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ฟังก์ชันสำหรับสร้างวันที่ที่หายไป
def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

# ฟังก์ชันสำหรับเติมคอลัมน์ 'code' ถ้ามี
def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

# ฟังก์ชันสำหรับจัดการค่าที่หายไปโดยใช้โมเดล
def handle_missing_values_by_week(data_clean, start_date, end_date, model_type='random_forest'):
    feature_cols = ['year', 'month', 'day', 'hour', 'minute',
                    'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month', 'wl_up_prev']

    data = data_clean.copy()

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.DateOffset(hours=23, minutes=45)

    # Filter data based on the datetime range
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    # Generate all missing dates within the selected range
    data_with_all_dates = generate_missing_dates(data)
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates['datetime'])
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    # เติมค่า missing ใน wl_up_prev
    if 'wl_up_prev' in data_with_all_dates.columns:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up_prev'].interpolate(method='linear')
    else:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up'].shift(1).interpolate(method='linear')

    if len(data_missing) == 0:
        st.write("No missing values to predict.")
        return data_with_all_dates

    # Train initial model with all available data
    X_train, y_train = prepare_features(data_not_missing)
    model = train_and_evaluate_model(X_train, y_train, model_type=model_type)

    # ตรวจสอบว่ามีโมเดลที่ถูกฝึกหรือไม่
    if model is None:
        st.error("ไม่สามารถสร้างโมเดลได้ กรุณาตรวจสอบข้อมูล")
        return data_with_all_dates

    # Fill missing values
    for idx, row in data_missing.iterrows():
        X_missing = row[feature_cols].values.reshape(1, -1)
        try:
            predicted_value = model.predict(X_missing)[0]
            # บันทึกค่าที่เติมในคอลัมน์ wl_forecast และ timestamp
            data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
            data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในแถว {idx} ได้: {e}")
            continue

    # สร้างคอลัมน์ wl_up2 ที่รวมข้อมูลเดิมกับค่าที่เติม
    data_with_all_dates['wl_up2'] = data_with_all_dates['wl_up'].combine_first(data_with_all_dates['wl_forecast'])

    data_with_all_dates.reset_index(drop=True, inplace=True)
    return data_with_all_dates

# ฟังก์ชันสำหรับลบข้อมูลตามช่วงวันที่ที่กำหนด
def delete_data_by_date_range(data, delete_start_date, delete_end_date):
    # Convert delete_start_date and delete_end_date to datetime
    delete_start_date = pd.to_datetime(delete_start_date)
    delete_end_date = pd.to_datetime(delete_end_date)

    # ตรวจสอบว่าช่วงวันที่ต้องการลบข้อมูลอยู่ในช่วงของ data หรือไม่
    data_to_delete = data[(data['datetime'] >= delete_start_date) & (data['datetime'] <= delete_end_date)]

    # เพิ่มการตรวจสอบว่าถ้าจำนวนข้อมูลที่ถูกลบมีมากเกินไป
    if len(data_to_delete) == 0:
        st.warning(f"ไม่พบข้อมูลระหว่าง {delete_start_date} และ {delete_end_date}.")
    elif len(data_to_delete) > (0.3 * len(data)):  # ตรวจสอบว่าถ้าลบเกิน 30% ของข้อมูล
        st.warning("คำเตือน: มีข้อมูลมากเกินไปที่จะลบ การดำเนินการลบถูกยกเลิก")
    else:
        # ลบข้อมูลโดยตั้งค่า wl_up เป็น NaN
        data.loc[data_to_delete.index, 'wl_up'] = np.nan

    return data

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำ
def calculate_accuracy_metrics(original, filled):
    # ผสานข้อมูลตาม datetime
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')

    # ลบข้อมูลที่มี NaN ออก
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(merged_data['wl_up'], merged_data['wl_up2'])
    mae = mean_absolute_error(merged_data['wl_up'], merged_data['wl_up2'])
    r2 = r2_score(merged_data['wl_up'], merged_data['wl_up2'])

    st.header("ผลค่าความแม่นยำ", divider='gray')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

    return mse, mae, r2, merged_data

# ฟังก์ชันสำหรับสร้างตารางเปรียบเทียบ
def create_comparison_table_streamlit(forecasted_data, actual_data):
    comparison_df = pd.DataFrame({
        'Datetime': actual_data['datetime'],
        'ค่าจริง': actual_data['wl_up'],
        'ค่าที่พยากรณ์': actual_data['wl_up2']
    })
    return comparison_df

# ฟังก์ชันสำหรับสร้างกราฟข้อมูลพร้อมการพยากรณ์
def plot_results(data_before, data_filled, data_deleted, data_deleted_option=False):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
    })

    # เงื่อนไขในการสร้าง DataFrame สำหรับข้อมูลหลังลบ
    if data_deleted_option:
        data_after_deleted = pd.DataFrame({
            'วันที่': data_deleted['datetime'],
            'ข้อมูลหลังลบ': data_deleted['wl_up']
        })
    else:
        data_after_deleted = None

    # รวมข้อมูล
    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')

    if data_after_deleted is not None:
        combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    # กำหนดรายการ y ที่จะแสดงในกราฟ
    y_columns = ['ข้อมูลหลังเติมค่า', 'ข้อมูลเดิม']
    if data_after_deleted is not None:
        y_columns.append('ข้อมูลหลังลบ')

    # Plot ด้วย Plotly
    fig = px.line(combined_data, x='วันที่', y=y_columns,
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    # แสดงกราฟ
    st.header("ข้อมูลหลังจากการเติมค่าที่หายไป", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    data_filled_selected = data_filled[['datetime', 'wl_up', 'wl_forecast', 'timestamp']]
    st.dataframe(data_filled_selected, use_container_width=True)

    # ตรวจสอบว่ามีค่าจริงให้เปรียบเทียบหรือไม่ก่อนเรียกฟังก์ชันคำนวณความแม่นยำ
    merged_data = pd.merge(data_before[['datetime', 'wl_up']], data_filled[['datetime', 'wl_up2']], on='datetime')
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])
    comparison_data = merged_data[merged_data['wl_up2'] != merged_data['wl_up']]

    if comparison_data.empty:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีค่าจริงให้เปรียบเทียบ")
    else:
        calculate_accuracy_metrics(data_before, data_filled)

# ฟังก์ชันสำหรับสร้างกราฟข้อมูลจากสถานีทั้งสอง
def plot_data_combined_two_stations(data, forecasted=None, upstream_data=None, downstream_data=None, label='ระดับน้ำ'):
    fig = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    fig.update_traces(connectgaps=False)
    
    # แสดงค่าจริงของสถานีที่ต้องการพยากรณ์
    fig.add_scatter(x=data.index, y=data['wl_up'], mode='lines', name='ค่าจริง (สถานีที่พยากรณ์)', line=dict(color='blue'))
    
    # แสดงค่าจริงของสถานี upstream (ถ้ามี)
    if upstream_data is not None:
        fig.add_scatter(x=upstream_data.index, y=upstream_data['wl_up'], mode='lines', name='ค่าจริง (สถานี Upstream)', line=dict(color='green'))
    
    # แสดงค่าจริงของสถานี downstream (ถ้ามี)
    if downstream_data is not None:
        fig.add_scatter(x=downstream_data.index, y=downstream_data['wl_up'], mode='lines', name='ค่าจริง (สถานี Downstream)', line=dict(color='purple'))

    # แสดงค่าพยากรณ์
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression ทีละค่า (สถานีเดียว)
def forecast_with_linear_regression_single(data, forecast_start_date, forecast_days):
    # ตรวจสอบจำนวนวันที่พยากรณ์ให้อยู่ในขอบเขต 1-30 วัน
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    # กำหนดช่วงเวลาการฝึกโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = forecast_start_date - pd.Timedelta(days=30) + pd.Timedelta(minutes=15)  # ใช้ข้อมูลย้อนหลัง 30 วันในการเทรนโมเดล

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

    # เติมค่า NaN ด้วยค่าเฉลี่ยของ y_train ก่อนการเทรน
    training_data.fillna(method='ffill', inplace=True)
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
    forecast_periods = forecast_days * 96  # พยากรณ์ตามจำนวนวันที่เลือก (96 ช่วงเวลา 15 นาทีต่อวัน)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # สร้างชุดข้อมูลสำหรับการพยากรณ์
    combined_data = data.copy()

    # การพยากรณ์ทีละค่า
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

        # เตรียมข้อมูลสำหรับการพยากรณ์
        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred)[0]

        # ป้องกันการกระโดดของค่าพยากรณ์
        forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
        
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value

    return forecasted_data

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression สำหรับหลายสถานี (upstream และ downstream ถ้ามี)
def forecast_with_linear_regression_multi(data, forecast_start_date, forecast_days, upstream_data=None, downstream_data=None, delay_hours_up=0, delay_hours_down=0):
    # ตรวจสอบจำนวนวันที่พยากรณ์ให้อยู่ในขอบเขต 1-30 วัน
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    # เตรียมข้อมูลจาก upstream_data ถ้ามี
    if upstream_data is not None and not upstream_data.empty:
        upstream_data = upstream_data.copy()
        if delay_hours_up > 0:
            upstream_data.index = upstream_data.index + pd.Timedelta(hours=delay_hours_up)

    # เตรียมข้อมูลจาก downstream_data ถ้ามี
    if downstream_data is not None and not downstream_data.empty:
        downstream_data = downstream_data.copy()
        if delay_hours_down > 0:
            downstream_data.index = downstream_data.index + pd.Timedelta(hours=delay_hours_down)

    # กำหนดช่วงเวลาการฝึกโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = forecast_start_date - pd.Timedelta(days=30) + pd.Timedelta(minutes=15)  # ใช้ข้อมูลย้อนหลัง 30 วันในการเทรนโมเดล

    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # สร้างชุดข้อมูลสำหรับการเทรน
    training_data = data.loc[training_data_start:training_data_end].copy()

    # รวมข้อมูลจาก upstream และ downstream ถ้ามี
    if upstream_data is not None:
        training_data = training_data.join(upstream_data[['wl_up']], rsuffix='_upstream')
    if downstream_data is not None:
        training_data = training_data.join(downstream_data[['wl_up']], rsuffix='_downstream')

    # สร้างฟีเจอร์ lag
    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        if upstream_data is not None:
            training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)
        if downstream_data is not None:
            training_data[f'lag_{lag}_downstream'] = training_data['wl_up_downstream'].shift(lag)

    # เติมค่า NaN ด้วยค่าเฉลี่ยของ y_train ก่อนการเทรน
    training_data.fillna(method='ffill', inplace=True)
    training_data.dropna(inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้างฟีเจอร์ lag
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags]
    if upstream_data is not None:
        feature_cols += [f'lag_{lag}_upstream' for lag in lags]
    if downstream_data is not None:
        feature_cols += [f'lag_{lag}_downstream' for lag in lags]

    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # สร้าง DataFrame สำหรับการพยากรณ์
    forecast_periods = forecast_days * 96  # พยากรณ์ตามจำนวนวันที่เลือก (96 ช่วงเวลา 15 นาทีต่อวัน)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # สร้างชุดข้อมูลสำหรับการพยากรณ์
    combined_data = data.copy()
    if upstream_data is not None:
        combined_upstream = upstream_data.copy()
    if downstream_data is not None:
        combined_downstream = downstream_data.copy()

    # การพยากรณ์ทีละค่า
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ค่า lag ของสถานีหลัก
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_value = combined_data.at[lag_time, 'wl_up']
            else:
                lag_value = y_train.mean()
            lag_features[f'lag_{lag}'] = lag_value

            # ค่า lag ของ upstream
            if upstream_data is not None:
                if lag_time in combined_upstream.index and not pd.isnull(combined_upstream.at[lag_time, 'wl_up']):
                    lag_value_upstream = combined_upstream.at[lag_time, 'wl_up']
                else:
                    lag_value_upstream = y_train.mean()
                lag_features[f'lag_{lag}_upstream'] = lag_value_upstream

            # ค่า lag ของ downstream
            if downstream_data is not None:
                if lag_time in combined_downstream.index and not pd.isnull(combined_downstream.at[lag_time, 'wl_up']):
                    lag_value_downstream = combined_downstream.at[lag_time, 'wl_up']
                else:
                    lag_value_downstream = y_train.mean()
                lag_features[f'lag_{lag}_downstream'] = lag_value_downstream

        # เตรียมข้อมูลสำหรับการพยากรณ์
        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred)[0]

        # ป้องกันการกระโดดของค่าพยากรณ์
        forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
        
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' และสถานีอื่นๆ ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value
        if upstream_data is not None:
            combined_upstream.at[idx, 'wl_up'] = lag_features.get(f'lag_{lag}_upstream', y_train.mean())
        if downstream_data is not None:
            combined_downstream.at[idx, 'wl_up'] = lag_features.get(f'lag_{lag}_downstream', y_train.mean())

    return forecasted_data

# Streamlit UI
st.set_page_config(
    page_title="การพยากรณ์ระดับน้ำ",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
# การพยากรณ์ระดับน้ำ

แอป Streamlit สำหรับจัดการข้อมูลระดับน้ำ โดยใช้โมเดล **Linear Regression** และ **Random Forest** เพื่อเติมค่าที่ขาดหายไปและพยากรณ์ข้อมูล
ข้อมูลถูกประมวลผลและแสดงผลผ่านกราฟและการวัดค่าความแม่นยำ ผู้ใช้สามารถเลือกอัปโหลดไฟล์, 
กำหนดช่วงเวลาลบข้อมูล และเลือกวิธีการพยากรณ์ได้
""")
st.markdown("---")

# Sidebar: Upload files and choose date ranges
with st.sidebar:

    st.sidebar.title("เลือกวิธีการพยากรณ์")
    model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"),
        label_visibility="collapsed"  # ซ่อน label visibility
    )

    st.sidebar.title("ตั้งค่าข้อมูล")
    if model_choice == "Random Forest":
        with st.sidebar.expander("ตั้งค่า Random Forest", expanded=False):
            use_nearby_rf = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            # Checkbox สำหรับเลือกใช้ Upstream และ Downstream
            if use_nearby_rf:
                use_upstream_rf = st.checkbox("ใช้สถานี Upstream", value=True)
                use_downstream_rf = st.checkbox("ใช้สถานี Downstream", value=False)
            else:
                use_upstream_rf = False
                use_downstream_rf = False
            
            # อัปโหลดไฟล์
            if use_nearby_rf and use_upstream_rf:
                uploaded_up_rf = st.file_uploader("ข้อมูลสถานี Upstream", type="csv", key="uploader_up_rf")
            else:
                uploaded_up_rf = None

            if use_nearby_rf and use_downstream_rf:
                uploaded_down_rf = st.file_uploader("ข้อมูลสถานี Downstream", type="csv", key="uploader_down_rf")
            else:
                uploaded_down_rf = None

            # เพิ่มช่องกรอกเวลาห่างระหว่างสถานี
            # กำหนดค่าเริ่มต้นสำหรับ delay_hours_up_rf และ delay_hours_down_rf
            delay_hours_up_rf = 0
            delay_hours_down_rf = 0

            if use_nearby_rf:
                if use_upstream_rf:
                    delay_days_up_rf = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (วัน)", value=0, min_value=0)
                    delay_hours_up_rf = delay_days_up_rf * 24  # แปลงวันเป็นชั่วโมง
                    total_time_lag_up_rf = pd.Timedelta(days=delay_days_up_rf)
                else:
                    total_time_lag_up_rf = pd.Timedelta(days=0)

                if use_downstream_rf:
                    delay_days_down_rf = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (วัน)", value=0, min_value=0)
                    delay_hours_down_rf = delay_days_down_rf * 24  # แปลงวันเป็นชั่วโมง
                    total_time_lag_down_rf = pd.Timedelta(days=delay_days_down_rf)
                else:
                    total_time_lag_down_rf = pd.Timedelta(days=0)
            else:
                total_time_lag_up_rf = pd.Timedelta(days=0)
                total_time_lag_down_rf = pd.Timedelta(days=0)

            # อัปโหลดไฟล์หลัก
            uploaded_file_rf = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1_rf")

        # เลือกช่วงวันที่ใน sidebar
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            start_date_rf = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-05-01"))
            end_date_rf = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-05-31"))
            
            # เพิ่มตัวเลือกว่าจะลบข้อมูลหรือไม่
            delete_data_option_rf = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

            if delete_data_option_rf:
                # แสดงช่องกรอกข้อมูลสำหรับการลบข้อมูลเมื่อผู้ใช้ติ๊กเลือก
                st.header("เลือกช่วงที่ต้องการลบข้อมูล")
                delete_start_date_rf = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date_rf, key='delete_start_rf')
                delete_start_time_rf = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time_rf')
                delete_end_date_rf = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date_rf, key='delete_end_rf')
                delete_end_time_rf = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time_rf')

        process_button_rf = st.button("ประมวลผล Random Forest", type="primary")

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_nearby_lr = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            # Checkbox สำหรับเลือกใช้ Upstream และ Downstream
            if use_nearby_lr:
                use_upstream_lr = st.checkbox("ใช้สถานี Upstream", value=True)
                use_downstream_lr = st.checkbox("ใช้สถานี Downstream", value=False)
            else:
                use_upstream_lr = False
                use_downstream_lr = False
            
            # อัปโหลดไฟล์
            if use_nearby_lr and use_upstream_lr:
                uploaded_up_lr = st.file_uploader("ข้อมูลสถานี Upstream", type="csv", key="uploader_up_lr")
            else:
                uploaded_up_lr = None

            if use_nearby_lr and use_downstream_lr:
                uploaded_down_lr = st.file_uploader("ข้อมูลสถานี Downstream", type="csv", key="uploader_down_lr")
            else:
                uploaded_down_lr = None

            # เพิ่มช่องกรอกเวลาห่างระหว่างสถานี
            # กำหนดค่าเริ่มต้นสำหรับ delay_hours_up_lr และ delay_hours_down_lr
            delay_hours_up_lr = 0
            delay_hours_down_lr = 0

            if use_nearby_lr:
                if use_upstream_lr:
                    delay_hours_up_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0)
                    total_time_lag_up_lr = pd.Timedelta(hours=delay_hours_up_lr)
                else:
                    total_time_lag_up_lr = pd.Timedelta(hours=0)

                if use_downstream_lr:
                    delay_hours_down_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0)
                    total_time_lag_down_lr = pd.Timedelta(hours=delay_hours_down_lr)
                else:
                    total_time_lag_down_lr = pd.Timedelta(hours=0)
            else:
                total_time_lag_up_lr = pd.Timedelta(hours=0)
                total_time_lag_down_lr = pd.Timedelta(hours=0)

            # อัปโหลดไฟล์หลัก
            uploaded_fill_lr = st.file_uploader("ข้อมูลสถานีที่ต้องการพยากรณ์", type="csv", key="uploader_fill_lr")
            
        # แยกการเลือกช่วงข้อมูลสำหรับฝึกโมเดลและการพยากรณ์
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            training_start_date_lr = st.date_input("วันที่เริ่มต้นฝึกโมเดล", value=pd.to_datetime("2024-05-01"), key='training_start_lr')
            training_start_time_lr = st.time_input("เวลาเริ่มต้นฝึกโมเดล", value=pd.Timestamp("00:00:00").time(), key='training_start_time_lr')
            training_end_date_lr = st.date_input("วันที่สิ้นสุดฝึกโมเดล", value=pd.to_datetime("2024-05-31"), key='training_end_lr')
            training_end_time_lr = st.time_input("เวลาสิ้นสุดฝึกโมเดล", value=pd.Timestamp("23:45:00").time(), key='training_end_time_lr')

        with st.sidebar.expander("ตั้งค่าการพยากรณ์", expanded=False):
            forecast_days_lr = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)

        process_button_lr = st.button("ประมวลผล Linear Regression", type="primary")

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    if process_button_rf:
        if uploaded_file_rf is not None:
            try:
                main_df_rf = load_data(uploaded_file_rf)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์หลัก: {e}")
                main_df_rf = pd.DataFrame()

            if main_df_rf.empty:
                st.error("ไฟล์ CSV สำหรับเติมข้อมูลว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
            else:
                main_df_rf = clean_data(main_df_rf)
                if main_df_rf.empty:
                    st.error("หลังจากการทำความสะอาดข้อมูลแล้วไม่มีข้อมูลที่เหลือ")
                else:
                    main_df_rf = generate_missing_dates(main_df_rf)
                    main_df_rf['datetime'] = pd.to_datetime(main_df_rf['datetime'], errors='coerce').dt.tz_localize(None)
                    main_df_rf = create_time_features(main_df_rf)

                    # โหลดข้อมูล Upstream ถ้ามี
                    if use_nearby_rf and use_upstream_rf and uploaded_up_rf is not None:
                        try:
                            upstream_df_rf = pd.read_csv(uploaded_up_rf)
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ Upstream: {e}")
                            upstream_df_rf = pd.DataFrame()

                        upstream_df_rf = clean_data(upstream_df_rf)
                        if not upstream_df_rf.empty:
                            upstream_df_rf = generate_missing_dates(upstream_df_rf)
                            upstream_df_rf['datetime'] = pd.to_datetime(upstream_df_rf['datetime'], errors='coerce').dt.tz_localize(None)
                            upstream_df_rf = create_time_features(upstream_df_rf)
                    else:
                        upstream_df_rf = None

                    # โหลดข้อมูล Downstream ถ้ามี
                    if use_nearby_rf and use_downstream_rf and uploaded_down_rf is not None:
                        try:
                            downstream_df_rf = pd.read_csv(uploaded_down_rf)
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ Downstream: {e}")
                            downstream_df_rf = pd.DataFrame()

                        downstream_df_rf = clean_data(downstream_df_rf)
                        if not downstream_df_rf.empty:
                            downstream_df_rf = generate_missing_dates(downstream_df_rf)
                            downstream_df_rf['datetime'] = pd.to_datetime(downstream_df_rf['datetime'], errors='coerce').dt.tz_localize(None)
                            downstream_df_rf = create_time_features(downstream_df_rf)
                    else:
                        downstream_df_rf = None

                    # ลบข้อมูลตามช่วงที่กำหนด
                    if delete_data_option_rf:
                        delete_start_datetime_rf = pd.Timestamp.combine(delete_start_date_rf, delete_start_time_rf)
                        delete_end_datetime_rf = pd.Timestamp.combine(delete_end_date_rf, delete_end_time_rf)
                        main_df_rf = delete_data_by_date_range(main_df_rf, delete_start_datetime_rf, delete_end_datetime_rf)

                    # จัดการ missing values โดยใช้ Random Forest
                    filled_data_rf = handle_missing_values_by_week(main_df_rf, start_date_rf, end_date_rf, model_type='random_forest')

                    # แสดงกราฟผลลัพธ์
                    st.header("กราฟข้อมูลก่อนและหลังการเติมค่า (Random Forest)", divider='gray')
                    st.plotly_chart(plot_results(main_df_rf, filled_data_rf, main_df_rf, delete_data_option_rf), use_container_width=True)

                    # เสนอให้ผู้ใช้ดาวน์โหลดข้อมูลที่เติมแล้ว
                    csv_rf = filled_data_rf.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="ดาวน์โหลดข้อมูลที่เติมแล้ว (Random Forest)",
                        data=csv_rf,
                        file_name='randomforest_filled_data.csv',
                        mime='text/csv',
                    )
        else:
            st.error("กรุณาอัปโหลดไฟล์หลักสำหรับ Random Forest")

elif model_choice == "Linear Regression":
    if process_button_lr:
        if uploaded_fill_lr is not None:
            try:
                target_df_lr = pd.read_csv(uploaded_fill_lr)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์: {e}")
                target_df_lr = pd.DataFrame()

            if target_df_lr.empty:
                st.error("ไฟล์ CSV สำหรับเติมข้อมูลว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
            else:
                target_df_lr = clean_data(target_df_lr)
                if target_df_lr.empty:
                    st.error("หลังจากการทำความสะอาดข้อมูลแล้วไม่มีข้อมูลที่เหลือ")
                else:
                    target_df_lr = generate_missing_dates(target_df_lr)
                    target_df_lr['datetime'] = pd.to_datetime(target_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                    target_df_lr = create_time_features(target_df_lr)

                    # โหลดข้อมูล Upstream ถ้ามี
                    if use_nearby_lr and use_upstream_lr and uploaded_up_lr is not None:
                        try:
                            upstream_df_lr = pd.read_csv(uploaded_up_lr)
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ Upstream: {e}")
                            upstream_df_lr = pd.DataFrame()

                        upstream_df_lr = clean_data(upstream_df_lr)
                        if not upstream_df_lr.empty:
                            upstream_df_lr = generate_missing_dates(upstream_df_lr)
                            upstream_df_lr['datetime'] = pd.to_datetime(upstream_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                            upstream_df_lr = create_time_features(upstream_df_lr)
                    else:
                        upstream_df_lr = None

                    # โหลดข้อมูล Downstream ถ้ามี
                    if use_nearby_lr and use_downstream_lr and uploaded_down_lr is not None:
                        try:
                            downstream_df_lr = pd.read_csv(uploaded_down_lr)
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ Downstream: {e}")
                            downstream_df_lr = pd.DataFrame()

                        downstream_df_lr = clean_data(downstream_df_lr)
                        if not downstream_df_lr.empty:
                            downstream_df_lr = generate_missing_dates(downstream_df_lr)
                            downstream_df_lr['datetime'] = pd.to_datetime(downstream_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                            downstream_df_lr = create_time_features(downstream_df_lr)
                    else:
                        downstream_df_lr = None

                    # รวมข้อมูลจาก upstream และ downstream ถ้ามี
                    if use_nearby_lr and (use_upstream_lr or use_downstream_lr):
                        merged_training_data_lr = merge_data(target_df_lr, upstream_df_lr if use_upstream_lr else None)
                        merged_training_data_lr = merge_data(merged_training_data_lr, downstream_df_lr if use_downstream_lr else None)
                    else:
                        merged_training_data_lr = merge_data(target_df_lr)

                    if process_button_lr:
                        with st.spinner("กำลังพยากรณ์..."):
                            training_start_datetime_lr = pd.Timestamp.combine(training_start_date_lr, training_start_time_lr)
                            training_end_datetime_lr = pd.Timestamp.combine(training_end_date_lr, training_end_time_lr)
                            training_data_lr = merged_training_data_lr[
                                (merged_training_data_lr['datetime'] >= training_start_datetime_lr) & 
                                (merged_training_data_lr['datetime'] <= training_end_datetime_lr)
                            ].copy()

                            if training_data_lr.empty:
                                st.error("ไม่มีข้อมูลในช่วงเวลาที่เลือกสำหรับการฝึกโมเดล")
                            else:
                                forecast_start_date_actual_lr = training_end_datetime_lr + pd.Timedelta(minutes=15)
                                forecast_end_date_actual_lr = forecast_start_date_actual_lr + pd.Timedelta(days=forecast_days_lr)
                                max_datetime_lr = target_df_lr['datetime'].max()

                                if forecast_end_date_actual_lr > max_datetime_lr:
                                    st.warning("ข้อมูลจริงในช่วงเวลาที่พยากรณ์ไม่ครบถ้วนหรือไม่มีข้อมูล")

                                forecasted_data_lr = forecast_with_linear_regression_multi(
                                    data=target_df_lr.set_index('datetime'),
                                    upstream_data=upstream_df_lr.set_index('datetime') if upstream_df_lr is not None else None,
                                    downstream_data=downstream_df_lr.set_index('datetime') if downstream_df_lr is not None else None,
                                    forecast_start_date=forecast_start_date_actual_lr,
                                    forecast_days=forecast_days_lr,
                                    delay_hours_up=delay_hours_up_lr if use_nearby_lr and use_upstream_lr else 0,
                                    delay_hours_down=delay_hours_down_lr if use_nearby_lr and use_downstream_lr else 0
                                )

                                if not forecasted_data_lr.empty:
                                    st.header("กราฟข้อมูลพร้อมการพยากรณ์ (Linear Regression)", divider='gray')
                                    st.plotly_chart(
                                        plot_data_combined_two_stations(
                                            target_df_lr.set_index('datetime'), 
                                            forecasted_data_lr, 
                                            upstream_df_lr.set_index('datetime') if upstream_df_lr is not None else None, 
                                            downstream_df_lr.set_index('datetime') if downstream_df_lr is not None else None, 
                                            label='สถานีที่ต้องการทำนาย'
                                        ), 
                                        use_container_width=True
                                    )

                                    # เตรียมข้อมูลสำหรับการคำนวณค่าความแม่นยำ
                                    filled_lr = forecasted_data_lr.reset_index().rename(columns={'index': 'datetime'})
                                    filled_lr['wl_up2'] = filled_lr['wl_up']
                                    filled_lr.drop(columns=['wl_up'], inplace=True)

                                    mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics(
                                        original=target_df_lr,
                                        filled=filled_lr
                                    )

                                    if not merged_data_lr.empty:
                                        st.header("ตารางข้อมูลเปรียบเทียบ", divider='gray')
                                        comparison_table_lr = create_comparison_table_streamlit(forecasted_data_lr, merged_data_lr)
                                        st.dataframe(comparison_table_lr, use_container_width=True)

                                        st.header("ผลค่าความแม่นยำ", divider='gray')
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(label="Mean Squared Error (MSE)", value=f"{mse_lr:.4f}")
                                        with col2:
                                            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_lr:.4f}")
                                        with col3:
                                            st.metric(label="R-squared (R²)", value=f"{r2_lr:.4f}")
                                    else:
                                        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                                else:
                                    st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
        else:
            st.error("กรุณาอัปโหลดไฟล์สำหรับ Linear Regression")












