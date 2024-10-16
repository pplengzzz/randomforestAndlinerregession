import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta

# ฟังก์ชันสำหรับการโหลดข้อมูล
def load_data(file):
    message_placeholder = st.empty()
    if file is None:
        st.error("ไม่มีไฟล์ที่อัปโหลด กรุณาอัปโหลดไฟล์ CSV")
        return None

    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("ไฟล์ CSV ว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
            return None
        message_placeholder.success("ไฟล์ถูกโหลดเรียบร้อยแล้ว")
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
        message_placeholder.empty()

# ฟังก์ชันสำหรับการทำความสะอาดข้อมูล
def clean_data(df):
    data_clean = df.copy()
    if 'datetime' not in data_clean.columns:
        st.error("ข้อมูลไม่มีคอลัมน์ 'datetime' กรุณาตรวจสอบไฟล์ข้อมูลของคุณ")
        return pd.DataFrame()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    if 'wl_up' not in data_clean.columns:
        st.error("ข้อมูลไม่มีคอลัมน์ 'wl_up' กรุณาตรวจสอบไฟล์ข้อมูลของคุณ")
        return pd.DataFrame()
    data_clean['wl_up'] = pd.to_numeric(data_clean['wl_up'], errors='coerce')
    data_clean = data_clean.dropna(subset=['wl_up'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    if data_clean.empty:
        st.error("หลังจากการทำความสะอาดข้อมูลแล้ว ไม่มีข้อมูลเหลืออยู่ กรุณาตรวจสอบไฟล์ข้อมูลของคุณ")
        return pd.DataFrame()
    data_clean.set_index('datetime', inplace=True)
    # ลบค่า NaT ใน Index ถ้ามี
    data_clean = data_clean[~data_clean.index.isnull()]
    if data_clean.empty:
        st.error("หลังจากการตั้งค่า index แล้ว ข้อมูลว่างเปล่า กรุณาตรวจสอบข้อมูลของคุณ")
        return pd.DataFrame()
    if not isinstance(data_clean.index, pd.DatetimeIndex):
        st.error("Index ไม่ใช่ชนิด DatetimeIndex ไม่สามารถทำการ resample ได้")
        return pd.DataFrame()
    if data_clean.select_dtypes(include=['number']).empty:
        st.error("ไม่มีคอลัมน์ตัวเลขสำหรับการ resample")
        return pd.DataFrame()
    # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการ resample หรือไม่
    if len(data_clean) < 2:
        st.error("มีข้อมูลไม่เพียงพอสำหรับการ resample")
        return pd.DataFrame()
    # เลือกเฉพาะคอลัมน์ตัวเลขก่อนทำการ resample
    numeric_cols = data_clean.select_dtypes(include=['number']).columns
    data_clean = data_clean[numeric_cols].resample('15T').mean()
    data_clean = data_clean.interpolate(method='linear')
    data_clean.sort_index(inplace=True)
    data_clean['diff'] = data_clean['wl_up'].diff().abs()
    threshold = data_clean['diff'].median() * 5
    data_clean.loc[data_clean['diff'] > threshold, 'wl_up'] = np.nan
    data_clean['wl_up'] = data_clean['wl_up'].interpolate(method='linear')
    data_clean.drop(columns=['diff'], inplace=True)
    data_clean.reset_index(inplace=True)
    return data_clean

# ฟังก์ชันสำหรับการสร้างฟีเจอร์เวลา
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

# ฟังก์ชันสำหรับการสร้างฟีเจอร์ล่าช้า
def create_lag_features(data, lags=[1, 4, 96, 192]):
    for lag in lags:
        data[f'lag_{lag}'] = data['wl_up'].shift(lag)
    return data

# ฟังก์ชันสำหรับการสร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
def create_moving_average_features(data, window=672):
    data[f'ma_{window}'] = data['wl_up'].rolling(window=window, min_periods=1).mean()
    return data

# ฟังก์ชันสำหรับเตรียมฟีเจอร์
def prepare_features(data_clean, lags=[1, 4, 96, 192], window=672):
    # สร้างฟีเจอร์เวลา
    data_clean = create_time_features(data_clean)

    # สร้างฟีเจอร์ wl_up_prev
    data_clean['wl_up_prev'] = data_clean['wl_up'].shift(1)
    data_clean['wl_up_prev'] = data_clean['wl_up_prev'].interpolate(method='linear')

    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]
    
    # สร้างฟีเจอร์ lag
    data_clean = create_lag_features(data_clean, lags)
    
    # สร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
    data_clean = create_moving_average_features(data_clean, window)
    
    # เพิ่มฟีเจอร์ lag และค่าเฉลี่ยเคลื่อนที่เข้าไปใน feature_cols
    lag_cols = [f'lag_{lag}' for lag in lags]
    ma_col = f'ma_{window}'
    feature_cols.extend(lag_cols)
    feature_cols.append(ma_col)
    
    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag
    data_clean = data_clean.dropna(subset=feature_cols)
    
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

# ฟังก์ชันสำหรับการฝึกและประเมินผลโมเดล
def train_and_evaluate_model(X, y, model_type='random_forest'):
    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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
        'n_estimators': [200, 300, 400, 500, 600],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=30,
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        scoring='neg_mean_absolute_error'
    )
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

# ฟังก์ชันสำหรับฝึกโมเดล Linear Regression
def train_linear_regression_model(X_train, y_train):
    pipeline = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    pipeline.fit(X_train, y_train)
    return pipeline

# ฟังก์ชันเพิ่มเติม
def generate_missing_dates(data):
    if data['datetime'].isnull().all():
        st.error("ไม่มีข้อมูลวันที่ในข้อมูลที่ให้มา")
        st.stop()
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

def handle_missing_values_by_week(data_clean, start_date, end_date, model_type='random_forest'):
    lags = [1, 4, 96, 192]
    window = 672

    data = data_clean.copy()

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.Timedelta(minutes=15)

    # Filter data based on the datetime range
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    # ตรวจสอบว่ามีข้อมูลในช่วงที่เลือกหรือไม่
    if data.empty:
        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
        st.stop()

    # Generate all missing dates within the selected range
    data_with_all_dates = generate_missing_dates(data)
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates['datetime'])
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    # เติมค่า missing ใน wl_up_prev
    data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up'].shift(1)
    data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up_prev'].interpolate(method='linear')

    # สร้างฟีเจอร์เวลา
    data_with_all_dates = create_time_features(data_with_all_dates)

    # สร้างฟีเจอร์ lag และค่าเฉลี่ยเคลื่อนที่
    data_with_all_dates = create_lag_features(data_with_all_dates, lags=lags)
    data_with_all_dates = create_moving_average_features(data_with_all_dates, window=window)

    # เติมค่า missing ในฟีเจอร์ lag และ ma
    lag_cols = [f'lag_{lag}' for lag in lags]
    ma_col = f'ma_{window}'
    data_with_all_dates[lag_cols] = data_with_all_dates[lag_cols].interpolate(method='linear')
    data_with_all_dates[ma_col] = data_with_all_dates[ma_col].interpolate(method='linear')

    # Update data_missing and data_not_missing after adding lag and ma
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    if len(data_missing) == 0:
        st.write("ไม่มีค่าที่หายไปให้พยากรณ์")
        return data_with_all_dates

    # Train initial model with all available data
    X_train, y_train = prepare_features(data_not_missing, lags=lags, window=window)
    model = train_and_evaluate_model(X_train, y_train, model_type=model_type)

    # ตรวจสอบว่ามีโมเดลที่ถูกฝึกหรือไม่
    if model is None:
        st.error("ไม่สามารถสร้างโมเดลได้ กรุณาตรวจสอบข้อมูล")
        return data_with_all_dates

    # เตรียมฟีเจอร์สำหรับการพยากรณ์
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month', 'wl_up_prev',
        'lag_1', 'lag_4', 'lag_96', 'lag_192', 'ma_672'
    ]

    # Fill missing values
    for idx, row in data_missing.iterrows():
        X_missing = row[feature_cols].values.reshape(1, -1)
        try:
            predicted_value = model.predict(X_missing)[0]
            # บันทึกค่าที่เติมในคอลัมน์ wl_forecast และ timestamp
            data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
            data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()
            # อัพเดตค่า wl_up ด้วยค่าที่พยากรณ์ เพื่อใช้ในการพยากรณ์ครั้งถัดไป
            data_with_all_dates.loc[idx, 'wl_up'] = predicted_value
            # อัพเดตฟีเจอร์ที่ต้องใช้ในการพยากรณ์
            data_with_all_dates.loc[idx+1:, 'wl_up_prev'] = data_with_all_dates['wl_up'].shift(1)
            data_with_all_dates.loc[idx+1:, lag_cols] = data_with_all_dates['wl_up'].shift(lags)
            data_with_all_dates.loc[idx+1:, ma_col] = data_with_all_dates['wl_up'].rolling(window=window, min_periods=1).mean().shift(1)
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในแถว {idx} ได้: {e}")
            continue

    # สร้างคอลัมน์ wl_up2 ที่รวมข้อมูลเดิมกับค่าที่เติม
    data_with_all_dates['wl_up2'] = data_with_all_dates['wl_up']

    data_with_all_dates.reset_index(drop=True, inplace=True)
    return data_with_all_dates

def delete_data_by_date_range(data, delete_start_datetime, delete_end_datetime):
    # ตรวจสอบว่าช่วงวันที่ต้องการลบข้อมูลอยู่ในช่วงของ data หรือไม่
    data_to_delete = data[(data['datetime'] >= delete_start_datetime) & (data['datetime'] <= delete_end_datetime)]

    # เพิ่มการตรวจสอบว่าถ้าจำนวนข้อมูลที่ถูกลบมีมากเกินไป
    if len(data_to_delete) == 0:
        st.warning(f"ไม่พบข้อมูลระหว่าง {delete_start_datetime} และ {delete_end_datetime}.")
    elif len(data_to_delete) > (0.3 * len(data)):  # ตรวจสอบว่าถ้าลบเกิน 30% ของข้อมูล
        st.warning("คำเตือน: มีข้อมูลมากเกินไปที่จะลบ การดำเนินการลบถูกยกเลิก")
    else:
        # ลบข้อมูลโดยตั้งค่า wl_up เป็น NaN
        data.loc[data_to_delete.index, 'wl_up'] = np.nan

    return data

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำ
def calculate_accuracy_metrics(original, filled, data_deleted):
    # ผสานข้อมูลตาม datetime
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')

    # เลือกเฉพาะข้อมูลที่ถูกลบ
    deleted_datetimes = data_deleted[data_deleted['wl_up'].isna()]['datetime']
    merged_data = merged_data[merged_data['datetime'].isin(deleted_datetimes)]

    # ลบข้อมูลที่มี NaN ออก
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])

    # ตรวจสอบว่ามีข้อมูลเพียงพอสำหรับการคำนวณหรือไม่
    if merged_data.empty:
        st.info("ไม่มีข้อมูลในช่วงที่ลบสำหรับการคำนวณค่าความแม่นยำ")
        return

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(merged_data['wl_up'], merged_data['wl_up2'])
    mae = mean_absolute_error(merged_data['wl_up'], merged_data['wl_up2'])
    r2 = r2_score(merged_data['wl_up'], merged_data['wl_up2'])

    st.header("ผลค่าความแม่นยำในช่วงที่ลบข้อมูล", divider='gray')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

# ฟังก์ชันสำหรับสร้างกราฟและตารางข้อมูล
def plot_results(data_before, data_filled, data_deleted, model_type='random_forest', data_deleted_option=False):
    # สร้าง DataFrame สำหรับข้อมูลก่อนเติมค่า
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    # สร้าง DataFrame สำหรับข้อมูลหลังเติมค่า
    if model_type == 'random_forest':
        data_after_filled = pd.DataFrame({
            'วันที่': data_filled['datetime'],
            'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
        })
    elif model_type == 'linear_regression':
        data_after_filled = pd.DataFrame({
            'วันที่': data_filled['datetime'],
            'ค่าที่พยากรณ์': data_filled['wl_up_pred']
        })

    # สร้าง DataFrame สำหรับข้อมูลหลังลบ (ถ้ามี)
    if data_deleted_option:
        data_after_deleted = pd.DataFrame({
            'วันที่': data_deleted['datetime'],
            'ข้อมูลหลังลบ': data_deleted['wl_up']
        })
    else:
        data_after_deleted = None

    # ผสานข้อมูลสำหรับกราฟ
    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')

    if data_after_deleted is not None:
        combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    # กำหนดรายการ y ที่จะแสดงในกราฟ
    if model_type == 'random_forest':
        y_columns = ['ข้อมูลหลังเติมค่า', 'ข้อมูลเดิม']
    elif model_type == 'linear_regression':
        y_columns = ['ค่าที่พยากรณ์', 'ข้อมูลเดิม']

    if data_after_deleted is not None:
        y_columns.append('ข้อมูลหลังลบ')

    # Plot ด้วย Plotly
    fig = px.line(combined_data, x='วันที่', y=y_columns,
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    # แสดงกราฟ
    st.header("ผลการพยากรณ์", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

    # แสดงตารางข้อมูลหลังเติมค่า
    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    if model_type == 'random_forest':
        data_filled_selected = data_filled[['datetime', 'wl_up', 'wl_forecast', 'timestamp']].copy()
        if 'code' in data_filled.columns:
            data_filled_selected['code'] = data_filled['code']
        st.dataframe(data_filled_selected, use_container_width=True)
    elif model_type == 'linear_regression':
        data_filled_selected = data_filled[['datetime', 'wl_up', 'wl_up_pred']].copy()
        st.dataframe(data_filled_selected, use_container_width=True)

    # ตรวจสอบว่ามีค่าจริงให้เปรียบเทียบหรือไม่ก่อนเรียกฟังก์ชันคำนวณความแม่นยำ
    if model_type == 'random_forest':
        if data_deleted_option:
            calculate_accuracy_metrics(data_before, data_filled, data_deleted)
        else:
            st.header("ผลค่าความแม่นยำ", divider='gray')
            st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีการลบข้อมูล")
    elif model_type == 'linear_regression':
        mse, mae, r2, merged_data = calculate_accuracy_metrics_linear(data_before, data_filled)
        if mse is not None:
            st.header("ผลค่าความแม่นยำ", divider='gray')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
            with col2:
                st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
            with col3:
                st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

# ฟังก์ชันสำหรับแสดงกราฟตัวอย่างข้อมูลที่อัปโหลด
def plot_data_preview(df_pre, df_up_pre=None, df_down_pre=None, total_time_lag_upstream=pd.Timedelta(hours=0), total_time_lag_downstream=pd.Timedelta(hours=0)):
    data_pre1 = pd.DataFrame({
        'datetime': df_pre['datetime'],
        'สถานีที่ต้องการทำนาย': df_pre['wl_up']
    })

    combined_data_pre = data_pre1.copy()

    if df_up_pre is not None:
        data_pre2 = pd.DataFrame({
            'datetime': df_up_pre['datetime'] + total_time_lag_upstream,
            'สถานีน้ำ Upstream': df_up_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre2, on='datetime', how='outer')

    if df_down_pre is not None:
        data_pre3 = pd.DataFrame({
            'datetime': df_down_pre['datetime'] - total_time_lag_downstream,
            'สถานีน้ำ Downstream': df_down_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre3, on='datetime', how='outer')

    # กำหนดรายการ y ที่จะแสดงในกราฟ
    y_columns = ['สถานีที่ต้องการทำนาย']
    if df_up_pre is not None:
        y_columns.append('สถานีน้ำ Upstream')
    if df_down_pre is not None:
        y_columns.append('สถานีน้ำ Downstream')

    # Plot ด้วย Plotly
    fig = px.line(
        combined_data_pre, 
        x='datetime', 
        y=y_columns,
        labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
        title='ข้อมูลจากสถานีต่างๆ',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title="วันที่", 
        yaxis_title="ระดับน้ำ (wl_up)"
    )

    # แสดงกราฟ
    st.header("กราฟแสดงข้อมูลที่อัปโหลด", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับรวมข้อมูลจากสถานีต่างๆ
def merge_data(df1, df2=None, df3=None):
    merged_df = df1.copy()
    if df2 is not None:
        merged_df = pd.merge(merged_df, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))
    if df3 is not None:
        merged_df = pd.merge(merged_df, df3[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))
    return merged_df

# ฟังก์ชันสำหรับการฝึกและพยากรณ์ด้วย Linear Regression
def train_and_forecast_LR(target_data, upstream_data=None, downstream_data=None, use_upstream=False, use_downstream=False, forecast_days=2, travel_time_up=0, travel_time_down=0):
    # ทำความสะอาดข้อมูล
    target_data = clean_data(target_data)
    if target_data.empty:
        st.error("ไม่สามารถดำเนินการต่อได้ เนื่องจากข้อมูลที่ต้องการทำนายไม่มีข้อมูลหลังจากการทำความสะอาด")
        return None
    target_data = create_time_features(target_data)

    if use_upstream and upstream_data is not None:
        upstream_data = clean_data(upstream_data)
        if upstream_data.empty:
            st.error("ไม่สามารถดำเนินการต่อได้ เนื่องจากข้อมูล Upstream ไม่มีข้อมูลหลังจากการทำความสะอาด")
            return None
        upstream_data = create_time_features(upstream_data)
        # เลื่อนข้อมูล upstream ตาม travel_time_up (เป็นชั่วโมง)
        upstream_shift = timedelta(hours=travel_time_up)
        upstream_data['datetime'] = upstream_data['datetime'] + upstream_shift
        # รวมข้อมูล
        target_data = pd.merge(target_data, upstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))

    if use_downstream and downstream_data is not None:
        downstream_data = clean_data(downstream_data)
        if downstream_data.empty:
            st.error("ไม่สามารถดำเนินการต่อได้ เนื่องจากข้อมูล Downstream ไม่มีข้อมูลหลังจากการทำความสะอาด")
            return None
        downstream_data = create_time_features(downstream_data)
        # เลื่อนข้อมูล downstream ตาม travel_time_down (เป็นชั่วโมง)
        downstream_shift = timedelta(hours=travel_time_down)
        downstream_data['datetime'] = downstream_data['datetime'] + downstream_shift
        # รวมข้อมูล
        target_data = pd.merge(target_data, downstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))

    # เติมค่าที่ขาดหายด้วยวิธี Forward Fill สำหรับ upstream และ downstream (ถ้ามี)
    if use_upstream and 'wl_upstream' in target_data.columns:
        target_data['wl_upstream'] = target_data['wl_upstream'].ffill()

    if use_downstream and 'wl_up_downstream' in target_data.columns:
        target_data['wl_up_downstream'] = target_data['wl_up_downstream'].ffill()

    # สร้างฟีเจอร์แบบล่าช้า (Lag Features)
    lags = [1, 2, 4, 8]  # ล่าช้า 15 นาที, 30 นาที, 1 ชั่วโมง, และ 2 ชั่วโมง
    for lag in lags:
        target_data[f'wl_up_target_lag_{lag}'] = target_data['wl_up'].shift(lag)
        if use_upstream and 'wl_upstream' in target_data.columns:
            target_data[f'wl_upstream_lag_{lag}'] = target_data['wl_upstream'].shift(lag)
        if use_downstream and 'wl_up_downstream' in target_data.columns:
            target_data[f'wl_up_downstream_lag_{lag}'] = target_data['wl_up_downstream'].shift(lag)

    # ลบแถวที่มีค่า NaN จากการล่าช้า
    target_data = target_data.dropna().copy()

    # ตรวจสอบว่ามีข้อมูลสำหรับการฝึกโมเดลหรือไม่
    if target_data.empty:
        st.error("ไม่มีข้อมูลหลังจากการกรองช่วงวันที่ กรุณาตรวจสอบช่วงวันที่ที่กำหนด.")
        return None

    # เตรียมข้อมูลสำหรับการฝึกโมเดล
    features = [f'wl_up_target_lag_{lag}' for lag in lags]
    if use_upstream and any(f'wl_upstream_lag_{lag}' in target_data.columns for lag in lags):
        features += [f'wl_upstream_lag_{lag}' for lag in lags]
    if use_downstream and any(f'wl_up_downstream_lag_{lag}' in target_data.columns for lag in lags):
        features += [f'wl_up_downstream_lag_{lag}' for lag in lags]

    # ตรวจสอบว่ามีฟีเจอร์ที่ต้องการทั้งหมดหรือไม่
    missing_features = [feat for feat in features if feat not in target_data.columns]
    if missing_features:
        st.error(f"ขาดฟีเจอร์ที่ต้องการ: {missing_features}")
        return None

    X = target_data[features]
    y = target_data['wl_up']

    # ตรวจสอบว่าชุดข้อมูลไม่ว่างเปล่า
    if X.empty or y.empty:
        st.error("ชุดข้อมูลสำหรับการฝึกโมเดลว่างเปล่า กรุณาตรวจสอบการทำความสะอาดข้อมูล.")
        return None

    # สร้าง Pipeline สำหรับการฝึกโมเดล
    pipeline = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    # ฝึกโมเดลบนชุดข้อมูลทั้งหมด
    pipeline.fit(X, y)

    # การพยากรณ์อนาคต
    future_dates = [target_data['datetime'].max() + timedelta(minutes=15 * i) for i in range(1, forecast_days * 96 + 1)]  # 96 ช่วงต่อวัน (24*4)

    # เตรียมข้อมูลจริงในช่วงเวลาพยากรณ์ (ถ้ามี)
    actual_data = target_data[['datetime', 'wl_up']].copy()

    future_predictions = []
    current_data = target_data.copy()

    for date in future_dates:
        # สร้างฟีเจอร์ล่าช้าสำหรับวันถัดไป
        input_features = {}
        for lag in lags:
            past_time = date - timedelta(minutes=15 * lag)
            past_row = current_data[current_data['datetime'] == past_time]
            if not past_row.empty:
                input_features[f'wl_up_target_lag_{lag}'] = past_row['wl_up'].values[0]
                if use_upstream and 'wl_upstream' in past_row.columns:
                    input_features[f'wl_upstream_lag_{lag}'] = past_row['wl_upstream'].values[0]
                if use_downstream and 'wl_up_downstream' in past_row.columns:
                    input_features[f'wl_up_downstream_lag_{lag}'] = past_row['wl_up_downstream'].values[0]
            else:
                # หากไม่มีข้อมูล ให้หยุดการพยากรณ์
                st.warning(f"ไม่สามารถสร้างฟีเจอร์สำหรับวันที่ {date} ได้ เนื่องจากข้อมูลขาดหาย")
                break

        if len(input_features) < len(features):
            # หากฟีเจอร์ไม่ครบ ให้หยุดการพยากรณ์
            st.warning(f"ฟีเจอร์ไม่ครบสำหรับวันที่ {date}")
            break

        # สร้าง DataFrame สำหรับการพยากรณ์
        input_df = pd.DataFrame([input_features])

        # จัดเรียงฟีเจอร์ให้ตรงกับลำดับที่ใช้ในการฝึกโมเดล
        input_df = input_df[features]

        # ทำการพยากรณ์
        try:
            pred = pipeline.predict(input_df)[0]
        except ValueError as ve:
            st.warning(f"เกิดข้อผิดพลาดในการพยากรณ์วันที่ {date}: {ve}")
            break

        future_predictions.append({'datetime': date, 'wl_up_pred': pred})

        # เพิ่มข้อมูลที่พยากรณ์แล้วลงใน current_data
        new_row = {
            'datetime': date,
            'wl_up': pred
        }
        if use_upstream and 'wl_upstream' in input_features:
            new_row['wl_upstream'] = input_features[f'wl_upstream_lag_{min(lags)}']
        if use_downstream and 'wl_up_downstream' in input_features:
            new_row['wl_up_downstream'] = input_features[f'wl_up_downstream_lag_{min(lags)}']

        # เพิ่มแถวใหม่ลงใน current_data โดยใช้ pd.concat แทน append
        new_row_df = pd.DataFrame([new_row])
        current_data = pd.concat([current_data, new_row_df], ignore_index=True)

    future_df = pd.DataFrame(future_predictions)

    # รวมข้อมูลจริงและข้อมูลพยากรณ์เข้าด้วยกัน
    combined_data = pd.concat([target_data[['datetime', 'wl_up']], future_df], ignore_index=True)
    combined_data = combined_data.sort_values('datetime').reset_index(drop=True)

    return combined_data

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำของ Linear Regression
def calculate_accuracy_metrics_linear(original, forecasted):
    # ผสานข้อมูลตาม datetime
    merged_data = pd.merge(original[['datetime', 'wl_up']], forecasted[['datetime', 'wl_up_pred']], on='datetime')

    # ลบข้อมูลที่มี NaN ออก
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up_pred'])

    if merged_data.empty:
        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None, merged_data

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(merged_data['wl_up'], merged_data['wl_up_pred'])
    mae = mean_absolute_error(merged_data['wl_up'], merged_data['wl_up_pred'])
    r2 = r2_score(merged_data['wl_up'], merged_data['wl_up_pred'])

    return mse, mae, r2, merged_data

# ส่วนของ Streamlit UI
st.set_page_config(
    page_title="การพยากรณ์ระดับน้ำ",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
# การพยากรณ์ระดับน้ำ

แอป Streamlit สำหรับจัดการข้อมูลระดับน้ำ โดยใช้โมเดล **Random Forest** หรือ **Linear Regression** เพื่อเติมค่าที่ขาดหายไปและพยากรณ์ข้อมูล
ข้อมูลถูกประมวลผลและแสดงผลผ่านกราฟและการวัดค่าความแม่นยำ ผู้ใช้สามารถเลือกอัปโหลดไฟล์, 
กำหนดช่วงเวลาลบข้อมูล และเลือกวิธีการพยากรณ์ได้
""")
st.markdown("---")

# Sidebar: Upload files and choose date ranges
with st.sidebar:

    st.sidebar.title("เลือกวิธีการพยากรณ์")
    model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"),
        label_visibility="collapsed"
    )

    st.sidebar.title("ตั้งค่าข้อมูล")
    if model_choice == "Random Forest":
        with st.sidebar.expander("ตั้งค่า Random Forest", expanded=False):
            use_upstream = st.checkbox("ต้องการใช้สถานี Upstream", value=False)
            use_downstream = st.checkbox("ต้องการใช้สถานี Downstream", value=False)
            
            # การอัปโหลดไฟล์สถานีหลักและสถานีน้ำ Upstream
            uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1")
            if use_upstream:
                uploaded_up_file = st.file_uploader("ข้อมูลระดับน้ำ Upstream", type="csv", key="uploader_up")
                time_lag_upstream = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0)
                total_time_lag_upstream = pd.Timedelta(hours=time_lag_upstream)
            else:
                uploaded_up_file = None
                total_time_lag_upstream = pd.Timedelta(hours=0)
            
            # การอัปโหลดไฟล์สถานีน้ำ Downstream
            if use_downstream:
                uploaded_down_file = st.file_uploader("ข้อมูลระดับน้ำ Downstream", type="csv", key="uploader_down")
                time_lag_downstream = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0)
                total_time_lag_downstream = pd.Timedelta(hours=time_lag_downstream)
            else:
                uploaded_down_file = None
                total_time_lag_downstream = pd.Timedelta(hours=0)

        # เลือกช่วงวันที่ใน sidebar
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-05-01"))
            end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-05-31"))
            
            # เพิ่มตัวเลือกว่าจะลบข้อมูลหรือไม่
            delete_data_option = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

            if delete_data_option:
                # แสดงช่องกรอกข้อมูลสำหรับการลบข้อมูลเมื่อผู้ใช้ติ๊กเลือก
                st.header("เลือกช่วงที่ต้องการลบข้อมูล")
                delete_start_date = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date, key='delete_start')
                delete_start_time = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time')
                delete_end_date = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date, key='delete_end')
                delete_end_time = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time')

        process_button = st.button("ประมวลผล Random Forest", type="primary")

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_upstream_lr = st.checkbox("ต้องการใช้สถานี Upstream", value=False)
            use_downstream_lr = st.checkbox("ต้องการใช้สถานี Downstream", value=False)
            
            # การอัปโหลดไฟล์สถานีหลักและสถานีน้ำ Upstream
            uploaded_file_lr = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1_lr")
            if use_upstream_lr:
                uploaded_up_file_lr = st.file_uploader("ข้อมูลระดับน้ำ Upstream", type="csv", key="uploader_up_lr")
                time_lag_upstream_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0, key="time_lag_upstream_lr")
            else:
                uploaded_up_file_lr = None
                time_lag_upstream_lr = 0
            
            # การอัปโหลดไฟล์สถานีน้ำ Downstream
            if use_downstream_lr:
                uploaded_down_file_lr = st.file_uploader("ข้อมูลระดับน้ำ Downstream", type="csv", key="uploader_down_lr")
                time_lag_downstream_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0, key="time_lag_downstream_lr")
            else:
                uploaded_down_file_lr = None
                time_lag_downstream_lr = 0
            
        # แยกการเลือกช่วงข้อมูลสำหรับฝึกโมเดลและการพยากรณ์
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            training_start_date_lr = st.date_input("วันที่เริ่มต้นฝึกโมเดล", value=pd.to_datetime("2024-05-01"), key='training_start_lr')
            training_start_time_lr = st.time_input("เวลาเริ่มต้นฝึกโมเดล", value=pd.Timestamp("00:00:00").time(), key='training_start_time_lr')
            training_end_date_lr = st.date_input("วันที่สิ้นสุดฝึกโมเดล", value=pd.to_datetime("2024-05-31"), key='training_end_lr')
            training_end_time_lr = st.time_input("เวลาสิ้นสุดฝึกโมเดล", value=pd.Timestamp("23:45:00").time(), key='training_end_time_lr')

        with st.sidebar.expander("ตั้งค่าการพยากรณ์", expanded=False):
            forecast_days_lr = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)

        process_button_lr = st.button("ประมวลผล Linear Regression", type="primary")

# ฟังก์ชันสำหรับแสดงผลกราฟและตารางสำหรับทั้ง RF และ LR
def display_model_results(model_type, data_before, data_handled, data_deleted, delete_data_option):
    plot_results(
        data_before=data_before,
        data_filled=data_handled,
        data_deleted=data_deleted,
        model_type=model_type,
        data_deleted_option=delete_data_option
    )

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    if uploaded_file or uploaded_up_file or uploaded_down_file:
        if uploaded_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการทำนาย")
        if use_upstream and uploaded_up_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Upstream")
        if use_downstream and uploaded_down_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream")

        if uploaded_file and (not use_upstream or uploaded_up_file) and (not use_downstream or uploaded_down_file):
            df = load_data(uploaded_file)

            if df is not None:
                df_pre = clean_data(df)
                if df_pre.empty:
                    st.stop()
                df_pre = generate_missing_dates(df_pre)

                # ถ้าเลือกใช้ไฟล์ Upstream
                if use_upstream:
                    if uploaded_up_file is not None:
                        df_up = load_data(uploaded_up_file)
                        if df_up is not None:
                            df_up_pre = clean_data(df_up)
                            if df_up_pre.empty:
                                st.stop()
                            df_up_pre = generate_missing_dates(df_up_pre)
                        else:
                            df_up_pre = None
                    else:
                        st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Upstream")
                        df_up_pre = None
                else:
                    df_up_pre = None

                # ถ้าเลือกใช้ไฟล์ Downstream
                if use_downstream:
                    if uploaded_down_file is not None:
                        df_down = load_data(uploaded_down_file)
                        if df_down is not None:
                            df_down_pre = clean_data(df_down)
                            if df_down_pre.empty:
                                st.stop()
                            df_down_pre = generate_missing_dates(df_down_pre)
                        else:
                            df_down_pre = None
                    else:
                        st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream")
                        df_down_pre = None
                else:
                    df_down_pre = None

                # แสดงกราฟตัวอย่าง
                plot_data_preview(df_pre, df_up_pre, df_down_pre, total_time_lag_upstream, total_time_lag_downstream)

                if process_button:
                    processing_placeholder = st.empty()
                    processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                    # ปรับค่า end_date เฉพาะถ้าเลือกช่วงเวลาแล้ว
                    end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1) - pd.Timedelta(minutes=15)

                    # กรองข้อมูลตามช่วงวันที่เลือก
                    df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= end_date_dt)]

                    # ตรวจสอบว่ามีข้อมูลในช่วงที่เลือกหรือไม่
                    if df_filtered.empty:
                        st.warning("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
                        processing_placeholder.empty()
                        st.stop()

                    # ถ้าใช้ Upstream และมีไฟล์ Upstream และ df_up_pre ไม่ใช่ None
                    if use_upstream and uploaded_up_file and df_up_pre is not None:
                        # ปรับเวลาของสถานี Upstream ตามเวลาห่างที่ระบุ
                        df_up_pre['datetime'] = pd.to_datetime(df_up_pre['datetime']).dt.tz_localize(None)
                        df_up_filtered = df_up_pre[(df_up_pre['datetime'] >= pd.to_datetime(start_date)) & (df_up_pre['datetime'] <= end_date_dt)]
                        df_up_filtered['datetime'] = df_up_filtered['datetime'] + total_time_lag_upstream
                        df_up_clean = clean_data(df_up_filtered)
                    else:
                        df_up_clean = None

                    # ถ้าเลือกใช้ไฟล์ Downstream
                    if use_downstream and uploaded_down_file and df_down_pre is not None:
                        # ปรับเวลาของสถานี Downstream ตามเวลาห่างที่ระบุ
                        df_down_pre['datetime'] = pd.to_datetime(df_down_pre['datetime']).dt.tz_localize(None)
                        df_down_filtered = df_down_pre[
                            (df_down_pre['datetime'] >= pd.to_datetime(start_date)) & 
                            (df_down_pre['datetime'] <= end_date_dt)
                        ]
                        df_down_filtered['datetime'] = df_down_filtered['datetime'] - total_time_lag_downstream
                        df_down_clean = clean_data(df_down_filtered)
                    else:
                        df_down_clean = None

                    # ทำความสะอาดข้อมูลหลัก
                    df_clean = clean_data(df_filtered)
                    if df_clean.empty:
                        st.stop()

                    # เก็บข้อมูลหลังการทำความสะอาดแต่ก่อนการรวมข้อมูล
                    df_before_deletion = df_clean.copy()

                    # รวมข้อมูลจากสถานี Upstream และ Downstream ถ้ามี
                    df_merged = merge_data(df_clean, df_up_clean, df_down_clean)

                    # ตรวจสอบว่าผู้ใช้เลือกที่จะลบข้อมูลหรือไม่
                    if delete_data_option:
                        delete_start_datetime = pd.Timestamp.combine(delete_start_date, delete_start_time)
                        delete_end_datetime = pd.Timestamp.combine(delete_end_date, delete_end_time)
                        df_deleted = delete_data_by_date_range(df_merged, delete_start_datetime, delete_end_datetime)
                    else:
                        df_deleted = df_merged.copy()

                    # Generate all dates
                    df_clean = generate_missing_dates(df_deleted)

                    # Fill NaN values in 'code' column
                    df_clean = fill_code_column(df_clean)

                    # Handle missing values by week
                    df_handled = handle_missing_values_by_week(df_clean, start_date, end_date, model_type='random_forest')

                    # Remove the processing message after the processing is complete
                    processing_placeholder.empty()

                    # Plot the results using Streamlit's line chart
                    display_model_results(
                        model_type='random_forest',
                        data_before=df_before_deletion,
                        data_handled=df_handled,
                        data_deleted=df_deleted,
                        delete_data_option=delete_data_option
                    )

            st.markdown("---")

    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Random Forest")

elif model_choice == "Linear Regression":
    if uploaded_file_lr or uploaded_up_file_lr or uploaded_down_file_lr:
        if uploaded_file_lr is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการทำนายสำหรับ Linear Regression")
        if use_upstream_lr and uploaded_up_file_lr is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Upstream สำหรับ Linear Regression")
        if use_downstream_lr and uploaded_down_file_lr is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream สำหรับ Linear Regression")

        if uploaded_file_lr and (not use_upstream_lr or uploaded_up_file_lr) and (not use_downstream_lr or uploaded_down_file_lr):
            target_df_lr = load_data(uploaded_file_lr)

            if target_df_lr is not None:
                target_df_lr['datetime'] = pd.to_datetime(target_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                # แสดงกราฟข้อมูลที่อัปโหลดทันที
                plot_data_preview(target_df_lr, None, None)
                if process_button_lr:
                    # กรองข้อมูลตามช่วงวันที่เลือก
                    training_start_datetime_lr = pd.Timestamp.combine(training_start_date_lr, training_start_time_lr)
                    training_end_datetime_lr = pd.Timestamp.combine(training_end_date_lr, training_end_time_lr)
                    target_df_lr = target_df_lr[
                        (target_df_lr['datetime'] >= training_start_datetime_lr) & 
                        (target_df_lr['datetime'] <= training_end_datetime_lr)
                    ]

                    if target_df_lr.empty:
                        st.error("ไม่มีข้อมูลในช่วงเวลาที่เลือกสำหรับการฝึกโมเดล")
                    else:
                        if use_upstream_lr and uploaded_up_file_lr is not None:
                            upstream_df_lr = load_data(uploaded_up_file_lr)
                            if upstream_df_lr is not None and not upstream_df_lr.empty:
                                upstream_df_lr['datetime'] = pd.to_datetime(upstream_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                            else:
                                upstream_df_lr = None
                                st.warning("ข้อมูล Upstream ว่างเปล่าหรือมีปัญหา")
                        else:
                            upstream_df_lr = None

                        if use_downstream_lr and uploaded_down_file_lr is not None:
                            downstream_df_lr = load_data(uploaded_down_file_lr)
                            if downstream_df_lr is not None and not downstream_df_lr.empty:
                                downstream_df_lr['datetime'] = pd.to_datetime(downstream_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                            else:
                                downstream_df_lr = None
                                st.warning("ข้อมูล Downstream ว่างเปล่าหรือมีปัญหา")
                        else:
                            downstream_df_lr = None

                        # ทำความสะอาดข้อมูล
                        target_df_lr = clean_data(target_df_lr)
                        if target_df_lr.empty:
                            st.stop()

                        if use_upstream_lr and upstream_df_lr is not None:
                            upstream_df_lr = clean_data(upstream_df_lr)
                            if upstream_df_lr.empty:
                                st.stop()
                        if use_downstream_lr and downstream_df_lr is not None:
                            downstream_df_lr = clean_data(downstream_df_lr)
                            if downstream_df_lr.empty:
                                st.stop()

                        # แสดงกราฟข้อมูลที่อัปโหลด
                        plot_data_preview(target_df_lr, upstream_df_lr, downstream_df_lr, 
                                          pd.Timedelta(hours=time_lag_upstream_lr), pd.Timedelta(hours=time_lag_downstream_lr))

                        combined_data_lr = train_and_forecast_LR(
                            target_data=target_df_lr,
                            upstream_data=upstream_df_lr,
                            downstream_data=downstream_df_lr,
                            use_upstream=use_upstream_lr,
                            use_downstream=use_downstream_lr,
                            forecast_days=forecast_days_lr,
                            travel_time_up=time_lag_upstream_lr,
                            travel_time_down=time_lag_downstream_lr
                        )

                        if combined_data_lr is not None and not combined_data_lr.empty:
                            # เก็บข้อมูลก่อนการพยากรณ์สำหรับการเปรียบเทียบ
                            data_before_lr = target_df_lr.copy()
                            data_handled_lr = combined_data_lr.copy()

                            # Plot the results using the same function
                            display_model_results(
                                model_type='linear_regression',
                                data_before=data_before_lr,
                                data_handled=data_handled_lr,
                                data_deleted=None,
                                delete_data_option=False
                            )
                        else:
                            st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
            else:
                st.error("กรุณาอัปโหลดไฟล์สำหรับ Linear Regression")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Linear Regression")



































