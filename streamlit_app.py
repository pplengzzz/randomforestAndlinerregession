import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ฟังก์ชันสำหรับโหลดข้อมูล
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

# ฟังก์ชันสำหรับทำความสะอาดข้อมูล
def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]

    # จัดการกับ spike
    data_clean.sort_values('datetime', inplace=True)
    data_clean.reset_index(drop=True, inplace=True)
    data_clean['diff'] = data_clean['wl_up'].diff().abs()
    threshold = data_clean['diff'].median() * 5

    # ระบุตำแหน่งที่เป็น spike
    data_clean['is_spike'] = data_clean['diff'] > threshold
    data_clean.loc[data_clean['is_spike'], 'wl_up'] = np.nan
    data_clean.drop(columns=['diff', 'is_spike'], inplace=True)
    data_clean['wl_up'] = data_clean['wl_up'].interpolate(method='linear')

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

# ฟังก์ชันสำหรับสร้างฟีเจอร์ lag และ lead
def create_lag_lead_features(data, lags=[1, 4, 96, 192], leads=[1, 4, 96, 192]):
    for lag in lags:
        data[f'lag_{lag}'] = data['wl_up'].shift(lag)
    for lead in leads:
        data[f'lead_{lead}'] = data['wl_up'].shift(-lead)
    return data

# ฟังก์ชันสำหรับสร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
def create_moving_average_features(data, window=672):
    data[f'ma_{window}'] = data['wl_up'].rolling(window=window, min_periods=1).mean()
    return data

# ฟังก์ชันสำหรับเตรียมฟีเจอร์
def prepare_features(data_clean, lags=[1, 4, 96, 192], leads=[1, 4, 96, 192], window=672):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]

    # สร้างฟีเจอร์ lag และ lead
    data_clean = create_lag_lead_features(data_clean, lags, leads)

    # สร้างฟีเจอร์ค่าเฉลี่ยเคลื่อนที่
    data_clean = create_moving_average_features(data_clean, window)

    # เพิ่มฟีเจอร์ lag และ lead เข้าไปใน feature_cols
    lag_cols = [f'lag_{lag}' for lag in lags]
    lead_cols = [f'lead_{lead}' for lead in leads]
    ma_col = f'ma_{window}'
    feature_cols.extend(lag_cols + lead_cols)
    feature_cols.append(ma_col)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag และ lead
    data_clean = data_clean.dropna(subset=feature_cols)

    X = data_clean[feature_cols[9:]]
    y = data_clean['wl_up']
    return X, y

# ฟังก์ชันสำหรับฝึกและประเมินผลโมเดล
def train_and_evaluate_model(X, y, model_type='random_forest'):
    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ฝึกโมเดล
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

# ฟังก์ชันสำหรับฝึก Random Forest
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
        verbose=2,
        random_state=42,
        scoring='neg_mean_absolute_error'
    )
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

# ฟังก์ชันสำหรับฝึก Linear Regression
def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ฟังก์ชันสำหรับสร้างวันที่ที่ขาดหาย
def generate_missing_dates(data):
    if data['datetime'].isnull().all():
        st.error("ไม่มีข้อมูลวันที่ในข้อมูลที่ให้มา")
        st.stop()
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

# ฟังก์ชันสำหรับเติมค่าในคอลัมน์ 'code'
def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

# ฟังก์ชันสำหรับหาจำนวนทศนิยม
def get_decimal_places(series):
    series_non_null = series.dropna().astype(str)
    decimal_counts = series_non_null.apply(lambda x: len(x.split('.')[-1]) if '.' in x else 0)
    if not decimal_counts.empty:
        return decimal_counts.mode()[0]
    else:
        return 2

# ฟังก์ชันสำหรับจัดการค่าที่ขาดหายไปช่วงแรก
def handle_initial_missing_values(data, initial_days=2, freq='15T'):
    initial_periods = initial_days * 24 * (60 // 15)
    for i in range(initial_periods):
        if pd.isna(data['wl_up'].iloc[i]):
            if i == 0:
                data.at[i, 'wl_up'] = data['wl_up'].mean()
            else:
                data.at[i, 'wl_up'] = data['wl_up'].iloc[i-1]
    return data

# ฟังก์ชันสำหรับจัดการค่าที่ขาดหายไปเป็นรายสัปดาห์
def handle_missing_values_by_week(data_clean, start_date, end_date, model_type='random_forest'):
    feature_cols = [
        'wl_up_prev',
        'lag_1', 'lag_4', 'lag_96', 'lag_192',
        'lead_1', 'lead_4', 'lead_96', 'lead_192',
        'ma_672'
    ]

    # เติมค่าที่ขาดหายไปในช่วงเริ่มต้น
    initial_periods = 2 * 24 * (60 // 15)
    initial_indices = data_clean.index[:initial_periods]
    filled_initial = data_clean.loc[initial_indices, 'wl_up'].isna()

    data_clean = handle_initial_missing_values(data_clean, initial_days=2)

    # ตั้งค่า wl_forecast และ timestamp สำหรับค่าที่เติมในช่วงเริ่มต้น
    data_clean.loc[initial_indices[filled_initial], 'wl_forecast'] = data_clean.loc[initial_indices[filled_initial], 'wl_up']
    data_clean.loc[initial_indices[filled_initial], 'timestamp'] = pd.Timestamp.now()

    data = data_clean.copy()

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data based on the datetime range
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    # ตรวจสอบว่ามีข้อมูลในช่วงที่เลือกหรือไม่
    if data.empty:
        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
        st.stop()

    # Generate all missing dates within the selected range
    data_with_all_dates = generate_missing_dates(data)
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates['datetime'])

    # เติมค่า missing ใน wl_up_prev
    if 'wl_up_prev' in data_with_all_dates.columns:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up_prev'].interpolate(method='linear')
    else:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up'].shift(1).interpolate(method='linear')

    # สร้างฟีเจอร์ lag และ lead รวมถึงค่าเฉลี่ยเคลื่อนที่
    data_with_all_dates = create_lag_lead_features(data_with_all_dates, lags=[1, 4, 96, 192], leads=[1, 4, 96, 192])
    data_with_all_dates = create_moving_average_features(data_with_all_dates, window=672)

    # เติมค่า missing ในฟีเจอร์ lag และ lead
    lag_cols = ['lag_1', 'lag_4', 'lag_96', 'lag_192']
    lead_cols = ['lead_1', 'lead_4', 'lead_96', 'lead_192']
    ma_col = 'ma_672'
    data_with_all_dates[lag_cols + lead_cols] = data_with_all_dates[lag_cols + lead_cols].interpolate(method='linear')
    data_with_all_dates[ma_col] = data_with_all_dates[ma_col].interpolate(method='linear')

    # แบ่งข้อมูลเป็นช่วงที่ขาดหายไปและไม่ขาดหายไป
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    if len(data_missing) == 0:
        st.write("ไม่มีค่าที่ขาดหายไปสำหรับการพยากรณ์")
        return data_with_all_dates

    # สร้าง DataFrame เพื่อเก็บผลลัพธ์
    data_filled = data_with_all_dates.copy()

    # ค้นหากลุ่มของค่าที่ขาดหายไป
    data_filled['missing_group'] = (data_filled['wl_up'].notnull() != data_filled['wl_up'].notnull().shift()).cumsum()
    missing_groups = data_filled[data_filled['wl_up'].isnull()].groupby('missing_group')

    # เตรียมโมเดล Random Forest หนึ่งครั้งก่อนวนลูป
    X_train, y_train = prepare_features(data_not_missing)

    # หา number of decimal places
    decimal_places = get_decimal_places(data_clean['wl_up'])

    # ใช้ข้อมูลจากสัปดาห์ก่อนหน้าและสัปดาห์ถัดไปถ้าข้อมูลในสัปดาห์ปัจจุบันไม่เพียงพอ
    if len(data_not_missing) < 192:
        week_prev = data_clean[
            (data_clean['datetime'] < start_date) & 
            (data_clean['datetime'] >= start_date - pd.Timedelta(weeks=1))
        ]
        
        week_next = data_clean[
            (data_clean['datetime'] > end_date) & 
            (data_clean['datetime'] <= end_date + pd.Timedelta(weeks=1))
        ]

        data_not_missing = pd.concat([data_not_missing, week_prev, week_next])
        
        X_train, y_train = prepare_features(data_not_missing)

    model = train_and_evaluate_model(X_train, y_train, model_type=model_type)

    # ตรวจสอบว่ามีโมเดลที่ถูกฝึกหรือไม่
    if model is None:
        st.error("ไม่สามารถสร้างโมเดลได้ กรุณาตรวจสอบข้อมูล")
        return data_with_all_dates

    for group_name, group_data in missing_groups:
        missing_length = len(group_data)
        idx_start = group_data.index[0]
        idx_end = group_data.index[-1]

        if missing_length <= 3:
            # เติมค่าด้วย interpolation ถ้าขาดหายไปไม่เกิน 3 แถว
            data_filled.loc[idx_start:idx_end, 'wl_up'] = np.nan
            data_filled['wl_up'] = data_filled['wl_up'].interpolate(method='linear')
            # ปัดเศษค่าที่เติม
            data_filled.loc[idx_start:idx_end, 'wl_up'] = data_filled.loc[idx_start:idx_end, 'wl_up'].round(decimal_places)
            # อัปเดตคอลัมน์ wl_forecast และ timestamp
            data_filled.loc[idx_start:idx_end, 'wl_forecast'] = data_filled.loc[idx_start:idx_end, 'wl_up']
            data_filled.loc[idx_start:idx_end, 'timestamp'] = pd.Timestamp.now()
        else:
            # ใช้ Random Forest ในการเติมค่าที่ขาดหายไป
            for idx in group_data.index:
                X_missing = data_filled.loc[idx, feature_cols].values.reshape(1, -1)
                try:
                    predicted_value = model.predict(X_missing)[0]
                    # ปัดเศษค่าที่พยากรณ์
                    predicted_value = round(predicted_value, decimal_places)
                    # บันทึกค่าที่เติมในคอลัมน์ wl_up และ wl_forecast
                    data_filled.at[idx, 'wl_forecast'] = predicted_value
                    data_filled.at[idx, 'wl_up'] = predicted_value
                    data_filled.at[idx, 'timestamp'] = pd.Timestamp.now()
                except Exception as e:
                    st.warning(f"ไม่สามารถพยากรณ์ค่าในแถว {idx} ได้: {e}")
                    continue

    # สร้างคอลัมน์ wl_up2 ที่รวมข้อมูลเดิมกับค่าที่เติม
    data_filled['wl_up2'] = data_filled['wl_up']

    # ลบคอลัมน์ที่ไม่จำเป็น
    data_filled.drop(columns=['missing_group'], inplace=True)

    data_filled.reset_index(drop=True, inplace=True)
    return data_filled

# ฟังก์ชันสำหรับลบข้อมูลตามช่วงวันที่
def delete_data_by_date_range(data, delete_start_date, delete_end_date):
    # Convert delete_start_date and delete_end_date to datetime
    delete_start_date = pd.to_datetime(delete_start_date)
    delete_end_date = pd.to_datetime(delete_end_date)

    # ตรวจสอบว่าช่วงวันที่ต้องการลบข้อมูลอยู่ในช่วงของ data หรือไม่
    data_to_delete = data[(data['datetime'] >= delete_start_date) & (data['datetime'] <= delete_end_date)]

    # เพิ่มการตรวจสอบว่าถ้าจำนวนข้อมูลที่ถูกลบมีมากเกินไป
    if len(data_to_delete) == 0:
        st.warning(f"ไม่พบข้อมูลระหว่าง {delete_start_date} และ {delete_end_date}.")
    elif len(data_to_delete) > (0.3 * len(data)):
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

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำสำหรับ Linear Regression
def calculate_accuracy_metrics_linear(original, filled):
    # ผสานข้อมูลตาม datetime
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')

    # ลบข้อมูลที่มี NaN ออก
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])

    if merged_data.empty:
        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None, merged_data

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(merged_data['wl_up'], merged_data['wl_up2'])
    mae = mean_absolute_error(merged_data['wl_up'], merged_data['wl_up2'])
    r2 = r2_score(merged_data['wl_up'], merged_data['wl_up2'])

    return mse, mae, r2, merged_data

# ฟังก์ชันสำหรับสร้างตารางเปรียบเทียบ
def create_comparison_table_streamlit(forecasted_data, actual_data):
    comparison_df = pd.DataFrame({
        'Datetime': actual_data['datetime'],
        'ค่าจริง': actual_data['wl_up'],
        'ค่าที่พยากรณ์': actual_data['wl_up2']
    })
    return comparison_df

# ฟังก์ชันสำหรับแสดงผลลัพธ์
def plot_results(data_before, data_filled, data_deleted, data_deleted_option=False):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
    })

    if data_deleted_option and data_deleted is not None:
        data_after_deleted = pd.DataFrame({
            'วันที่': data_deleted['datetime'],
            'ข้อมูลหลังลบ': data_deleted['wl_up']
        })
    else:
        data_after_deleted = None

    # รวมข้อมูลสำหรับกราฟ
    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')

    if data_after_deleted is not None and not data_after_deleted.empty:
        combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    # กำหนดลำดับของ y_columns และระบุสีตามเงื่อนไข
    if data_after_deleted is not None and not data_after_deleted.empty:
        y_columns = ["ข้อมูลเดิม", "ข้อมูลหลังเติมค่า", "ข้อมูลหลังลบ"]
        color_discrete_map = {
            "ข้อมูลหลังลบ": "#00cc96",
            "ข้อมูลหลังเติมค่า": "#636efa",
            "ข้อมูลเดิม": "#ef553b"
        }
    else:
        y_columns = ["ข้อมูลหลังเติมค่า", "ข้อมูลเดิม"]
        color_discrete_map = {
            "ข้อมูลเดิม": "#ef553b",
            "ข้อมูลหลังเติมค่า": "#636efa"
        }

    # วาดกราฟด้วย Plotly โดยระบุสีของเส้น
    fig = px.line(combined_data, x='วันที่', y=y_columns,
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  color_discrete_map=color_discrete_map)

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    # แสดงกราฟ
    st.header("ข้อมูลหลังจากการเติมค่าที่หายไป", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

    # แสดงตารางข้อมูลหลังเติมค่า
    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    data_filled_selected = data_filled[['code', 'datetime', 'wl_up2', 'wl_forecast', 'timestamp']]
    st.dataframe(data_filled_selected, use_container_width=True)

    # คำนวณค่าความแม่นยำถ้ามีการลบข้อมูล
    if data_deleted_option:
        calculate_accuracy_metrics(data_before, data_filled, data_deleted)
    else:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีการลบข้อมูล")

# ฟังก์ชันสำหรับแสดงกราฟข้อมูลจากสถานีต่างๆ
def plot_data_preview(df_pre, df2_pre, df3_pre, total_time_lag_upstream, total_time_lag_downstream):
    data_pre1 = pd.DataFrame({
        'datetime': df_pre['datetime'],
        'สถานีที่ต้องการทำนาย': df_pre['wl_up']
    })

    combined_data_pre = data_pre1.copy()

    if df2_pre is not None:
        data_pre2 = pd.DataFrame({
            'datetime': df2_pre['datetime'] + total_time_lag_upstream,
            'สถานีน้ำ Upstream': df2_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre2, on='datetime', how='outer')

    if df3_pre is not None:
        data_pre3 = pd.DataFrame({
            'datetime': df3_pre['datetime'] - total_time_lag_downstream,
            'สถานีน้ำ Downstream': df3_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre3, on='datetime', how='outer')

    # ตั้งค่า datetime เป็น index
    combined_data_pre.set_index('datetime', inplace=True)

    # กำหนด y_columns สำหรับกราฟ
    y_columns = combined_data_pre.columns.tolist()

    # Plot ด้วย Plotly
    fig = px.line(
        combined_data_pre,
        y=y_columns,
        labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'สถานี'},
        title='ข้อมูลจากสถานีต่างๆ',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title="วันที่",
        yaxis_title="ระดับน้ำ (wl_up)",
        legend_title="สถานี"
    )

    # แสดงกราฟ
    st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับรวมข้อมูล
def merge_data(df1, df2=None, df3=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))
    else:
        merged_df = df1.copy()
    
    if df3 is not None:
        merged_df = pd.merge(merged_df, df3[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))
    return merged_df

# ฟังก์ชันสำหรับรวมข้อมูล (สำหรับ Linear Regression)
def merge_data_linear(df1, df2=None, suffix='_prev'):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', suffix))
    else:
        df1[f'wl_up{suffix}'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันสำหรับสร้างกราฟข้อมูลและการพยากรณ์ (สำหรับ Linear Regression)
def plot_results_lr(data_before, data_forecasted):
    data_before = data_before.reset_index()

    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_forecasted = data_forecasted.reset_index()
    # ตั้งชื่อคอลัมน์ 'index' เป็น 'datetime' ถ้ายังไม่ได้ตั้ง
    if 'datetime' not in data_forecasted.columns:
        data_forecasted.rename(columns={'index': 'datetime'}, inplace=True)
    data_forecasted['วันที่'] = data_forecasted['datetime']
    data_forecasted['ข้อมูลพยากรณ์'] = data_forecasted['wl_up']

    # รวมข้อมูลสำหรับกราฟ
    combined_data = pd.merge(data_before_filled, data_forecasted[['วันที่', 'ข้อมูลพยากรณ์']], on='วันที่', how='outer')

    y_columns = ["ข้อมูลเดิม", "ข้อมูลพยากรณ์"]
    color_discrete_map = {
        "ข้อมูลเดิม": "#ef553b",
        "ข้อมูลพยากรณ์": "#636efa"
    }

    # วาดกราฟด้วย Plotly โดยระบุสีของเส้น
    fig = px.line(combined_data, x='วันที่', y=y_columns,
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  color_discrete_map=color_discrete_map)

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    # แสดงกราฟ
    st.header("ข้อมูลการพยากรณ์ระดับน้ำ", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

    # แสดงตารางข้อมูลการพยากรณ์
    st.header("ตารางแสดงข้อมูลการพยากรณ์", divider='gray')
    data_forecasted_selected = data_forecasted[['วันที่', 'ข้อมูลพยากรณ์']]
    st.dataframe(data_forecasted_selected, use_container_width=True)

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression สำหรับหลายสถานี
def forecast_with_linear_regression_multi(data, forecast_start_date, forecast_days, upstream_data=None, downstream_data=None, delay_hours_up=0, delay_hours_down=0):
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    lags = [1, 4, 96, 192]

    feature_cols = [f'lag_{lag}' for lag in lags]
    if upstream_data is not None:
        feature_cols += [f'lag_{lag}_upstream' for lag in lags]
    if downstream_data is not None:
        feature_cols += [f'lag_{lag}_downstream' for lag in lags]

    training_data = data.copy()

    if upstream_data is not None:
        upstream_data_shifted = upstream_data.copy()
        upstream_data_shifted.index = upstream_data_shifted.index + pd.Timedelta(hours=delay_hours_up)
        training_data = training_data.join(upstream_data_shifted[['wl_up']], rsuffix='_upstream')
    if downstream_data is not None:
        downstream_data_shifted = downstream_data.copy()
        downstream_data_shifted.index = downstream_data_shifted.index + pd.Timedelta(hours=delay_hours_down)
        training_data = training_data.join(downstream_data_shifted[['wl_up']], rsuffix='_downstream')

    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        if upstream_data is not None:
            training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)
        if downstream_data is not None:
            training_data[f'lag_{lag}_downstream'] = training_data['wl_up_downstream'].shift(lag)

    training_data.fillna(method='ffill', inplace=True)
    training_data.dropna(inplace=True)

    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    model = LinearRegression()
    model.fit(X_train, y_train)

    forecast_periods = forecast_days * 96
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])
    forecasted_data.index.name = 'datetime'  # ตั้งชื่อดัชนีเป็น 'datetime'

    combined_data = data.copy()
    if upstream_data is not None:
        combined_upstream = upstream_data.copy()
        combined_upstream.index = combined_upstream.index + pd.Timedelta(hours=delay_hours_up)
    else:
        combined_upstream = None
    if downstream_data is not None:
        combined_downstream = downstream_data.copy()
        combined_downstream.index = combined_downstream.index + pd.Timedelta(hours=delay_hours_down)
    else:
        combined_downstream = None

    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_features[f'lag_{lag}'] = combined_data.at[lag_time, 'wl_up']
            else:
                lag_features[f'lag_{lag}'] = y_train.mean()
            if upstream_data is not None:
                if lag_time in combined_upstream.index and not pd.isnull(combined_upstream.at[lag_time, 'wl_up']):
                    lag_features[f'lag_{lag}_upstream'] = combined_upstream.at[lag_time, 'wl_up']
                else:
                    lag_features[f'lag_{lag}_upstream'] = y_train.mean()
            if downstream_data is not None:
                if lag_time in combined_downstream.index and not pd.isnull(combined_downstream.at[lag_time, 'wl_up']):
                    lag_features[f'lag_{lag}_downstream'] = combined_downstream.at[lag_time, 'wl_up']
                else:
                    lag_features[f'lag_{lag}_downstream'] = y_train.mean()
        X_pred = pd.DataFrame([lag_features], columns=feature_cols)
        try:
            forecast_value = model.predict(X_pred)[0]
            forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
            forecasted_data.at[idx, 'wl_up'] = forecast_value
            combined_data.at[idx, 'wl_up'] = forecast_value
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในเวลา {idx} ได้: {e}")
    return forecasted_data

# เริ่มต้นส่วนของ Streamlit
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

# Sidebar
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

            # อัปโหลดไฟล์สถานีหลัก
            uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1")

        # เลือกช่วงวันที่ใน sidebar
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-08-01"))
            end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-08-31")) + pd.DateOffset(hours=23, minutes=45)
            
            # เพิ่มตัวเลือกว่าจะลบข้อมูลหรือไม่
            delete_data_option = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

            if delete_data_option:
                # แสดงช่องกรอกข้อมูลสำหรับการลบข้อมูลเมื่อผู้ใช้ติ๊กเลือก
                st.header("เลือกช่วงที่ต้องการลบข้อมูล")
                delete_start_date = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date, key='delete_start')
                delete_start_time = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time')
                delete_end_date = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date, key='delete_end')
                delete_end_time = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time')

        process_button = st.button("ประมวลผล", type="primary")

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

# ส่วนหลักของแอป
if model_choice == "Random Forest":
    if uploaded_file or uploaded_up_file or uploaded_down_file:
        # ตรวจสอบว่าไฟล์ข้อมูลระดับน้ำที่ต้องการทำนายถูกอัปโหลดหรือไม่
        if uploaded_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการทำนาย")
        # ตรวจสอบว่าไฟล์ข้อมูลสถานี Upstream ถูกอัปโหลดหรือไม่ ถ้าเลือกใช้
        if use_upstream and uploaded_up_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Upstream")
        # ตรวจสอบว่าไฟล์ข้อมูลสถานี Downstream ถูกอัปโหลดหรือไม่ ถ้าเลือกใช้
        if use_downstream and uploaded_down_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream")

        # ถ้ามีการอัปโหลดไฟล์ทั้งหมดที่ต้องการ
        if uploaded_file and (not use_upstream or uploaded_up_file) and (not use_downstream or uploaded_down_file):
            df = load_data(uploaded_file)

            if df is not None:
                df_pre = clean_data(df)
                df_pre = generate_missing_dates(df_pre)

                # ถ้าเลือกใช้ไฟล์ Upstream
                if use_upstream:
                    if uploaded_up_file is not None:
                        df_up = load_data(uploaded_up_file)
                        if df_up is not None:
                            df_up_pre = clean_data(df_up)
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
                            df_down_pre = generate_missing_dates(df_down_pre)
                        else:
                            df_down_pre = None
                    else:
                        st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream")
                        df_down_pre = None
                else:
                    df_down_pre = None

                # แสดงกราฟตัวอย่างทันทีหลังจากอัปโหลดไฟล์
                st.header("กราฟข้อมูลหลังจากอัปโหลดไฟล์", divider='gray')
                plot_data_preview(df_pre, df_up_pre, df_down_pre, total_time_lag_upstream, total_time_lag_downstream)

                if process_button:
                    processing_placeholder = st.empty()
                    processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                    # ปรับค่า end_date เฉพาะถ้าเลือกช่วงเวลาแล้ว
                    end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1)

                    # กรองข้อมูลตามช่วงวันที่เลือก
                    df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date_dt))]

                    # ตรวจสอบว่ามีข้อมูลในช่วงที่เลือกหรือไม่
                    if df_filtered.empty:
                        st.warning("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกช่วงวันที่ที่มีข้อมูล")
                        processing_placeholder.empty()
                        st.stop()

                    # ถ้าใช้ Upstream และมีไฟล์ Upstream และ df_up_pre ไม่ใช่ None
                    if use_upstream and uploaded_up_file and df_up_pre is not None:
                        # ปรับเวลาของสถานี Upstream ตามเวลาห่างที่ระบุ
                        df_up_pre['datetime'] = pd.to_datetime(df_up_pre['datetime']).dt.tz_localize(None)
                        df_up_filtered = df_up_pre[(df_up_pre['datetime'] >= pd.to_datetime(start_date)) & (df_up_pre['datetime'] <= pd.to_datetime(end_date_dt))]
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
                            (df_down_pre['datetime'] <= pd.to_datetime(end_date_dt))
                        ]
                        df_down_filtered['datetime'] = df_down_filtered['datetime'] - total_time_lag_downstream  # **ถอยเวลา** แทนการเพิ่ม
                        df_down_clean = clean_data(df_down_filtered)
                    else:
                        df_down_clean = None

                    # ทำความสะอาดข้อมูลหลัก
                    df_clean = clean_data(df_filtered)

                    # **เก็บข้อมูลหลังการทำความสะอาดแต่ก่อนการรวมข้อมูล**
                    df_before_deletion = df_clean.copy()

                    # รวมข้อมูลจากสถานี Upstream และ Downstream ถ้ามี
                    df_merged = merge_data(df_clean, df_up_clean, df_down_clean)

                    # ตรวจสอบว่าผู้ใช้เลือกที่จะลบข้อมูลหรือไม่
                    if delete_data_option:
                        delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
                        delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}")
                        df_deleted = delete_data_by_date_range(df_merged, delete_start_datetime, delete_end_datetime)
                    else:
                        df_deleted = df_merged.copy()

                    # Generate all dates
                    df_clean = generate_missing_dates(df_deleted)

                    # Fill NaN values in 'code' column
                    df_clean = fill_code_column(df_clean)

                    # Create time features
                    df_clean = create_time_features(df_clean)

                    # เติมค่า missing ใน 'wl_up_prev'
                    if 'wl_up_prev' not in df_clean.columns:
                        df_clean['wl_up_prev'] = df_clean['wl_up'].shift(1)
                    df_clean['wl_up_prev'] = df_clean['wl_up_prev'].interpolate(method='linear')

                    # Handle missing values by week
                    df_handled = handle_missing_values_by_week(df_clean, start_date, end_date, model_type='random_forest')

                    # Remove the processing message after the processing is complete
                    processing_placeholder.empty()

                    # Plot the results using Streamlit's line chart
                    plot_results(df_before_deletion, df_handled, df_deleted, data_deleted_option=delete_data_option)

            st.markdown("---")

    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Random Forest")

elif model_choice == "Linear Regression":
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

                # แสดงกราฟข้อมูลทันทีหลังจากอัปโหลดไฟล์
                st.header("กราฟข้อมูลหลังจากอัปโหลดไฟล์", divider='gray')
                plot_data_preview(target_df_lr, upstream_df_lr, downstream_df_lr, total_time_lag_up_lr, total_time_lag_down_lr)

                if process_button_lr:
                    with st.spinner("กำลังพยากรณ์..."):
                        training_start_datetime_lr = pd.Timestamp.combine(training_start_date_lr, training_start_time_lr)
                        training_end_datetime_lr = pd.Timestamp.combine(training_end_date_lr, training_end_time_lr)
                        target_df_lr = target_df_lr.set_index('datetime')
                        training_data_lr = target_df_lr[
                            (target_df_lr.index >= training_start_datetime_lr) &
                            (target_df_lr.index <= training_end_datetime_lr)
                        ].copy()
                        if training_data_lr.empty:
                            st.error("ไม่มีข้อมูลในช่วงเวลาที่เลือกสำหรับการฝึกโมเดล")
                            st.stop()
                        else:
                            forecast_start_date_actual_lr = training_end_datetime_lr + pd.Timedelta(minutes=15)
                            forecast_end_date_actual_lr = forecast_start_date_actual_lr + pd.Timedelta(days=forecast_days_lr)
                            max_datetime_lr = target_df_lr.index.max()
                            if forecast_end_date_actual_lr > max_datetime_lr:
                                st.warning("ข้อมูลจริงในช่วงเวลาที่พยากรณ์ไม่ครบถ้วนหรือไม่มีข้อมูล")
                            forecasted_data_lr = forecast_with_linear_regression_multi(
                                data=target_df_lr,
                                upstream_data=upstream_df_lr.set_index('datetime') if upstream_df_lr is not None else None,
                                downstream_data=downstream_df_lr.set_index('datetime') if downstream_df_lr is not None else None,
                                forecast_start_date=forecast_start_date_actual_lr,
                                forecast_days=forecast_days_lr,
                                delay_hours_up=delay_hours_up_lr if use_nearby_lr and use_upstream_lr else 0,
                                delay_hours_down=delay_hours_down_lr if use_nearby_lr and use_downstream_lr else 0
                            )
                            if not forecasted_data_lr.empty:
                                # แสดงผลลัพธ์เหมือนกับ Random Forest
                                plot_results_lr(target_df_lr, forecasted_data_lr)
                                st.markdown("---")
                                filled_lr = forecasted_data_lr.reset_index().rename(columns={'index': 'datetime'})
                                filled_lr['wl_up2'] = filled_lr['wl_up']
                                filled_lr.drop(columns=['wl_up'], inplace=True)
                                mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics_linear(
                                    original=target_df_lr.reset_index(),
                                    filled=filled_lr
                                )
                                if mse_lr is not None:
                                    st.header("ผลค่าความแม่นยำ", divider='gray')
                                    st.markdown("---")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(label="Mean Squared Error (MSE)", value=f"{mse_lr:.4f}")
                                    with col2:
                                        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_lr:.4f}")
                                    with col3:
                                        st.metric(label="R-squared (R²)", value=f"{r2_lr:.4f}")
                            else:
                                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Linear Regression")












































