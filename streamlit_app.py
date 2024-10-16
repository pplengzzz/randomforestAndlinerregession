import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    # แปลงคอลัมน์ 'wl_up' เป็นตัวเลขและลบค่าที่ไม่ใช่ตัวเลข
    data_clean['wl_up'] = pd.to_numeric(data_clean['wl_up'], errors='coerce')
    data_clean = data_clean.dropna(subset=['wl_up'])
    # กรองค่า wl_up ที่ไม่เหมาะสม
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    # ตั้งค่า datetime เป็น index
    data_clean.set_index('datetime', inplace=True)
    # ทำการ resample และคำนวณค่าเฉลี่ย
    data_clean = data_clean.resample('15T').mean()
    data_clean = data_clean.interpolate(method='linear')
    # จัดการกับ spike
    data_clean.sort_index(inplace=True)
    data_clean['diff'] = data_clean['wl_up'].diff().abs()
    threshold = data_clean['diff'].median() * 5
    data_clean['is_spike'] = data_clean['diff'] > threshold
    data_clean.loc[data_clean['is_spike'], 'wl_up'] = np.nan
    data_clean.drop(columns=['diff', 'is_spike'], inplace=True)
    data_clean['wl_up'] = data_clean['wl_up'].interpolate(method='linear')
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
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month', 'wl_up_prev',
        'lag_1', 'lag_4', 'lag_96', 'lag_192', 'ma_672'
    ]

    data = data_clean.copy()

    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.DateOffset(hours=23, minutes=45)

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
    if 'wl_up_prev' in data_with_all_dates.columns:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up_prev'].interpolate(method='linear')
    else:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up'].shift(1).interpolate(method='linear')

    # สร้างฟีเจอร์ lag และค่าเฉลี่ยเคลื่อนที่
    data_with_all_dates = create_lag_features(data_with_all_dates, lags=[1, 4, 96, 192])
    data_with_all_dates = create_moving_average_features(data_with_all_dates, window=672)

    # เติมค่า missing ในฟีเจอร์ lag และ ma
    lag_cols = ['lag_1', 'lag_4', 'lag_96', 'lag_192']
    ma_col = 'ma_672'
    data_with_all_dates[lag_cols] = data_with_all_dates[lag_cols].interpolate(method='linear')
    data_with_all_dates[ma_col] = data_with_all_dates[ma_col].interpolate(method='linear')

    # Update data_missing and data_not_missing after adding lag and ma
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    if len(data_missing) == 0:
        st.write("ไม่มีค่าที่หายไปให้พยากรณ์")
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
def plot_results(data_before, data_filled, data_deleted, data_deleted_option=False):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
    })

    if data_deleted_option:
        data_after_deleted = pd.DataFrame({
            'วันที่': data_deleted['datetime'],
            'ข้อมูลหลังลบ': data_deleted['wl_up']
        })
    else:
        data_after_deleted = None

    # ผสานข้อมูลเพื่อให้แน่ใจว่าค่า wl_up ก่อนถูกลบแสดงในตาราง
    data_filled_with_original = pd.merge(
        data_filled,
        data_before[['datetime', 'wl_up']],
        on='datetime',
        how='left',
        suffixes=('', '_original')
    )

    # แทนที่ค่า 'wl_up' ใน data_filled ด้วยค่า wl_up ดั้งเดิม
    data_filled_with_original['wl_up'] = data_filled_with_original['wl_up_original']

    # รวมข้อมูลสำหรับกราฟ
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

    # แสดงตารางข้อมูลหลังเติมค่า
    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    data_filled_selected = data_filled_with_original[['code', 'datetime', 'wl_up', 'wl_forecast', 'timestamp']]
    st.dataframe(data_filled_selected, use_container_width=True)

    # ตรวจสอบว่ามีค่าจริงให้เปรียบเทียบหรือไม่ก่อนเรียกฟังก์ชันคำนวณความแม่นยำ
    merged_data = pd.merge(data_before[['datetime', 'wl_up']], data_filled[['datetime', 'wl_up2']], on='datetime')
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])
    comparison_data = merged_data[merged_data['wl_up2'] != merged_data['wl_up']]

    if data_deleted_option:
        calculate_accuracy_metrics(data_before, data_filled, data_deleted)
    else:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีการลบข้อมูล")

# ฟังก์ชันสำหรับแสดงกราฟตัวอย่างข้อมูล
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

    # กำหนดรายการ y ที่จะแสดงในกราฟ
    y_columns = ['สถานีที่ต้องการทำนาย']
    if df2_pre is not None:
        y_columns.append('สถานีน้ำ Upstream')
    if df3_pre is not None:
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
    st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับรวมข้อมูลจากสถานีต่างๆ
def merge_data(df1, df2=None, df3=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))
    else:
        merged_df = df1.copy()
    
    if df3 is not None:
        merged_df = pd.merge(merged_df, df3[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))
    return merged_df

def merge_data_linear(df1, df2=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_nearby'))
    else:
        # ถ้าไม่มี df2 ให้สร้างคอลัมน์ 'wl_up_nearby' จาก 'wl_up' ของ df1 (shifted by 1)
        df1['wl_up_nearby'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันสำหรับสร้างกราฟข้อมูลจากสถานีทั้งสอง (สำหรับ Linear Regression)
def plot_data_combined_LR_stations(data, forecasted=None, nearby_data=None, label='ระดับน้ำ'):
    # กราฟแรก: ข้อมูลจริง
    fig_actual = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำจริงที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    fig_actual.update_traces(connectgaps=False, name='สถานีที่ต้องการพยากรณ์')
    
    # แสดงค่าจริงของสถานีใกล้เคียง (ถ้ามี)
    if nearby_data is not None:
        fig_actual.add_scatter(x=nearby_data.index, y=nearby_data['wl_up'], mode='lines', name='สถานีใกล้เคียง', line=dict(color='green'))
    
    fig_actual.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)", legend_title="สถานี")

    # กราฟที่สอง: ค่าพยากรณ์เทียบกับค่าจริงในช่วงพยากรณ์
    if forecasted is not None and not forecasted.empty:
        # กำหนดช่วงเวลาพยากรณ์
        forecast_start = forecasted.index.min()
        forecast_end = forecasted.index.max()
        actual_forecast_period = data[(data.index >= forecast_start) & (data.index <= forecast_end)]
        
        fig_forecast = px.line(forecasted, x=forecasted.index, y='wl_up', title='ระดับน้ำที่พยากรณ์', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
        fig_forecast.update_traces(connectgaps=False, name='ค่าที่พยากรณ์', line=dict(color='red'))
        
        # เทียบกับค่าจริงในช่วงพยากรณ์
        if not actual_forecast_period.empty:
            fig_forecast.add_scatter(x=actual_forecast_period.index, y=actual_forecast_period['wl_up'], mode='lines', name='ค่าจริง', line=dict(color='blue'))
        
        fig_forecast.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)", legend_title="ประเภทข้อมูล")
    else:
        fig_forecast = None

    # แสดงกราฟ
    st.plotly_chart(fig_actual, use_container_width=True)
    if fig_forecast is not None:
        st.plotly_chart(fig_forecast, use_container_width=True)

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression สำหรับหลายสถานี (ใกล้เคียง)
def forecast_with_linear_regression_multi(data, forecast_start_date, forecast_days, nearby_data=None, delay_hours=0):
    # ตรวจสอบจำนวนวันที่พยากรณ์ให้อยู่ในขอบเขต 1-30 วัน
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    lags = [1, 4, 96, 192]  # lag 15 นาที, 1 ชั่วโมง, 1 วัน, 2 วัน

    # กำหนดฟีเจอร์ให้คงที่เสมอ
    feature_cols = [f'lag_{lag}' for lag in lags] + [f'lag_{lag}_nearby' for lag in lags]

    # เตรียมข้อมูลจาก nearby_data ถ้ามี
    if nearby_data is not None and not nearby_data.empty:
        nearby_data = nearby_data.copy()
        if delay_hours > 0:
            nearby_data.index = nearby_data.index + pd.Timedelta(hours=delay_hours)
    else:
        # สร้าง DataFrame เปล่าสำหรับ nearby
        nearby_data = pd.DataFrame({'wl_up': [0]*len(data)}, index=data.index)

    # สร้างชุดข้อมูลสำหรับการเทรน
    training_data = data.copy()
    training_data = training_data.join(nearby_data[['wl_up']], rsuffix='_nearby')

    # สร้างฟีเจอร์ lag
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        training_data[f'lag_{lag}_nearby'] = training_data['wl_up_nearby'].shift(lag)

    # เติมค่า NaN ด้วยค่าเฉลี่ยของ y_train ก่อนการเทรน
    training_data.fillna(method='ffill', inplace=True)
    training_data.dropna(inplace=True)

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
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
    combined_nearby = nearby_data.copy()

    # การพยากรณ์ทีละค่า
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ค่า lag ของสถานีหลัก
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_features[f'lag_{lag}'] = combined_data.at[lag_time, 'wl_up']
            else:
                lag_features[f'lag_{lag}'] = y_train.mean()
            # ค่า lag ของ nearby
            if lag_time in combined_nearby.index and not pd.isnull(combined_nearby.at[lag_time, 'wl_up']):
                lag_features[f'lag_{lag}_nearby'] = combined_nearby.at[lag_time, 'wl_up']
            else:
                lag_features[f'lag_{lag}_nearby'] = y_train.mean()

        # เตรียมข้อมูลสำหรับการพยากรณ์
        X_pred = pd.DataFrame([lag_features], columns=feature_cols)

        try:
            # พยากรณ์ค่า
            forecast_value = model.predict(X_pred)[0]

            # ป้องกันการกระโดดของค่าพยากรณ์
            forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
            
            forecasted_data.at[idx, 'wl_up'] = forecast_value

            # อัปเดต 'combined_data' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
            combined_data.at[idx, 'wl_up'] = forecast_value
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในเวลา {idx} ได้: {e}")

    return forecasted_data

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำของ Linear Regression
def calculate_accuracy_metrics_linear(original, forecasted):
    # ผสานข้อมูลตาม datetime
    merged_data = pd.merge(original[['datetime', 'wl_up']], forecasted.reset_index().rename(columns={'index': 'datetime'}), on='datetime')

    # ลบข้อมูลที่มี NaN ออก
    merged_data = merged_data.dropna(subset=['wl_up_x', 'wl_up_y'])

    if merged_data.empty:
        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None, merged_data

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(merged_data['wl_up_x'], merged_data['wl_up_y'])
    mae = mean_absolute_error(merged_data['wl_up_x'], merged_data['wl_up_y'])
    r2 = r2_score(merged_data['wl_up_x'], merged_data['wl_up_y'])

    return mse, mae, r2, merged_data

# ฟังก์ชันสำหรับสร้างตารางเปรียบเทียบ
def create_comparison_table_streamlit(forecasted_data, merged_data):
    comparison_df = pd.DataFrame({
        'Datetime': merged_data['datetime'],
        'ค่าจริง': merged_data['wl_up_x'],
        'ค่าที่พยากรณ์': merged_data['wl_up_y']
    })
    return comparison_df

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
            use_nearby_lr = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            # อัปโหลดไฟล์สถานีใกล้เคียงถ้าเลือกใช้
            if use_nearby_lr:
                uploaded_nearby_lr = st.file_uploader("ข้อมูลสถานีใกล้เคียง", type="csv", key="uploader_nearby_lr")
                delay_hours_lr = st.number_input("ระบุเวลาห่างระหว่างสถานีใกล้เคียง (ชั่วโมง)", value=0, min_value=0)
            else:
                uploaded_nearby_lr = None
                delay_hours_lr = 0

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

                # แสดงกราฟตัวอย่าง
                plot_data_preview(df_pre, df_up_pre, df_down_pre, total_time_lag_upstream, total_time_lag_downstream)

                if process_button:
                    processing_placeholder = st.empty()
                    processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                    # ปรับค่า end_date เฉพาะถ้าเลือกช่วงเวลาแล้ว
                    end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1)

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

                    # เก็บข้อมูลหลังการทำความสะอาดแต่ก่อนการรวมข้อมูล
                    df_before_deletion = df_clean.copy()

                    # รวมข้อมูลจากสถานี Upstream และ Downstream ถ้ามี
                    df_merged = merge_data(df_clean, df_up_clean, df_down_clean)

                    # ตรวจสอบว่าผู้ใช้เลือกที่จะลบข้อมูลหรือไม่
                    if delete_data_option:
                        delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
                        delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}") + pd.DateOffset(hours=23, minutes=45)
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
                    target_df_lr.set_index('datetime', inplace=True)

                    # โหลดข้อมูลสถานีใกล้เคียงถ้ามี
                    if use_nearby_lr and uploaded_nearby_lr is not None:
                        try:
                            nearby_df_lr = pd.read_csv(uploaded_nearby_lr)
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์สถานีใกล้เคียง: {e}")
                            nearby_df_lr = pd.DataFrame()

                        nearby_df_lr = clean_data(nearby_df_lr)
                        if not nearby_df_lr.empty:
                            nearby_df_lr = generate_missing_dates(nearby_df_lr)
                            nearby_df_lr['datetime'] = pd.to_datetime(nearby_df_lr['datetime'], errors='coerce').dt.tz_localize(None)
                            nearby_df_lr = create_time_features(nearby_df_lr)
                            nearby_df_lr.set_index('datetime', inplace=True)
                        else:
                            nearby_df_lr = None
                    else:
                        nearby_df_lr = None

                    # รวมข้อมูลจากสถานีใกล้เคียงถ้ามี
                    if use_nearby_lr and nearby_df_lr is not None:
                        merged_training_data_lr = merge_data_linear(target_df_lr.reset_index(), nearby_df_lr.reset_index())
                    else:
                        merged_training_data_lr = merge_data_linear(target_df_lr.reset_index())

                    if process_button_lr:
                        with st.spinner("กำลังพยากรณ์..."):
                            training_start_datetime_lr = pd.Timestamp.combine(training_start_date_lr, training_start_time_lr)
                            training_end_datetime_lr = pd.Timestamp.combine(training_end_date_lr, training_end_time_lr)
                            training_data_lr = merged_training_data_lr[
                                (merged_training_data_lr['datetime'] >= training_start_datetime_lr) & 
                                (merged_training_data_lr['datetime'] <= training_end_datetime_lr)
                            ].copy()
                            training_data_lr.set_index('datetime', inplace=True)

                            # ตรวจสอบว่ามีข้อมูลในช่วงที่เลือกหรือไม่
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
                                    data=training_data_lr,
                                    forecast_start_date=forecast_start_date_actual_lr,
                                    forecast_days=forecast_days_lr,
                                    nearby_data=nearby_df_lr if use_nearby_lr and nearby_df_lr is not None else None,
                                    delay_hours=delay_hours_lr if use_nearby_lr else 0
                                )

                                if not forecasted_data_lr.empty:
                                    st.header("กราฟข้อมูลพร้อมการพยากรณ์ (Linear Regression)")
                                    plot_data_combined_LR_stations(
                                        target_df_lr, 
                                        forecasted_data_lr, 
                                        nearby_df_lr if use_nearby_lr and nearby_df_lr is not None else None, 
                                        label='สถานีที่ต้องการทำนาย'
                                    )
                                    st.markdown("---")

                                    mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics_linear(
                                        original=target_df_lr.reset_index(),
                                        forecasted=forecasted_data_lr
                                    )

                                    if mse_lr is not None:
                                        st.header("ตารางข้อมูลเปรียบเทียบ")
                                        comparison_table_lr = create_comparison_table_streamlit(forecasted_data_lr, merged_data_lr)
                                        st.dataframe(comparison_table_lr, use_container_width=True)
                                        
                                        st.header("ผลค่าความแม่นยำ")
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
            st.error("กรุณาอัปโหลดไฟล์สำหรับ Linear Regression")




























