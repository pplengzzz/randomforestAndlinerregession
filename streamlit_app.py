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

# ฟังก์ชันจากโค้ดแรก (Random Forest)
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

# def fix_outliers_based_on_neighbors(data, threshold=0.5, decimal_threshold=0.01):
#     # คำนวณค่าเฉลี่ยระหว่างค่าแถวก่อนหน้าและแถวถัดไป
#     data['avg_neighbors'] = (data['wl_up'].shift(1) + data['wl_up'].shift(-1)) / 2
    
#     # คำนวณความแตกต่างระหว่างค่าปัจจุบันกับค่าเฉลี่ยของแถวข้างเคียง
#     data['diff_avg'] = (data['wl_up'] - data['avg_neighbors']).abs()
    
#     # ระบุแถวที่เป็น outlier โดยอิงตาม threshold ทั่วไป
#     large_outlier_condition = data['diff_avg'] > threshold
    
#     # ระบุแถวที่เป็น outlier โดยอิงตามความแตกต่างเล็กน้อย (decimal threshold)
#     small_outlier_condition = (data['diff_avg'] > decimal_threshold) & (data['diff_avg'] <= threshold)
    
#     # รวมเงื่อนไขการตรวจจับ outlier
#     outlier_condition = large_outlier_condition | small_outlier_condition
    
#     # ตั้งค่า NaN สำหรับค่า wl_up ที่เป็น outlier
#     data.loc[outlier_condition, 'wl_up'] = np.nan
    
#     # ใช้ interpolation เพื่อเติมค่าที่เป็น NaN
#     data['wl_up'] = data['wl_up'].interpolate(method='linear')
    
#     # ลบคอลัมน์ที่ไม่ต้องการหลังจากการประมวลผล
#     data.drop(columns=['avg_neighbors', 'diff_avg'], inplace=True)
    
#     return data

def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    
    # เรียกใช้ฟังก์ชันเพื่อจัดการกับค่า outliers ตามค่าเฉลี่ยของแถวข้างเคียง
    # data_clean = fix_outliers_based_on_neighbors(data_clean)
    
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

def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ฟังก์ชันเพิ่มเติมจากโค้ดแรก
def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

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
    data_filled_selected = data_filled[['code', 'datetime', 'wl_up', 'wl_forecast', 'timestamp']]
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

def plot_data_preview(df_pre, df2_pre, total_time_lag):
    data_pre1 = pd.DataFrame({
        'datetime': df_pre['datetime'],
        'สถานีที่ต้องการทำนาย': df_pre['wl_up']
    })

    if df2_pre is not None:
        data_pre2 = pd.DataFrame({
            'datetime': df2_pre['datetime'] + total_time_lag,  # ขยับวันที่ของสถานีก่อนหน้าตามเวลาห่างที่ระบุ
            'สถานีก่อนหน้า': df2_pre['wl_up']
        })
        combined_data_pre = pd.merge(data_pre1, data_pre2, on='datetime', how='outer')

        # Plot ด้วย Plotly และกำหนด color_discrete_sequence
        fig = px.line(
            combined_data_pre, 
            x='datetime', 
            y=['สถานีที่ต้องการทำนาย', 'สถานีก่อนหน้า'],
            labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
            title='ข้อมูลจากทั้งสองสถานี',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            xaxis_title="วันที่", 
            yaxis_title="ระดับน้ำ (wl_up)"
        )

        # แสดงกราฟ
        st.plotly_chart(fig, use_container_width=True)

    else:
        # ถ้าไม่มีไฟล์ที่สอง ให้แสดงกราฟของไฟล์แรกเท่านั้น
        fig = px.line(
            data_pre1, 
            x='datetime', 
            y='สถานีที่ต้องการทำนาย',
            labels={'สถานีที่ต้องการทำนาย': 'ระดับน้ำ (wl_up)'},
            title='ข้อมูลสถานีที่ต้องการทำนาย',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            xaxis_title="วันที่", 
            yaxis_title="ระดับน้ำ (wl_up)"
        )

        st.plotly_chart(fig, use_container_width=True)

def merge_data(df1, df2=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_prev'))
    else:
        # ถ้าไม่มี df2 ให้สร้างคอลัมน์ 'wl_up_prev' จาก 'wl_up' ของ df1 (shifted by 1)
        df1['wl_up_prev'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression ทีละค่า
def forecast_with_linear_regression_single(data, forecast_start_date, forecast_days):
    # ตรวจสอบจำนวนวันที่พยากรณ์ให้อยู่ในขอบเขต 1-30 วัน
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    # กำหนดช่วงเวลาการฝึกโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = forecast_start_date - pd.Timedelta(days=forecast_days) + pd.Timedelta(minutes=15)

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

    # สเกลข้อมูล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

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
        X_pred_scaled = scaler.transform(X_pred)

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred_scaled)[0]

        # ป้องกันการกระโดดของค่าพยากรณ์
        forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
        
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value

    return forecasted_data

# ฟังก์ชันสำหรับพยากรณ์ด้วย Linear Regression สองสถานี (ถ้ามี)
def forecast_with_linear_regression_two(data, upstream_data, forecast_start_date, forecast_days, delay_hours):
    # ตรวจสอบจำนวนวันที่พยากรณ์ให้อยู่ในขอบเขต 1-30 วัน
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    # เตรียมข้อมูลจาก upstream_data
    if not upstream_data.empty:
        upstream_data = upstream_data.copy()
        if delay_hours > 0:
            upstream_data.index = upstream_data.index + pd.Timedelta(hours=delay_hours)

    # กำหนดช่วงเวลาการฝึกโมเดล
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = forecast_start_date - pd.Timedelta(days=forecast_days) + pd.Timedelta(minutes=15)

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

    # เติมค่า NaN ด้วยค่าเฉลี่ยของ y_train ก่อนการเทรน
    training_data.fillna(method='ffill', inplace=True)
    training_data.dropna(inplace=True)

    # ตรวจสอบว่ามีข้อมูลเพียงพอหลังจากสร้างฟีเจอร์ lag
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    if not upstream_data.empty:
        feature_cols = [f'lag_{lag}' for lag in lags] + [f'lag_{lag}_upstream' for lag in lags]
    else:
        feature_cols = [f'lag_{lag}' for lag in lags]

    # กำหนดเฉพาะฟีเจอร์ที่มีอยู่จริงในข้อมูลการเทรน
    feature_cols = [col for col in feature_cols if col in training_data.columns]

    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # สเกลข้อมูล
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # เทรนโมเดล Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # สร้าง DataFrame สำหรับการพยากรณ์
    forecast_periods = forecast_days * 96  # พยากรณ์ตามจำนวนวันที่เลือก (96 ช่วงเวลา 15 นาทีต่อวัน)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # สร้างชุดข้อมูลสำหรับการพยากรณ์
    combined_data = data.copy()
    if not upstream_data.empty:
        combined_upstream = upstream_data.copy()

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
            if not upstream_data.empty:
                if lag_time in combined_upstream.index and not pd.isnull(combined_upstream.at[lag_time, 'wl_up']):
                    lag_value_upstream = combined_upstream.at[lag_time, 'wl_up']
                else:
                    lag_value_upstream = y_train.mean()
                lag_features[f'lag_{lag}_upstream'] = lag_value_upstream

        # เตรียมข้อมูลสำหรับการพยากรณ์
        X_pred = pd.DataFrame([lag_features])
        X_pred_scaled = scaler.transform(X_pred)

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred_scaled)[0]

        # ป้องกันการกระโดดของค่าพยากรณ์
        forecast_value = np.clip(forecast_value, combined_data['wl_up'].min(), combined_data['wl_up'].max())
        
        forecasted_data.at[idx, 'wl_up'] = forecast_value

        # อัปเดต 'combined_data' และ 'combined_upstream' ด้วยค่าที่พยากรณ์เพื่อใช้ในการพยากรณ์ครั้งถัดไป
        combined_data.at[idx, 'wl_up'] = forecast_value
        if not upstream_data.empty:
            combined_upstream.at[idx, 'wl_up'] = lag_features.get(f'lag_{lag}_upstream', y_train.mean())

    return forecasted_data

# ฟังก์ชันสำหรับสร้างกราฟข้อมูลพร้อมการพยากรณ์
def plot_data_combined_two_stations(data, forecasted=None, upstream_data=None, label='ระดับน้ำ'):
    fig = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    fig.update_traces(connectgaps=False)
    
    # แสดงค่าจริงของสถานีที่ต้องการพยากรณ์
    fig.add_scatter(x=data.index, y=data['wl_up'], mode='lines', name='ค่าจริง (สถานีที่พยากรณ์)', line=dict(color='blue'))
    
    # แสดงค่าจริงของสถานี upstream (ถ้ามี)
    if upstream_data is not None:
        fig.add_scatter(x=upstream_data.index, y=upstream_data['wl_up'], mode='lines', name='ค่าจริง (สถานี upstream)', line=dict(color='green'))

    # แสดงค่าพยากรณ์
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig


# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำ
def calculate_error_metrics(original, forecasted):
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

# ฟังก์ชันสำหรับแสดงตารางเปรียบเทียบ
def create_comparison_table_streamlit(forecasted_data, actual_data):
    comparison_df = pd.DataFrame({
        'Datetime': actual_data['datetime'],
        'ค่าจริง': actual_data['Actual'],
        'ค่าที่พยากรณ์': actual_data['Forecasted']
    })
    return comparison_df

# ฟังก์ชันสำหรับแสดงกราฟตัวอย่างข้อมูล
def plot_data_preview(df_pre, df2_pre, total_time_lag):
    data_pre1 = pd.DataFrame({
        'datetime': df_pre['datetime'],
        'สถานีที่ต้องการทำนาย': df_pre['wl_up']
    })

    if df2_pre is not None:
        data_pre2 = pd.DataFrame({
            'datetime': df2_pre['datetime'] + total_time_lag,  # ขยับวันที่ของสถานีก่อนหน้าตามเวลาห่างที่ระบุ
            'สถานีก่อนหน้า': df2_pre['wl_up']
        })
        combined_data_pre = pd.merge(data_pre1, data_pre2, on='datetime', how='outer')

        # Plot ด้วย Plotly และกำหนด color_discrete_sequence
        fig = px.line(
            combined_data_pre, 
            x='datetime', 
            y=['สถานีที่ต้องการทำนาย', 'สถานีก่อนหน้า'],
            labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
            title='ข้อมูลจากทั้งสองสถานี',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            xaxis_title="วันที่", 
            yaxis_title="ระดับน้ำ (wl_up)"
        )

        # แสดงกราฟ
        st.plotly_chart(fig, use_container_width=True)

    else:
        # ถ้าไม่มีไฟล์ที่สอง ให้แสดงกราฟของไฟล์แรกเท่านั้น
        fig = px.line(
            data_pre1, 
            x='datetime', 
            y='สถานีที่ต้องการทำนาย',
            labels={'สถานีที่ต้องการทำนาย': 'ระดับน้ำ (wl_up)'},
            title='ข้อมูลสถานีที่ต้องการทำนาย',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_layout(
            xaxis_title="วันที่", 
            yaxis_title="ระดับน้ำ (wl_up)"
        )

        st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับรวมข้อมูลจากสองสถานี
def merge_data(df1, df2=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_prev'))
    else:
        # ถ้าไม่มี df2 ให้สร้างคอลัมน์ 'wl_up_prev' จาก 'wl_up' ของ df1 (shifted by 1)
        df1['wl_up_prev'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# Streamlit UI
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
    model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"),
        label_visibility="collapsed"  # ซ่อน label visibility
    )

    st.sidebar.title("ตั้งค่าข้อมูล")
    if model_choice == "Random Forest":
        with st.sidebar.expander("ตั้งค่า Random Forest", expanded=False):
            use_second_file = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            # สลับตำแหน่งการอัปโหลดไฟล์
            if use_second_file:
                uploaded_file2 = st.file_uploader("ข้อมูลระดับที่ใช้ฝึกโมเดล", type="csv", key="uploader2")
                uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1")
            else:
                uploaded_file2 = None  # กำหนดให้เป็น None ถ้าไม่ใช้ไฟล์ที่สอง
                uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1")

            # เพิ่มช่องกรอกเวลาห่างระหว่างสถานี ถ้าใช้ไฟล์ที่สอง
            if use_second_file:
                time_lag_days = st.number_input("ระบุเวลาห่างระหว่างสถานี (วัน)", value=0, min_value=0)
                total_time_lag = pd.Timedelta(days=time_lag_days)
            else:
                total_time_lag = pd.Timedelta(days=0)

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

        process_button = st.button("ประมวลผล", type="primary")

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_upstream = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)

            if use_upstream:
                uploaded_up_file = st.file_uploader("ข้อมูลระดับที่ใช้ฝึกโมเดล", type="csv", key="uploader_up_lr")

            uploaded_fill_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader_fill_lr")
            
            if use_upstream:
                delay_hours = st.number_input("ระบุเวลาห่างระหว่างสถานี (ชั่วโมง)", value=0, min_value=0)

        # แยกการเลือกช่วงข้อมูลสำหรับฝึกโมเดลและการพยากรณ์
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            training_start_date = st.date_input("วันที่เริ่มต้นฝึกโมเดล", value=pd.to_datetime("2024-05-01"), key='training_start_lr')
            training_start_time = st.time_input("เวลาเริ่มต้นฝึกโมเดล", value=pd.Timestamp("00:00:00").time(), key='training_start_time_lr')
            training_end_date = st.date_input("วันที่สิ้นสุดฝึกโมเดล", value=pd.to_datetime("2024-05-31"), key='training_end_lr')
            training_end_time = st.time_input("เวลาสิ้นสุดฝึกโมเดล", value=pd.Timestamp("23:45:00").time(), key='training_end_time_lr')

        with st.sidebar.expander("ตั้งค่าการพยากรณ์", expanded=False):
            forecast_days = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)

        process_button2 = st.button("ประมวลผล", type="primary")

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    # ส่วนนี้คุณไม่ต้องการปรับปรุง
    pass
elif model_choice == "Linear Regression":
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

                # แสดงกราฟข้อมูลจากทั้งสองสถานี
                if use_upstream and upstream_df is not None:
                    # ใช้ฟังก์ชัน plot_data_preview เพื่อแสดงกราฟข้อมูลทั้งสองสถานี
                    plot_data_preview(target_df, upstream_df, pd.Timedelta(hours=delay_hours))
                else:
                    # แสดงเฉพาะกราฟของสถานีที่ต้องการทำนาย หากไม่มีสถานีใกล้เคียง
                    plot_data_preview(target_df, None, pd.Timedelta(hours=0))

                if process_button2:
                    with st.spinner("กำลังพยากรณ์..."):
                        # รวมวันที่และเวลาเพื่อสร้าง datetime สำหรับการพยากรณ์
                        training_start_datetime = pd.Timestamp.combine(training_start_date, training_start_time)
                        training_end_datetime = pd.Timestamp.combine(training_end_date, training_end_time)

                        # กรองข้อมูลสำหรับการฝึกโมเดลตามช่วงเวลาที่เลือก
                        training_data = target_df[(target_df['datetime'] >= training_start_datetime) & (target_df['datetime'] <= training_end_datetime)].copy()

                        if training_data.empty:
                            st.error("ไม่มีข้อมูลในช่วงเวลาที่เลือกสำหรับการฝึกโมเดล")
                        else:
                            forecast_start_date_actual = training_end_datetime + pd.Timedelta(minutes=15)

                            # ตรวจสอบว่าช่วงเวลาพยากรณ์ไม่เกินขอบเขตข้อมูลจริง
                            forecast_end_date_actual = forecast_start_date_actual + pd.Timedelta(days=forecast_days)
                            max_datetime = target_df['datetime'].max()
                            if forecast_end_date_actual > max_datetime:
                                st.warning("ข้อมูลจริงในช่วงเวลาที่พยากรณ์ไม่ครบถ้วนหรือไม่มีข้อมูล")

                            if use_upstream and upstream_df is not None and not upstream_df.empty:
                                # พยากรณ์ด้วย Linear Regression (สองสถานี)
                                forecasted_data = forecast_with_linear_regression_two(
                                    data=target_df.set_index('datetime'),
                                    upstream_data=upstream_df.set_index('datetime'),
                                    forecast_start_date=forecast_start_date_actual,
                                    forecast_days=forecast_days,
                                    delay_hours=delay_hours
                                )
                            else:
                                # พยากรณ์ด้วย Linear Regression (สถานีเดียว)
                                forecasted_data = forecast_with_linear_regression_single(
                                    data=target_df.set_index('datetime'),
                                    forecast_start_date=forecast_start_date_actual,
                                    forecast_days=forecast_days
                                )

                            if not forecasted_data.empty:
                                st.header("กราฟข้อมูลพร้อมการพยากรณ์", divider='gray')
                                st.plotly_chart(plot_data_combined_two_stations(target_df.set_index('datetime'), forecasted_data, label='สถานีที่ต้องการทำนาย'))

                                # ตรวจสอบและคำนวณค่าความแม่นยำ
                                mae, rmse, actual_forecasted_data = calculate_error_metrics(
                                    original=target_df,
                                    forecasted=forecasted_data
                                )

                                if actual_forecasted_data is not None:
                                    st.header("ตารางข้อมูลเปรียบเทียบ", divider='gray')
                                    comparison_table = create_comparison_table_streamlit(forecasted_data, actual_forecasted_data)
                                    st.dataframe(comparison_table, use_container_width=True)

                                    st.header("ผลค่าความแม่นยำ", divider='gray')
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
                                    with col2:
                                        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
                                else:
                                    st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                            else:
                                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Linear Regression")








