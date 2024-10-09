import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ฟังก์ชันจากโค้ดที่สอง (Linear Regression)
def plot_data(data, forecasted=None, label='ระดับน้ำ'):
    # ใช้ 'wl_up' ซึ่งเป็นคอลัมน์ระดับน้ำที่ถูกต้อง และตั้งค่า connectgaps=False
    fig = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    fig.update_traces(connectgaps=False)  # ไม่เชื่อมเส้นในกรณีที่ไม่มีข้อมูล
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig

def forecast_with_linear_regression(up_data, target_data, forecast_start_date):
    # ใช้ข้อมูล 672 แถวสุดท้ายจาก up_data ในการเทรนโมเดล
    training_data = up_data.iloc[-672:].copy()

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
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ดึงค่าจากข้อมูลจริงเท่านั้น
            if lag_time in up_data.index:
                lag_value = up_data.at[lag_time, 'wl_up']
            else:
                lag_value = np.nan
            lag_features[f'lag_{lag}'] = lag_value

        # ถ้ามีค่า lag ที่หายไป ให้ข้ามการพยากรณ์
        if np.any(pd.isnull(list(lag_features.values()))):
            continue

        # สร้าง DataFrame สำหรับฟีเจอร์ที่จะใช้ในการพยากรณ์
        X_pred = pd.DataFrame([lag_features])

        # พยากรณ์ค่า
        forecast_value = model.predict(X_pred)[0]
        forecasted_data.at[idx, 'wl_up'] = forecast_value

    # ลบแถวที่ไม่มีการพยากรณ์
    forecasted_data.dropna(inplace=True)

    return forecasted_data

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
    end_date = pd.to_datetime(end_date)

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
    
    # ตรวจสอบว่ามีข้อมูลที่สามารถเปรียบเทียบได้หรือไม่ (wl_up2 มีการเติมค่า)
    # สมมติว่า wl_up2 จะต่างจาก wl_up ถ้ามีการเติมค่า
    comparison_data = merged_data[merged_data['wl_up2'] != merged_data['wl_up']]
    
    if comparison_data.empty:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีค่าจริงให้เปรียบเทียบ")
    else:
        # คำนวณค่าความแม่นยำ
        mse = mean_squared_error(comparison_data['wl_up'], comparison_data['wl_up2'])
        mae = mean_absolute_error(comparison_data['wl_up'], comparison_data['wl_up2'])
        r2 = r2_score(comparison_data['wl_up'], comparison_data['wl_up2'])
    
        st.header("ผลค่าความแม่นยำ", divider='gray')
    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
        with col2:
            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
        with col3:
            st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

def plot_results(data_before, data_filled, data_deleted):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up2']
    })

    data_after_deleted = pd.DataFrame({
        'วันที่': data_deleted['datetime'],
        'ข้อมูลหลังลบ': data_deleted['wl_up']
    })

    # รวมข้อมูล
    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')
    combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    # Plot ด้วย Plotly
    fig = px.line(combined_data, x='วันที่', y=['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ'],
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  title="ข้อมูลหลังจากการเติมค่าที่หายไป")
    
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    
    # แสดงกราฟ
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

def plot_data_preview(data1, data2, total_time_lag):
    data_pre1 = pd.DataFrame({
        'วันที่': data1['datetime'],
        'สถานีที่ต้องการทำนาย': data1['wl_up']
    })

    if data2 is not None:
        data_pre2 = pd.DataFrame({
            'วันที่': data2['datetime'] + total_time_lag,  # ขยับวันที่ของสถานีก่อนหน้าตามเวลาห่างที่ระบุ
            'สถานีก่อนหน้า': data2['wl_up']
        })
        combined_data_pre = pd.merge(data_pre1, data_pre2, on='วันที่', how='outer')
        
        # กำหนดโทนสีแดงสำหรับแต่ละประเภทข้อมูล
        red_colors = ['#FF9999', '#FF4C4C']  # สีแดงอ่อนและสีแดงเข้ม

        fig = px.line(
            combined_data_pre,
            x='วันที่',
            y=['สถานีที่ต้องการทำนาย', 'สถานีก่อนหน้า'],
            labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
            title='ข้อมูลจากทั้งสองสถานี',
            color_discrete_sequence=red_colors
        )
        
        fig.update_layout(
            xaxis_title="วันที่",
            yaxis_title="ระดับน้ำ (wl_up)",
            legend_title="ประเภทข้อมูล",
            hovermode="x unified"
        )
        
        # เพิ่มการอัปเดตเลย์เอาต์เพื่อให้สามารถซูมและเลื่อนได้
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # ถ้าไม่มีไฟล์ที่สอง ให้แสดงกราฟของไฟล์แรกเท่านั้น
        red_colors_single = ['#FF4C4C']  # สีแดงเข้มสำหรับสายข้อมูลเดียว

        fig = px.line(
            data_pre1,
            x='วันที่',
            y='สถานีที่ต้องการทำนาย',
            labels={'สถานีที่ต้องการทำนาย': 'ระดับน้ำ (wl_up)'},
            title='ข้อมูลสถานี',
            color_discrete_sequence=red_colors_single
        )
        
        fig.update_layout(
            xaxis_title="วันที่",
            yaxis_title="ระดับน้ำ (wl_up)",
            hovermode="x unified"
        )
        
        # เพิ่มการอัปเดตเลย์เอาต์เพื่อให้สามารถซูมและเลื่อนได้
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)

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

แอป Streamlit สำหรับจัดการข้อมูลระดับน้ำ โดยใช้โมเดล **Random Forest** หรือ **Linear Regression** เพื่อเติมค่าที่ขาดหายไปและพยากรณ์ข้อมูล
ข้อมูลถูกประมวลผลและแสดงผลผ่านกราฟและการวัดค่าความแม่นยำ ผู้ใช้สามารถเลือกอัปโหลดไฟล์, 
กำหนดช่วงเวลาลบข้อมูล และเลือกวิธีการพยากรณ์ได้
""")
st.markdown("---")

# Sidebar: Upload files and choose date ranges
with st.sidebar:

    st.sidebar.title("เลือกวิธีการพยากรณ์")
    with st.sidebar.expander("ตั้งค่า Random Forest", expanded=True):
        model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"))

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
            uploaded_up_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีข้างบน (up)", type="csv")
            uploaded_target_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีที่ต้องการทำนาย", type="csv")

            # ย้ายการเลือกช่วงวันที่และเวลาไปที่ Sidebar
        with st.sidebar.expander("เลือกช่วงวันสำหรับพยากรณ์ในอีก 1 วันข้างหน้า", expanded=False):
            forecast_start_date = st.date_input("เลือกวันเริ่มต้นพยากรณ์", value=pd.to_datetime("2024-06-01"))
            forecast_start_time = st.time_input("เลือกเวลาเริ่มต้นพยากรณ์", value=pd.Timestamp("00:00:00").time())
            forecast_end_date = st.date_input("เลือกวันสิ้นสุดพยากรณ์", value=pd.to_datetime("2024-06-02"))
            forecast_end_time = st.time_input("เลือกเวลาสิ้นสุดพยากรณ์", value=pd.Timestamp("23:45:00").time())

        process_button2 = st.button("ประมวลผล", type="primary")

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            df_pre = clean_data(df)
            df_pre = generate_missing_dates(df_pre)

            # ถ้าเลือกใช้ไฟล์ที่สอง
            if use_second_file:
                if uploaded_file2 is not None:
                    df2 = load_data(uploaded_file2)
                    if df2 is not None:
                        df2_pre = clean_data(df2)
                        df2_pre = generate_missing_dates(df2_pre)
                    else:
                        df2_pre = None
                else:
                    st.warning("กรุณาอัปโหลดไฟล์ที่สอง (สถานีที่ก่อนหน้า)")
                    df2_pre = None
            else:
                df2_pre = None

            # แสดงกราฟตัวอย่าง
            plot_data_preview(df_pre, df2_pre, total_time_lag)

            if process_button:
                processing_placeholder = st.empty()
                processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                # ปรับค่า end_date เฉพาะถ้าเลือกช่วงเวลาแล้ว
                end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1)

                # กรองข้อมูลตามช่วงวันที่เลือก
                df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date_dt))]

                if use_second_file and uploaded_file2 and df2 is not None:
                    # ปรับเวลาของสถานีก่อนหน้าตามเวลาห่างที่ระบุ
                    df2['datetime'] = pd.to_datetime(df2['datetime']).dt.tz_localize(None)
                    df2_filtered = df2[(df2['datetime'] >= pd.to_datetime(start_date)) & (df2['datetime'] <= pd.to_datetime(end_date_dt))]
                    df2_filtered['datetime'] = df2_filtered['datetime'] + total_time_lag
                    df2_clean = clean_data(df2_filtered)
                else:
                    df2_clean = None

                # Clean data
                df_clean = clean_data(df_filtered)

                # รวมข้อมูลจากทั้งสองสถานี ถ้ามี
                df_merged = merge_data(df_clean, df2_clean)

                # ตรวจสอบว่าผู้ใช้เลือกที่จะลบข้อมูลหรือไม่
                if delete_data_option:
                    delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
                    delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}")
                    df_deleted = delete_data_by_date_range(df_merged, delete_start_datetime, delete_end_datetime)
                else:
                    df_deleted = df_merged.copy()  # ถ้าไม่เลือกลบก็ใช้ข้อมูลเดิมแทน

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

                # เก็บข้อมูลก่อนการลบ
                df_before_deletion = df_filtered.copy()

                # Handle missing values by week
                df_handled = handle_missing_values_by_week(df_clean, start_date, end_date, model_type='random_forest')

                # Remove the processing message after the processing is complete
                processing_placeholder.empty()

                # Plot the results using Streamlit's line chart
                plot_results(df_before_deletion, df_handled, df_deleted)
        st.markdown("---")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผล")
elif model_choice == "Linear Regression":
    if uploaded_up_file is not None and uploaded_target_file is not None:
        # อ่านข้อมูลจากไฟล์ CSV
        up_data = pd.read_csv(uploaded_up_file)
        target_data = pd.read_csv(uploaded_target_file)

        # แปลง column datetime ให้เป็น datetime และตั้งให้เป็น index
        for data in [up_data, target_data]:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data['datetime'] = data['datetime'].dt.tz_localize(None)
            data.set_index('datetime', inplace=True)

        # แสดงกราฟข้อมูล
        st.subheader('กราฟข้อมูลระดับน้ำ')
        st.plotly_chart(plot_data(up_data, label='สถานีข้างบน (up)'))
        st.plotly_chart(plot_data(target_data, label='สถานีที่ต้องการทำนาย'))

        # ส่วนการเลือกช่วงวันที่สำหรับพยากรณ์ถูกย้ายไปที่ Sidebar

        if process_button2 :
            # แสดงข้อความกำลังประมวลผล
            with st.spinner("กำลังพยากรณ์..."):
                # รวมวันที่และเวลา
                start_datetime = pd.Timestamp.combine(forecast_start_date, forecast_start_time)
                end_datetime = pd.Timestamp.combine(forecast_end_date, forecast_end_time)

                if start_datetime > end_datetime:
                    st.error("วันและเวลาที่เริ่มต้นต้องไม่เกินวันและเวลาสิ้นสุด")
                else:
                    # เลือกข้อมูลช่วงวันที่และเวลาที่สนใจ
                    selected_data = target_data[(target_data.index >= start_datetime) & (target_data.index <= end_datetime)].copy()

                    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่
                    if selected_data.empty:
                        st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
                    else:
                        # กำหนดวันที่เริ่มพยากรณ์เป็นเวลาถัดไปจากข้อมูลที่เลือก
                        forecast_start_date_actual = selected_data.index.max() + pd.Timedelta(minutes=15)

                        # พยากรณ์
                        forecasted_data = forecast_with_linear_regression(up_data, target_data, forecast_start_date_actual)

                        # ตรวจสอบว่ามีการพยากรณ์หรือไม่
                        if not forecasted_data.empty:
                            # แสดงกราฟข้อมูลพร้อมการพยากรณ์
                            st.subheader('กราฟข้อมูลพร้อมการพยากรณ์')
                            st.plotly_chart(plot_data(selected_data, forecasted_data, label='สถานีที่ต้องการทำนาย'))

                            # ตรวจสอบว่ามีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์หรือไม่
                            common_indices = forecasted_data.index.intersection(target_data.index)
                            if not common_indices.empty:
                                actual_data = target_data.loc[common_indices]
                                y_true = actual_data['wl_up']
                                y_pred = forecasted_data['wl_up'].loc[common_indices]

                                # ทำให้ y_true และ y_pred มีขนาดเท่ากัน
                                min_length = min(len(y_true), len(y_pred))
                                y_true = y_true[:min_length]
                                y_pred = y_pred[:min_length]

                                mae = mean_absolute_error(y_true, y_pred)
                                rmse = mean_squared_error(y_true, y_pred, squared=False)

                                # แสดงตารางที่มี datetime, ค่าจริง, ค่าพยากรณ์
                                st.subheader('ตารางข้อมูลเปรียบเทียบ')
                                comparison_table = pd.DataFrame({
                                    'datetime': forecasted_data.index[:min_length],
                                    'ค่าจริง (ถ้ามี)': y_true.values,
                                    'ค่าที่พยากรณ์': y_pred.values
                                })
                                st.dataframe(comparison_table)

                                # แสดงค่า MAE และ RMSE
                                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                            else:
                                st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                        else:
                            st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV สำหรับทั้งสองสถานีเพื่อเริ่มต้นการพยากรณ์")
