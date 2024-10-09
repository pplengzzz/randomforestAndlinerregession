import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ฟังก์ชันโหลดข้อมูล
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

# ฟังก์ชันทำความสะอาดข้อมูล
def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= 100) & (data_clean['wl_up'] <= 450)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    return data_clean

# ฟังก์ชันสร้างฟีเจอร์เวลา
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

# ฟังก์ชันเตรียมฟีเจอร์
def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month', 'wl_up_prev'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

# ฟังก์ชันฝึกและประเมินโมเดล
def train_and_evaluate_model(X, y, model_type='random_forest'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
    elif model_type == 'linear_regression':
        model = train_linear_regression_model(X_train, y_train)
    else:
        st.error("โมเดลที่เลือกไม่ถูกต้อง")
        return None

    if model is None:
        st.error("การฝึกโมเดลล้มเหลว")
        return None
    return model

# ฟังก์ชันฝึก Random Forest
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

# ฟังก์ชันฝึก Linear Regression
def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ฟังก์ชันสร้างช่วงวันที่ครบถ้วน
def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

# ฟังก์ชันเติมคอลัมน์ 'code'
def fill_code_column(data):
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

# ฟังก์ชันจัดการกับค่าที่หายไป
def handle_missing_values_by_week(data_clean, start_date, end_date, model_type='random_forest'):
    feature_cols = ['year', 'month', 'day', 'hour', 'minute',
                    'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month', 'wl_up_prev']

    data = data_clean.copy()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    data_with_all_dates = generate_missing_dates(data)
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates['datetime'])
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    if 'wl_up_prev' in data_with_all_dates.columns:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up_prev'].interpolate(method='linear')
    else:
        data_with_all_dates['wl_up_prev'] = data_with_all_dates['wl_up'].shift(1).interpolate(method='linear')

    if len(data_missing) == 0:
        st.write("No missing values to predict.")
        return data_with_all_dates

    X_train, y_train = prepare_features(data_not_missing)
    model = train_and_evaluate_model(X_train, y_train, model_type=model_type)

    if model is None:
        st.error("ไม่สามารถสร้างโมเดลได้ กรุณาตรวจสอบข้อมูล")
        return data_with_all_dates

    for idx, row in data_missing.iterrows():
        X_missing = row[feature_cols].values.reshape(1, -1)
        try:
            predicted_value = model.predict(X_missing)[0]
            data_with_all_dates.loc[idx, 'wl_forecast'] = predicted_value
            data_with_all_dates.loc[idx, 'timestamp'] = pd.Timestamp.now()
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในแถว {idx} ได้: {e}")
            continue

    data_with_all_dates['wl_up2'] = data_with_all_dates['wl_up'].combine_first(data_with_all_dates['wl_forecast'])

    data_with_all_dates.reset_index(drop=True, inplace=True)
    return data_with_all_dates

# ฟังก์ชันลบข้อมูลตามช่วงวันที่
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

# ฟังก์ชันคำนวณค่าความแม่นยำ
def calculate_accuracy_metrics(original, filled):
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')

    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])

    comparison_data = merged_data[merged_data['wl_up2'] != merged_data['wl_up']]

    if comparison_data.empty:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีค่าจริงให้เปรียบเทียบ")
    else:
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

# ฟังก์ชันแสดงผลลัพธ์
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

    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')
    combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    fig = px.line(combined_data, x='วันที่', y=['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ'],
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  title="ข้อมูลหลังจากการเติมค่าที่หายไป")

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    st.plotly_chart(fig, use_container_width=True)

    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    data_filled_selected = data_filled[['code', 'datetime', 'wl_up', 'wl_forecast', 'timestamp']]
    st.dataframe(data_filled_selected, use_container_width=True)

    merged_data = pd.merge(data_before[['datetime', 'wl_up']], data_filled[['datetime', 'wl_up2']], on='datetime')
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])
    comparison_data = merged_data[merged_data['wl_up2'] != merged_data['wl_up']]

    if comparison_data.empty:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีค่าจริงให้เปรียบเทียบ")
    else:
        calculate_accuracy_metrics(data_before, data_filled)

# ฟังก์ชันแสดงตัวอย่างข้อมูล
def plot_data_preview(data1, data2, total_time_lag):
    data_pre1 = pd.DataFrame({
        'วันที่': data1['datetime'],
        'สถานีที่ต้องการทำนาย': data1['wl_up']
    })

    if data2 is not None:
        data_pre2 = pd.DataFrame({
            'วันที่': data2['datetime'] + total_time_lag,
            'สถานีก่อนหน้า': data2['wl_up']
        })
        combined_data_pre = pd.merge(data_pre1, data_pre2, on='วันที่', how='outer')

        red_colors = ['#FF9999', '#FF4C4C']

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

        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        red_colors_single = ['#FF4C4C']

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

        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันรวมข้อมูลจากสถานีสอง
def merge_data(df1, df2=None):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_prev'))
    else:
        df1['wl_up_prev'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันพยากรณ์ด้วย Linear Regression จากโค้ดที่ 2 และ 3
def forecast_with_linear_regression(target_data, upstream_data, forecast_start_date, delay_hours=0):
    # เตรียมข้อมูลจาก upstream_data
    upstream_data = upstream_data.copy()
    if delay_hours > 0:
        upstream_data['datetime'] = upstream_data.index + pd.Timedelta(hours=delay_hours)
        upstream_data.set_index('datetime', inplace=True)

    # ใช้ข้อมูล 672 แถวสุดท้ายจาก upstream_data ในการเทรนโมเดล
    training_data = upstream_data.iloc[-672:].copy()

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
            if lag_time in target_data.index:
                lag_value = target_data.at[lag_time, 'wl_up']
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

# ฟังก์ชันแสดงกราฟข้อมูลพร้อมการพยากรณ์
def plot_data_combined(data, forecasted=None, label='ระดับน้ำ'):
    fig = px.line(data, x=data.index, y='wl_up', title=f'ระดับน้ำที่สถานี {label}', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
    fig.update_traces(connectgaps=False)
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    return fig

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
    with st.sidebar.expander("ตั้งค่าโมเดล", expanded=True):
        model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"))

    st.sidebar.title("ตั้งค่าข้อมูล")
    if model_choice == "Random Forest":
        with st.sidebar.expander("ตั้งค่า Random Forest", expanded=False):
            use_second_file = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            if use_second_file:
                uploaded_file2 = st.file_uploader("ข้อมูลระดับที่ใช้ฝึกโมเดล (สถานีที่ก่อนหน้า)", type="csv", key="uploader2_rf")
                uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1_rf")
            else:
                uploaded_file2 = None
                uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1_rf")

            if use_second_file:
                time_lag_days = st.number_input("ระบุเวลาห่างระหว่างสถานี (วัน)", value=0, min_value=0)
                total_time_lag = pd.Timedelta(days=time_lag_days)
            else:
                total_time_lag = pd.Timedelta(days=0)

        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-05-01"))
            end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-05-31"))
            
            delete_data_option = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

            if delete_data_option:
                st.header("เลือกช่วงที่ต้องการลบข้อมูล")
                delete_start_date = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date, key='delete_start_rf')
                delete_start_time = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time_rf')
                delete_end_date = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date, key='delete_end_rf')
                delete_end_time = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time_rf')

        process_button = st.button("ประมวลผล", type="primary")

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            uploaded_up_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีข้างบน (up)", type="csv")
            uploaded_target_file = st.file_uploader("เลือกไฟล์ CSV ของสถานีที่ต้องการทำนาย", type="csv")
            uploaded_fill_file = st.file_uploader("เลือกไฟล์ CSV สำหรับเติมข้อมูล", type="csv")

        with st.sidebar.expander("ตั้งค่าพยากรณ์", expanded=False):
            forecast_start_date = st.date_input("เลือกวันเริ่มต้นพยากรณ์", value=pd.to_datetime("2024-06-01"))
            forecast_start_time = st.time_input("เลือกเวลาเริ่มต้นพยากรณ์", value=pd.Timestamp("00:00:00").time())
            forecast_end_date = st.date_input("เลือกวันสิ้นสุดพยากรณ์", value=pd.to_datetime("2024-06-02"))
            forecast_end_time = st.time_input("เลือกเวลาสิ้นสุดพยากรณ์", value=pd.Timestamp("23:45:00").time())
            delay_hours = st.number_input("ระบุชั่วโมงหน่วงเวลาสำหรับสถานีข้างบน (up)", value=0, min_value=0)

            use_second_file_lr = st.checkbox("ต้องการใช้สถานีใกล้เคียงสำหรับ Linear Regression", value=False)

            if use_second_file_lr:
                uploaded_file2_lr = st.file_uploader("ข้อมูลระดับที่ใช้ฝึกโมเดล (สถานีที่ก่อนหน้า)", type="csv", key="uploader2_lr")
            else:
                uploaded_file2_lr = None

        process_button2 = st.button("ประมวลผล", type="primary")

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            df_pre = clean_data(df)
            df_pre = generate_missing_dates(df_pre)

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

            plot_data_preview(df_pre, df2_pre, total_time_lag)

            if process_button:
                processing_placeholder = st.empty()
                processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

                end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1)

                df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date_dt))]

                if use_second_file and uploaded_file2 and df2 is not None:
                    df2['datetime'] = pd.to_datetime(df2['datetime']).dt.tz_localize(None)
                    df2_filtered = df2[(df2['datetime'] >= pd.to_datetime(start_date)) & (df2['datetime'] <= pd.to_datetime(end_date_dt))]
                    df2_filtered['datetime'] = df2_filtered['datetime'] + total_time_lag
                    df2_clean = clean_data(df2_filtered)
                else:
                    df2_clean = None

                df_clean = clean_data(df_filtered)

                df_merged = merge_data(df_clean, df2_clean)

                if delete_data_option:
                    delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
                    delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}")
                    df_deleted = delete_data_by_date_range(df_merged, delete_start_datetime, delete_end_datetime)
                else:
                    df_deleted = df_merged.copy()

                df_clean = generate_missing_dates(df_deleted)
                df_clean = fill_code_column(df_clean)
                df_clean = create_time_features(df_clean)

                if 'wl_up_prev' not in df_clean.columns:
                    df_clean['wl_up_prev'] = df_clean['wl_up'].shift(1)
                df_clean['wl_up_prev'] = df_clean['wl_up_prev'].interpolate(method='linear')

                df_before_deletion = df_filtered.copy()

                df_handled = handle_missing_values_by_week(df_clean, start_date, end_date, model_type='random_forest')

                processing_placeholder.empty()

                plot_results(df_before_deletion, df_handled, df_deleted)
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผล")

elif model_choice == "Linear Regression":
    if uploaded_up_file is not None and uploaded_target_file is not None and uploaded_fill_file is not None:
        # โหลดข้อมูลสำหรับเติมข้อมูล
        fill_data = load_data(uploaded_fill_file)
        if fill_data is not None:
            fill_data = clean_data(fill_data)
            fill_data = generate_missing_dates(fill_data)
            fill_data = fill_data.set_index('datetime')
        else:
            fill_data = None

        # โหลดข้อมูลสถานี upstream และ target
        up_data = load_data(uploaded_up_file)
        target_data = load_data(uploaded_target_file)

        if up_data is not None and target_data is not None:
            up_data = clean_data(up_data)
            up_data = generate_missing_dates(up_data)
            up_data = up_data.set_index('datetime')

            target_data = clean_data(target_data)
            target_data = generate_missing_dates(target_data)
            target_data = target_data.set_index('datetime')

            st.subheader('กราฟข้อมูลระดับน้ำ')
            st.plotly_chart(plot_data_combined(target_data, label='สถานีที่ต้องการทำนาย'))
            st.plotly_chart(plot_data_combined(up_data, label='สถานีข้างบน (up)'))

            if use_second_file_lr:
                if uploaded_file2_lr is not None:
                    df2_lr = load_data(uploaded_file2_lr)
                    if df2_lr is not None:
                        df2_lr = clean_data(df2_lr)
                        df2_lr = generate_missing_dates(df2_lr)
                        df2_lr = df2_lr.set_index('datetime')
                    else:
                        df2_lr = None
                else:
                    st.warning("กรุณาอัปโหลดไฟล์ที่สอง (สถานีที่ก่อนหน้า) สำหรับ Linear Regression")
                    df2_lr = None
            else:
                df2_lr = None

            if process_button2:
                with st.spinner("กำลังพยากรณ์..."):
                    start_datetime = pd.Timestamp.combine(forecast_start_date, forecast_start_time)
                    end_datetime = pd.Timestamp.combine(forecast_end_date, forecast_end_time)

                    if start_datetime > end_datetime:
                        st.error("วันและเวลาที่เริ่มต้นต้องไม่เกินวันและเวลาสิ้นสุด")
                    else:
                        selected_data = target_data[(target_data.index >= start_datetime) & (target_data.index <= end_datetime)].copy()

                        if selected_data.empty:
                            st.error("ไม่มีข้อมูลในช่วงวันที่ที่เลือก กรุณาเลือกวันที่ใหม่")
                        else:
                            forecast_start_date_actual = selected_data.index.max() + pd.Timedelta(minutes=15)

                            # เติมค่าที่หายไปจาก fill_data ก่อนพยากรณ์
                            if fill_data is not None:
                                target_data = target_data.combine_first(fill_data['wl_up'])
                                target_data = target_data.fillna(method='ffill').fillna(method='bfill')

                            # รวมข้อมูลจากสถานีใกล้เคียงถ้ามี
                            if use_second_file_lr and df2_lr is not None:
                                # อาจต้องรวมข้อมูลจาก df2_lr เข้ากับ target_data หากต้องการ
                                target_data = merge_data(target_data.reset_index(), df2_lr.reset_index())
                                target_data = target_data.set_index('datetime')
                            
                            # พยากรณ์ด้วย Linear Regression
                            forecasted_data = forecast_with_linear_regression(
                                target_data, up_data, forecast_start_date_actual, delay_hours
                            )

                            if not forecasted_data.empty:
                                st.subheader('กราฟข้อมูลพร้อมการพยากรณ์')
                                st.plotly_chart(plot_data_combined(selected_data, forecasted_data, label='สถานีที่ต้องการทำนาย'))

                                common_indices = forecasted_data.index.intersection(target_data.index)
                                if not common_indices.empty:
                                    actual_data = target_data.loc[common_indices]
                                    y_true = actual_data['wl_up']
                                    y_pred = forecasted_data['wl_up'].loc[common_indices]

                                    min_length = min(len(y_true), len(y_pred))
                                    y_true = y_true[:min_length]
                                    y_pred = y_pred[:min_length]

                                    mae = mean_absolute_error(y_true, y_pred)
                                    rmse = mean_squared_error(y_true, y_pred, squared=False)

                                    st.subheader('ตารางข้อมูลเปรียบเทียบ')
                                    comparison_table = pd.DataFrame({
                                        'datetime': forecasted_data.index[:min_length],
                                        'ค่าจริง (ถ้ามี)': y_true.values,
                                        'ค่าที่พยากรณ์': y_pred.values
                                    })
                                    st.dataframe(comparison_table)

                                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                                else:
                                    st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                            else:
                                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV สำหรับทั้งสามไฟล์ (สถานีที่ต้องการทำนาย, สถานีข้างบน, และไฟล์เติมข้อมูล) เพื่อเริ่มต้นการพยากรณ์")
else:
    st.info("กรุณาเลือกโมเดลที่ต้องการใช้")





