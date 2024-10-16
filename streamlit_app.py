import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
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

# ฟังก์ชันสำหรับการทำความสะอาดข้อมูล (RF และ LR ใช้ร่วมกัน)
def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean.set_index('datetime', inplace=True)
    data_clean = data_clean.resample('15T').mean()
    data_clean = data_clean.interpolate(method='linear')
    data_clean['wl_up'] = pd.to_numeric(data_clean['wl_up'], errors='coerce')
    data_clean = data_clean.dropna(subset=['wl_up'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]

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

# ฟังก์ชันสำหรับการสร้างฟีเจอร์ล่าช้า (Lag Features)
def create_lag_features(data, lags=[1, 4, 96, 192]):
    for lag in lags:
        data[f'lag_{lag}'] = data['wl_up'].shift(lag)
    return data

# ฟังก์ชันสำหรับการฝึกและพยากรณ์ด้วย Random Forest (คงไว้เหมือนเดิม)
def train_and_forecast_RF(data_clean):
    # สร้างฟีเจอร์เวลา
    data_clean = create_time_features(data_clean)

    # สร้างฟีเจอร์ล่าช้า
    data_clean = create_lag_features(data_clean)

    # ลบแถวที่มีค่า NaN
    data_clean.dropna(inplace=True)

    # แบ่งฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = ['year', 'month', 'day', 'hour', 'minute', 'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month'] + [f'lag_{lag}' for lag in [1,4,96,192]]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # สร้างโมเดล Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # พยากรณ์
    y_pred = rf.predict(X_test)

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.header("ผลค่าความแม่นยำของ Random Forest", divider='gray')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

    # สร้าง DataFrame สำหรับการแสดงผล
    results_df = X_test.copy()
    results_df['Actual'] = y_test
    results_df['Predicted'] = y_pred
    results_df.reset_index(inplace=True)

    # แสดงกราฟ
    fig = px.line(results_df, x='datetime', y=['Actual', 'Predicted'], labels={'value': 'ระดับน้ำ', 'datetime': 'วันที่และเวลา', 'variable': 'ประเภทข้อมูล'})
    st.plotly_chart(fig, use_container_width=True)

    st.header("ตารางข้อมูลพยากรณ์ด้วย Random Forest", divider='gray')
    st.dataframe(results_df[['datetime', 'Actual', 'Predicted']], use_container_width=True)

# ฟังก์ชันสำหรับการฝึกและพยากรณ์ด้วย Linear Regression
def train_and_forecast_LR(target_data, upstream_data=None, downstream_data=None, use_upstream=False, use_downstream=False, forecast_days=2, travel_time_up=0, travel_time_down=0):
    # ทำความสะอาดข้อมูล
    target_data = clean_data(target_data)
    target_data = create_time_features(target_data)
    target_data.set_index('datetime', inplace=True)

    if use_upstream and upstream_data is not None:
        upstream_data = clean_data(upstream_data)
        upstream_data = create_time_features(upstream_data)
        upstream_data['datetime'] = upstream_data['datetime'] + timedelta(hours=travel_time_up)
        upstream_data.set_index('datetime', inplace=True)
        target_data = target_data.join(upstream_data[['wl_up']], rsuffix='_upstream')

    if use_downstream and downstream_data is not None:
        downstream_data = clean_data(downstream_data)
        downstream_data = create_time_features(downstream_data)
        downstream_data['datetime'] = downstream_data['datetime'] - timedelta(hours=travel_time_down)
        downstream_data.set_index('datetime', inplace=True)
        target_data = target_data.join(downstream_data[['wl_up']], rsuffix='_downstream')

    # เติมค่า missing values
    if use_upstream and 'wl_up_upstream' in target_data.columns:
        target_data['wl_up_upstream'] = target_data['wl_up_upstream'].interpolate(method='linear')

    if use_downstream and 'wl_up_downstream' in target_data.columns:
        target_data['wl_up_downstream'] = target_data['wl_up_downstream'].interpolate(method='linear')

    # สร้างฟีเจอร์ล่าช้า
    lags = [1, 4, 96, 192]
    for lag in lags:
        target_data[f'lag_{lag}'] = target_data['wl_up'].shift(lag)
        if use_upstream:
            target_data[f'lag_{lag}_upstream'] = target_data['wl_up_upstream'].shift(lag)
        if use_downstream:
            target_data[f'lag_{lag}_downstream'] = target_data['wl_up_downstream'].shift(lag)

    target_data.dropna(inplace=True)

    # กำหนดฟีเจอร์และตัวแปรเป้าหมาย
    feature_cols = [f'lag_{lag}' for lag in lags]
    if use_upstream:
        feature_cols += [f'lag_{lag}_upstream' for lag in lags]
    if use_downstream:
        feature_cols += [f'lag_{lag}_downstream' for lag in lags]

    X = target_data[feature_cols]
    y = target_data['wl_up']

    # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ฝึกโมเดล
    model = LinearRegression()
    model.fit(X_train, y_train)

    # การพยากรณ์อนาคต
    forecast_periods = forecast_days * 96  # 96 ช่วงเวลา 15 นาทีต่อวัน
    forecast_index = pd.date_range(start=target_data.index.max() + timedelta(minutes=15), periods=forecast_periods, freq='15T')
    forecast_df = pd.DataFrame(index=forecast_index)

    # เตรียมข้อมูลสำหรับการพยากรณ์
    combined_data = target_data.copy()
    for date in forecast_index:
        lag_features = {}
        for lag in lags:
            lag_time = date - timedelta(minutes=15 * lag)
            if lag_time in combined_data.index:
                lag_features[f'lag_{lag}'] = combined_data.at[lag_time, 'wl_up']
                if use_upstream and f'lag_{lag}_upstream' in combined_data.columns:
                    lag_features[f'lag_{lag}_upstream'] = combined_data.at[lag_time, f'lag_{lag}_upstream']
                if use_downstream and f'lag_{lag}_downstream' in combined_data.columns:
                    lag_features[f'lag_{lag}_downstream'] = combined_data.at[lag_time, f'lag_{lag}_downstream']
            else:
                lag_features[f'lag_{lag}'] = np.nan
                if use_upstream:
                    lag_features[f'lag_{lag}_upstream'] = np.nan
                if use_downstream:
                    lag_features[f'lag_{lag}_downstream'] = np.nan

        input_df = pd.DataFrame([lag_features], index=[date])
        if input_df.isnull().values.any():
            break
        pred = model.predict(input_df)[0]
        forecast_df.at[date, 'wl_up'] = pred
        # อัปเดตข้อมูลสำหรับการพยากรณ์ครั้งถัดไป
        new_row = input_df.copy()
        new_row['wl_up'] = pred
        combined_data = pd.concat([combined_data, new_row[['wl_up']]], axis=0)

    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'datetime'}, inplace=True)

    # คำนวณค่าความแม่นยำ
    mse = mean_squared_error(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    st.header("ผลค่าความแม่นยำของ Linear Regression", divider='gray')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

    return forecast_df

# ฟังก์ชันสำหรับการแสดงผลกราฟของ Linear Regression
def plot_results_LR(data_before, forecast_df, use_upstream=False, upstream_data=None, use_downstream=False, downstream_data=None):
    data_before = data_before.copy()
    data_before['label'] = 'ข้อมูลจริง'
    forecast_df = forecast_df.copy()
    forecast_df['label'] = 'ค่าพยากรณ์'

    combined_data = pd.concat([data_before[['datetime', 'wl_up', 'label']], forecast_df[['datetime', 'wl_up', 'label']]])

    fig = px.line(combined_data, x='datetime', y='wl_up', color='label', labels={'wl_up': 'ระดับน้ำ', 'datetime': 'วันที่และเวลา', 'label': 'ประเภทข้อมูล'})

    if use_upstream and upstream_data is not None:
        upstream_data = upstream_data.copy()
        upstream_data['label'] = 'สถานี Upstream'
        fig.add_scatter(x=upstream_data['datetime'], y=upstream_data['wl_up'], mode='lines', name='สถานี Upstream')

    if use_downstream and downstream_data is not None:
        downstream_data = downstream_data.copy()
        downstream_data['label'] = 'สถานี Downstream'
        fig.add_scatter(x=downstream_data['datetime'], y=downstream_data['wl_up'], mode='lines', name='สถานี Downstream')

    st.plotly_chart(fig, use_container_width=True)

    st.header("ตารางข้อมูลพยากรณ์ด้วย Linear Regression", divider='gray')
    st.dataframe(forecast_df, use_container_width=True)

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
            use_upstream_lr = st.checkbox("ต้องการใช้สถานี Upstream", value=False)
            use_downstream_lr = st.checkbox("ต้องการใช้สถานี Downstream", value=False)
            forecast_days_lr = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)

            # อัปโหลดไฟล์
            if use_upstream_lr:
                uploaded_up_lr = st.file_uploader("ข้อมูลระดับน้ำ Upstream", type="csv", key="uploader_up_lr")
                time_lag_up_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0)
            else:
                uploaded_up_lr = None
                time_lag_up_lr = 0

            if use_downstream_lr:
                uploaded_down_lr = st.file_uploader("ข้อมูลระดับน้ำ Downstream", type="csv", key="uploader_down_lr")
                time_lag_down_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0)
            else:
                uploaded_down_lr = None
                time_lag_down_lr = 0

            # อัปโหลดไฟล์หลัก
            uploaded_fill_lr = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการพยากรณ์", type="csv", key="uploader_fill_lr")

        process_button_lr = st.button("ประมวลผล Linear Regression", type="primary")

# Main content: Display results after file uploads and date selection
if model_choice == "Random Forest":
    if process_button:
        if uploaded_file is not None:
            data_rf = load_data(uploaded_file)
            if data_rf is not None:
                data_rf = clean_data(data_rf)
                if use_upstream and uploaded_up_file is not None:
                    upstream_data_rf = load_data(uploaded_up_file)
                    if upstream_data_rf is not None:
                        upstream_data_rf = clean_data(upstream_data_rf)
                else:
                    upstream_data_rf = None

                if use_downstream and uploaded_down_file is not None:
                    downstream_data_rf = load_data(uploaded_down_file)
                    if downstream_data_rf is not None:
                        downstream_data_rf = clean_data(downstream_data_rf)
                else:
                    downstream_data_rf = None

                # การประมวลผลเพิ่มเติมสำหรับ Random Forest ตามที่คุณมีในโค้ดเดิม
                train_and_forecast_RF(data_rf)
            else:
                st.error("ไม่สามารถโหลดข้อมูลที่ต้องการพยากรณ์ได้")
        else:
            st.error("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการพยากรณ์")

elif model_choice == "Linear Regression":
    if process_button_lr:
        if uploaded_fill_lr is not None:
            target_df_lr = load_data(uploaded_fill_lr)
            if target_df_lr is not None:
                if use_upstream_lr and uploaded_up_lr is not None:
                    upstream_df_lr = load_data(uploaded_up_lr)
                else:
                    upstream_df_lr = None

                if use_downstream_lr and uploaded_down_lr is not None:
                    downstream_df_lr = load_data(uploaded_down_lr)
                else:
                    downstream_df_lr = None

                forecast_df = train_and_forecast_LR(
                    target_data=target_df_lr,
                    upstream_data=upstream_df_lr,
                    downstream_data=downstream_df_lr,
                    use_upstream=use_upstream_lr,
                    use_downstream=use_downstream_lr,
                    forecast_days=forecast_days_lr,
                    travel_time_up=time_lag_up_lr,
                    travel_time_down=time_lag_down_lr
                )

                if forecast_df is not None:
                    plot_results_LR(target_df_lr, forecast_df, use_upstream_lr, upstream_df_lr, use_downstream_lr, downstream_df_lr)
                else:
                    st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหรือมีข้อผิดพลาดในการประมวลผล")
            else:
                st.error("ไม่สามารถโหลดข้อมูลที่ต้องการพยากรณ์ได้")
        else:
            st.error("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการพยากรณ์")



























