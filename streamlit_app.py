import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
from datetime import datetime, timedelta

# -------------------------------
# ฟังก์ชันสำหรับการทำความสะอาดข้อมูล
# -------------------------------
def clean_data(data):
    """
    ฟังก์ชันสำหรับทำความสะอาดข้อมูล โดยการลบข้อมูลที่ระดับน้ำต่ำกว่า 100
    และแปลงคอลัมน์ datetime ให้เป็นดัชนีของข้อมูล
    """
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    data = data.dropna(subset=['datetime'])
    data['datetime'] = data['datetime'].dt.tz_localize(None)
    data.set_index('datetime', inplace=True)

    # ลบข้อมูลที่มีค่า wl_up น้อยกว่า 100
    if 'wl_up' not in data.columns:
        st.error("คอลัมน์ 'wl_up' ไม่พบในข้อมูล กรุณาตรวจสอบไฟล์ CSV")
        return pd.DataFrame()
    
    data = data[data['wl_up'] >= 100]
    return data

# -------------------------------
# ฟังก์ชันสำหรับการแสดงกราฟข้อมูล
# -------------------------------
def plot_data_combined(original_data, forecasted=None, actual_forecasted=None, label='ข้อมูล'):
    """
    ฟังก์ชันสำหรับการแสดงกราฟข้อมูลจริงและค่าที่พยากรณ์
    """
    fig = px.line(original_data, x=original_data.index, y='wl_up', title='ระดับน้ำตามเวลา',
                  labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'}, name=label, color_discrete_sequence=['blue'])
    
    if forecasted is not None and not forecasted.empty:
        fig.add_scatter(x=forecasted.index, y=forecasted['wl_up'], mode='lines', name='ค่าที่พยากรณ์', line=dict(color='red'))
    
    if actual_forecasted is not None and not actual_forecasted.empty:
        fig.add_scatter(x=actual_forecasted.index, y=actual_forecasted['wl_up'], mode='lines', name='ค่าจริง (ช่วงพยากรณ์)', line=dict(color='green'))
    
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression
# --------------------------------------------
def forecast_with_linear_regression(data, forecast_start_date):
    """
    ฟังก์ชันสำหรับการพยากรณ์ระดับน้ำโดยใช้โมเดล Linear Regression
    """
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
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    # การพยากรณ์
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            # ดึงค่าจากข้อมูลจริงหรือค่าที่พยากรณ์ก่อนหน้า
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
            elif lag_time in forecasted_data.index:
                lag_value = forecasted_data.at[lag_time, 'wl_up']
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

# --------------------------------------------
# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression แบบสองสถานี
# --------------------------------------------
def forecast_with_linear_regression_two(data, upstream_data, forecast_start_date, delay_hours):
    """
    ฟังก์ชันสำหรับการพยากรณ์ระดับน้ำโดยใช้โมเดล Linear Regression แบบสองสถานี
    """
    # เลื่อนข้อมูล upstream ตาม delay_hours
    upstream_data = upstream_data.shift(freq=pd.Timedelta(hours=delay_hours))

    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = training_data_end - pd.Timedelta(days=3) + pd.Timedelta(minutes=15)

    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    training_data = data.loc[training_data_start:training_data_end].copy()
    training_data = training_data.join(upstream_data, rsuffix='_upstream')

    lags = [1, 4, 96, 192]
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)
        training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)

    # ลบแถวที่มีค่า NaN ในฟีเจอร์ lag
    training_data.dropna(inplace=True)
    if training_data.empty:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอหลังจากสร้างฟีเจอร์ lag")
        return pd.DataFrame()

    feature_cols = [f'lag_{lag}' for lag in lags] + [f'lag_{lag}_upstream' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    if X_train.empty or len(X_train) < 1:
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากไม่มีข้อมูลเพียงพอในการเทรนโมเดล")
        return pd.DataFrame()

    model = LinearRegression()
    model.fit(X_train, y_train)

    forecast_periods = 96
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index)
    forecasted_data['wl_up'] = np.nan

    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in data.index:
                lag_value = data.at[lag_time, 'wl_up']
                lag_value_upstream = upstream_data.at[lag_time, 'wl_up']
            elif lag_time in forecasted_data.index:
                lag_value = forecasted_data.at[lag_time, 'wl_up']
                lag_value_upstream = forecasted_data.at[lag_time, 'wl_up']
            else:
                lag_value = np.nan
                lag_value_upstream = np.nan

            lag_features[f'lag_{lag}'] = lag_value
            lag_features[f'lag_{lag}_upstream'] = lag_value_upstream

        X_pred = pd.DataFrame([lag_features], columns=feature_cols)
        if not X_pred.isnull().values.any():
            forecast_value = model.predict(X_pred)[0]
            forecasted_data.at[idx, 'wl_up'] = forecast_value

    forecasted_data.dropna(inplace=True)
    return forecasted_data

# --------------------------------------------
# ฟังก์ชันสำหรับการคำนวณค่า MAE และ RMSE
# --------------------------------------------
def calculate_error_metrics(data, forecasted_data):
    """
    ฟังก์ชันสำหรับการคำนวณค่า MAE และ RMSE จากข้อมูลจริงและค่าที่พยากรณ์
    """
    common_indices = forecasted_data.index.intersection(data.index)
    if not common_indices.empty:
        actual_data = data.loc[common_indices]
        y_true = actual_data['wl_up']
        y_pred = forecasted_data['wl_up'].loc[common_indices]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return mae, rmse, actual_data
    else:
        st.warning("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None

# --------------------------------------------
# ฟังก์ชันสำหรับการสร้างตารางเปรียบเทียบ
# --------------------------------------------
def create_comparison_table(forecasted_data, actual_data):
    """
    ฟังก์ชันสำหรับการสร้างตารางเปรียบเทียบค่าจริงและค่าที่พยากรณ์
    """
    comparison_df = pd.DataFrame({
        'Datetime': actual_data.index,
        'ค่าจริง': actual_data['wl_up'],
        'ค่าที่พยากรณ์': forecasted_data['wl_up'].loc[actual_data.index]
    })
    return comparison_df

# --------------------------------------------
# ฟังก์ชันสำหรับการเติมข้อมูลวันที่ที่ขาดหายไป
# --------------------------------------------
def generate_missing_dates(data):
    """
    ฟังก์ชันสำหรับสร้างข้อมูลวันที่ที่ขาดหายไปในชุดข้อมูล
    """
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='15T')
    data = data.reindex(full_index)
    data['wl_up'] = data['wl_up'].interpolate(method='linear')
    return data

# --------------------------------------------
# ฟังก์ชันสำหรับการสร้างฟีเจอร์เวลาเพิ่มเติม
# --------------------------------------------
def create_time_features(df):
    """
    ฟังก์ชันสำหรับสร้างฟีเจอร์เวลาเพิ่มเติม เช่น ชั่วโมงของวัน วันของสัปดาห์ เป็นต้น
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    return df

# --------------------------------------------
# ส่วนหลักของโปรแกรม Streamlit
# --------------------------------------------
def main():
    st.title("แอปพลิเคชันพยากรณ์ระดับน้ำด้วย Linear Regression")
    st.write("""
        แอปพลิเคชันนี้ใช้สำหรับพยากรณ์ระดับน้ำโดยใช้โมเดล Linear Regression 
        คุณสามารถอัปโหลดไฟล์ข้อมูล CSV ของสถานีที่ต้องการทำนาย และเลือกที่จะใช้ข้อมูลจากสถานีข้างเคียงได้
    """)

    st.sidebar.header("การตั้งค่า")

    model_choice = "Linear Regression"  # เนื่องจากเรามีเฉพาะ Linear Regression
    st.sidebar.subheader("เลือกโมเดล")
    st.sidebar.write(model_choice)

    uploaded_fill_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV สำหรับเติมข้อมูล (สถานีหลัก)", type=["csv"])
    use_upstream = st.sidebar.checkbox("ใช้ข้อมูลจากสถานีข้างเคียง")

    if use_upstream:
        uploaded_up_file = st.sidebar.file_uploader("อัปโหลดไฟล์ CSV สำหรับสถานีข้างเคียง", type=["csv"])
        delay_hours = st.sidebar.number_input("ระบุชั่วโมงที่ล่าช้า (delay hours)", min_value=0, max_value=24, value=2)
    else:
        uploaded_up_file = None
        delay_hours = 0

    forecast_start_date = st.sidebar.date_input("เลือกวันที่เริ่มพยากรณ์", datetime.today() - timedelta(days=1))
    forecast_end_date = st.sidebar.date_input("เลือกวันที่สิ้นสุดการพยากรณ์", datetime.today())

    process_button2 = st.sidebar.button("เริ่มการพยากรณ์")

    if model_choice == "Linear Regression":
        if uploaded_fill_file:
            # โหลดข้อมูลของสถานีที่ต้องการทำนาย
            try:
                target_df = pd.read_csv(uploaded_fill_file)
                st.sidebar.success("ไฟล์สถานีหลักถูกอัปโหลดเรียบร้อยแล้ว")
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
                    target_df = create_time_features(target_df)
                    if 'wl_up' not in target_df.columns:
                        st.error("คอลัมน์ 'wl_up' หายไปหลังจากการทำความสะอาดข้อมูล")
                        return

                    target_df['wl_up_prev'] = target_df['wl_up'].shift(1)
                    target_df['wl_up_prev'] = target_df['wl_up_prev'].interpolate(method='linear')

                    # ตรวจสอบและเติมค่า NaN ใน 'wl_up_prev'
                    target_df['wl_up_prev'].fillna(target_df['wl_up_prev'].mean(), inplace=True)

                    # เพิ่มการตรวจสอบข้อมูล
                    st.subheader("ข้อมูลสถานีหลักหลังการทำความสะอาดและเตรียมข้อมูล")
                    st.write("คอลัมน์ใน DataFrame:", target_df.columns.tolist())
                    st.write(target_df.head())

                    # โหลดข้อมูลสถานีใกล้เคียงถ้าเลือกใช้
                    if use_upstream and uploaded_up_file:
                        try:
                            upstream_df = pd.read_csv(uploaded_up_file)
                            st.sidebar.success("ไฟล์สถานีข้างเคียงถูกอัปโหลดเรียบร้อยแล้ว")
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
                                upstream_df = create_time_features(upstream_df)
                                upstream_df['wl_up_prev'] = upstream_df['wl_up'].shift(1)
                                upstream_df['wl_up_prev'] = upstream_df['wl_up_prev'].interpolate(method='linear')
                                # ตรวจสอบและเติมค่า NaN ใน 'wl_up_prev'
                                upstream_df['wl_up_prev'].fillna(upstream_df['wl_up_prev'].mean(), inplace=True)

                                # เพิ่มการตรวจสอบข้อมูล
                                st.subheader("ข้อมูลสถานีข้างเคียงหลังการทำความสะอาดและเตรียมข้อมูล")
                                st.write("คอลัมน์ใน DataFrame:", upstream_df.columns.tolist())
                                st.write(upstream_df.head())
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
                            # กำหนดวันที่เริ่มพยากรณ์เป็นเวลาถัดไปจากข้อมูลที่เลือก
                            forecast_start_datetime = target_df.index.max() + pd.Timedelta(minutes=15)
                            forecast_start_date_actual = forecast_start_datetime

                            # พยากรณ์
                            if not use_upstream or upstream_df.empty:
                                forecasted_data = forecast_with_linear_regression(target_df, forecast_start_date_actual)
                            else:
                                forecasted_data = forecast_with_linear_regression_two(
                                    data=target_df,
                                    upstream_data=upstream_df,
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
                                    comparison_table = create_comparison_table(forecasted_data, actual_forecasted_data)
                                    st.dataframe(comparison_table)

                                    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                                else:
                                    st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
                            else:
                                st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
        else:
            st.info("กรุณาอัปโหลดไฟล์ CSV สำหรับเติมข้อมูล เพื่อเริ่มต้นการพยากรณ์")

# เรียกใช้ฟังก์ชันหลัก
if __name__ == "__main__":
    main()






