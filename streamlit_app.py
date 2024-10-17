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
    data_clean.sort_values('datetime', inplace=True)
    data_clean.reset_index(drop=True, inplace=True)
    data_clean['diff'] = data_clean['wl_up'].diff().abs()
    threshold = data_clean['diff'].median() * 5
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

# ฟังก์ชันสำหรับสร้างช่วงวันที่ที่หายไป
def generate_missing_dates(data):
    if data['datetime'].isnull().all():
        st.error("ไม่มีข้อมูลวันที่ในข้อมูลที่ให้มา")
        st.stop()
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

# ฟังก์ชันสำหรับกราฟข้อมูลหลังอัปโหลดไฟล์
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
    y_columns = ['สถานีที่ต้องการทำนาย']
    if df2_pre is not None:
        y_columns.append('สถานีน้ำ Upstream')
    if df3_pre is not None:
        y_columns.append('สถานีน้ำ Downstream')
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
    st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับรวมข้อมูล
def merge_data_linear(df1, df2=None, suffix='_prev'):
    if df2 is not None:
        merged_df = pd.merge(df1, df2[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', suffix))
    else:
        df1[f'wl_up{suffix}'] = df1['wl_up'].shift(1)
        merged_df = df1.copy()
    return merged_df

# ฟังก์ชันสำหรับกราฟข้อมูลและการพยากรณ์
def plot_data_combined_LR_stations(data, forecasted=None, upstream_data=None, downstream_data=None, label='ระดับน้ำ'):
    combined_data = data.copy()
    if forecasted is not None and not forecasted.empty:
        combined_data['Predicted'] = np.nan
        combined_data.loc[forecasted.index, 'Predicted'] = forecasted['wl_up']
    else:
        combined_data['Predicted'] = np.nan
    fig = px.line(
        combined_data,
        x=combined_data.index,
        y=['wl_up', 'Predicted'],
        labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
        title='ระดับน้ำที่สถานี {}'.format(label),
        color_discrete_map={'wl_up': 'blue', 'Predicted': 'red'}
    )
    if upstream_data is not None:
        fig.add_scatter(x=upstream_data.index, y=upstream_data['wl_up'], mode='lines', name='สถานี Upstream', line=dict(color='green'))
    if downstream_data is not None:
        fig.add_scatter(x=downstream_data.index, y=downstream_data['wl_up'], mode='lines', name='สถานี Downstream', line=dict(color='purple'))
    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)", legend_title="ประเภทข้อมูล")
    st.plotly_chart(fig, use_container_width=True)

# ฟังก์ชันสำหรับการพยากรณ์ด้วย Linear Regression
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

# ฟังก์ชันสำหรับคำนวณค่าความแม่นยำ
def calculate_accuracy_metrics_linear(original, filled):
    merged_data = pd.merge(original[['datetime', 'wl_up']], filled[['datetime', 'wl_up2']], on='datetime')
    merged_data = merged_data.dropna(subset=['wl_up', 'wl_up2'])
    if merged_data.empty:
        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None, merged_data
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
        # ส่วนของ Random Forest (ไม่แก้ไขตามคำขอ)
        pass

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_nearby_lr = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            if use_nearby_lr:
                use_upstream_lr = st.checkbox("ใช้สถานี Upstream", value=True)
                use_downstream_lr = st.checkbox("ใช้สถานี Downstream", value=False)
            else:
                use_upstream_lr = False
                use_downstream_lr = False
            if use_nearby_lr and use_upstream_lr:
                uploaded_up_lr = st.file_uploader("ข้อมูลสถานี Upstream", type="csv", key="uploader_up_lr")
            else:
                uploaded_up_lr = None
            if use_nearby_lr and use_downstream_lr:
                uploaded_down_lr = st.file_uploader("ข้อมูลสถานี Downstream", type="csv", key="uploader_down_lr")
            else:
                uploaded_down_lr = None
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
            uploaded_fill_lr = st.file_uploader("ข้อมูลสถานีที่ต้องการพยากรณ์", type="csv", key="uploader_fill_lr")
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            training_start_date_lr = st.date_input("วันที่เริ่มต้นฝึกโมเดล", value=pd.to_datetime("2024-05-01"), key='training_start_lr')
            training_start_time_lr = st.time_input("เวลาเริ่มต้นฝึกโมเดล", value=pd.Timestamp("00:00:00").time(), key='training_start_time_lr')
            training_end_date_lr = st.date_input("วันที่สิ้นสุดฝึกโมเดล", value=pd.to_datetime("2024-05-31"), key='training_end_lr')
            training_end_time_lr = st.time_input("เวลาสิ้นสุดฝึกโมเดล", value=pd.Timestamp("23:45:00").time(), key='training_end_time_lr')
        with st.sidebar.expander("ตั้งค่าการพยากรณ์", expanded=False):
            forecast_days_lr = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)
        process_button_lr = st.button("ประมวลผล Linear Regression", type="primary")

# ส่วนหลักของแอป
if model_choice == "Linear Regression":
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
                                st.header("กราฟข้อมูลพร้อมการพยากรณ์ (Linear Regression)")
                                plot_data_combined_LR_stations(
                                    target_df_lr, 
                                    forecasted_data_lr, 
                                    upstream_df_lr.set_index('datetime') if upstream_df_lr is not None else None, 
                                    downstream_df_lr.set_index('datetime') if downstream_df_lr is not None else None, 
                                    label='สถานีที่ต้องการทำนาย'
                                )
                                st.markdown("---")
                                filled_lr = forecasted_data_lr.reset_index().rename(columns={'index': 'datetime'})
                                filled_lr['wl_up2'] = filled_lr['wl_up']
                                filled_lr.drop(columns=['wl_up'], inplace=True)
                                mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics_linear(
                                    original=target_df_lr.reset_index(),
                                    filled=filled_lr
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
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Linear Regression")










































