import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define all necessary functions

def load_data(file):
    """
    Load CSV data into a DataFrame.
    """
    if file is None:
        st.error("ไม่มีไฟล์ที่อัปโหลด กรุณาอัปโหลดไฟล์ CSV")
        return None
    
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("ไฟล์ CSV ว่างเปล่า กรุณาอัปโหลดไฟล์ที่มีข้อมูล")
            return None
        st.success("ไฟล์ถูกโหลดเรียบร้อยแล้ว")
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

def clean_data(df):
    """
    Clean the data by handling datetime and spike values.
    """
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= -100)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    
    # Handle spikes
    data_clean.sort_values('datetime', inplace=True)  # Sort by datetime
    data_clean.reset_index(drop=True, inplace=True)
    data_clean['diff'] = data_clean['wl_up'].diff().abs()
    threshold = data_clean['diff'].median() * 5  # Threshold for spike detection
    
    # Identify spikes
    data_clean['is_spike'] = data_clean['diff'] > threshold
    data_clean.loc[data_clean['is_spike'], 'wl_up'] = np.nan
    data_clean.drop(columns=['diff', 'is_spike'], inplace=True)
    data_clean['wl_up'] = data_clean['wl_up'].interpolate(method='linear')
    
    return data_clean

def generate_missing_dates(data):
    """
    Generate a complete datetime range with 15-minute frequency and merge with existing data.
    """
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

def fill_code_column(data):
    """
    Fill missing values in 'code' column using forward fill and backward fill.
    """
    if 'code' in data.columns:
        data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

def create_time_features(data_clean):
    """
    Create time-based features from 'datetime' column.
    """
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

def create_lag_features(data, lags):
    """
    Create lag features based on the specified lags.
    """
    for lag in lags:
        data[f'lag_{lag}'] = data['wl_up'].shift(lag)
    return data

def create_moving_average_features(data, window=672):
    """
    Create moving average features.
    window: number of periods (672 periods = 7 days at 15-minute intervals)
    """
    data[f'ma_{window}'] = data['wl_up'].rolling(window=window, min_periods=1).mean()
    return data

def prepare_features(data_clean, lags, window=672):
    """
    Prepare features and target variable for modeling.
    """
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month'
    ]
    
    # Create lag features
    data_clean = create_lag_features(data_clean, lags)
    
    # Create moving average features
    data_clean = create_moving_average_features(data_clean, window)
    
    # Add lag features to feature_cols
    lag_cols = [f'lag_{lag}' for lag in lags]
    ma_col = f'ma_{window}'
    feature_cols.extend(lag_cols)
    feature_cols.append(ma_col)
    
    # Handle missing 'wl_up_prev' feature
    if 'wl_up_prev' not in data_clean.columns:
        data_clean['wl_up_prev'] = data_clean['wl_up'].shift(1)
    data_clean['wl_up_prev'] = data_clean['wl_up_prev'].interpolate(method='linear')
    feature_cols.append('wl_up_prev')
    
    # Drop rows with NaN in feature columns
    data_clean = data_clean.dropna(subset=feature_cols)
    
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

def merge_data(main_data, upstream_data=None, downstream_data=None):
    """
    Merge main data with upstream and downstream data on 'datetime'.
    """
    if upstream_data is not None:
        main_data = pd.merge(main_data, upstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))
    if downstream_data is not None:
        main_data = pd.merge(main_data, downstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))
    return main_data

def merge_data_linear(main_data, upstream_data=None, downstream_data=None):
    """
    Similar to merge_data, but for Linear Regression.
    """
    return merge_data(main_data, upstream_data, downstream_data)

def delete_data_by_date_range(data, start_datetime, end_datetime):
    """
    Delete data within the specified datetime range by setting 'wl_up' to NaN.
    """
    mask = (data['datetime'] >= start_datetime) & (data['datetime'] <= end_datetime)
    data.loc[mask, 'wl_up'] = np.nan
    return data

def handle_missing_values_by_week(data, model_type='random_forest'):
    """
    Handle missing values, possibly differently based on model type.
    Placeholder function - implementation may vary.
    """
    # For simplicity, forward fill
    data['wl_up'] = data['wl_up'].fillna(method='ffill').fillna(method='bfill')
    return data

def train_random_forest(X_train, y_train):
    """
    Train Random Forest with hyperparameter tuning using RandomizedSearchCV.
    """
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

def train_linear_regression_model(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def forecast_with_linear_regression_multi(data, forecast_start_date, forecast_days, upstream_data=None, downstream_data=None, delay_hours_up=0, delay_hours_down=0):
    """
    Forecast future values using Linear Regression with multiple stations.
    """
    # Check forecast_days
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    lags = [1, 2, 4, 8]  # 15 min, 30 min, 1 hr, 2 hr

    feature_cols = [f'lag_{lag}' for lag in lags]
    if upstream_data is not None:
        feature_cols += [f'lag_{lag}_upstream' for lag in lags]
    if downstream_data is not None:
        feature_cols += [f'lag_{lag}_downstream' for lag in lags]

    # Adjust upstream and downstream data
    if upstream_data is not None and not upstream_data.empty:
        if delay_hours_up > 0:
            upstream_data = upstream_data.copy()
            upstream_data['datetime'] = upstream_data['datetime'] + pd.Timedelta(hours=delay_hours_up)
    else:
        upstream_data = None

    if downstream_data is not None and not downstream_data.empty:
        if delay_hours_down > 0:
            downstream_data = downstream_data.copy()
            downstream_data['datetime'] = downstream_data['datetime'] - pd.Timedelta(hours=delay_hours_down)
    else:
        downstream_data = None

    # Merge data
    training_data = merge_data_linear(data.copy(), upstream_data, downstream_data)

    # Create lag features
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up_target'].shift(lag)
        if upstream_data is not None:
            training_data[f'lag_{lag}_upstream'] = training_data['wl_up_upstream'].shift(lag)
        if downstream_data is not None:
            training_data[f'lag_{lag}_downstream'] = training_data['wl_up_downstream'].shift(lag)

    # Fill missing values
    training_data.fillna(method='ffill', inplace=True)
    training_data.dropna(inplace=True)

    # Prepare X and y
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up_target']

    # Train Linear Regression model
    pipeline = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    pipeline.fit(X_train, y_train)

    # Forecast periods
    forecast_periods = forecast_days * 96  # 96 periods per day (15 min)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # Prepare combined data for forecasting
    combined_data = data.copy()
    if upstream_data is not None:
        combined_upstream = upstream_data.set_index('datetime')['wl_up']
    else:
        combined_upstream = pd.Series([0]*len(combined_data), index=combined_data.index)
    if downstream_data is not None:
        combined_downstream = downstream_data.set_index('datetime')['wl_up']
    else:
        combined_downstream = pd.Series([0]*len(combined_data), index=combined_data.index)

    # Forecast loop
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up_target']):
                lag_features[f'lag_{lag}'] = combined_data.at[lag_time, 'wl_up_target']
            else:
                lag_features[f'lag_{lag}'] = y_train.mean()
            if upstream_data is not None:
                if lag_time in combined_upstream.index and not pd.isnull(combined_upstream.at[lag_time]):
                    lag_features[f'lag_{lag}_upstream'] = combined_upstream.at[lag_time]
                else:
                    lag_features[f'lag_{lag}_upstream'] = y_train.mean()
            if downstream_data is not None:
                if lag_time in combined_downstream.index and not pd.isnull(combined_downstream.at[lag_time]):
                    lag_features[f'lag_{lag}_downstream'] = combined_downstream.at[lag_time]
                else:
                    lag_features[f'lag_{lag}_downstream'] = y_train.mean()

        # Create DataFrame for prediction
        X_pred = pd.DataFrame([lag_features], columns=feature_cols)

        try:
            forecast_value = pipeline.predict(X_pred)[0]
            # Clip forecast_value to be within min and max of historical data
            forecast_value = np.clip(forecast_value, data['wl_up_target'].min(), data['wl_up_target'].max())
            forecasted_data.at[idx, 'wl_up'] = forecast_value

            # Update combined_data for next iteration
            combined_data = combined_data.append({'datetime': idx, 'wl_up_target': forecast_value}, ignore_index=True)
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในเวลา {idx} ได้: {e}")

    return forecasted_data

def forecast_with_linear_regression_single(data, forecast_start_date, forecast_days, lags=[1,2,4,8]):
    """
    Forecast future values using Linear Regression for single station.
    """
    # Check forecast_days
    if forecast_days < 1 or forecast_days > 30:
        st.error("สามารถพยากรณ์ได้ตั้งแต่ 1 ถึง 30 วัน")
        return pd.DataFrame()

    # Define training period: last 30 days before forecast_start_date
    training_data_end = forecast_start_date - pd.Timedelta(minutes=15)
    training_data_start = forecast_start_date - pd.Timedelta(days=30) + pd.Timedelta(minutes=15)

    if training_data_start < data.index.min():
        st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลสำหรับการเทรนไม่เพียงพอ")
        return pd.DataFrame()

    # Slice training data
    training_data = data.loc[training_data_start:training_data_end].copy()

    # Create lag features
    for lag in lags:
        training_data[f'lag_{lag}'] = training_data['wl_up'].shift(lag)

    # Drop rows with NaN
    training_data = training_data.dropna()

    # Features and target
    feature_cols = [f'lag_{lag}' for lag in lags]
    X_train = training_data[feature_cols]
    y_train = training_data['wl_up']

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Forecast periods
    forecast_periods = forecast_days * 96  # 96 periods per day (15 min)
    forecast_index = pd.date_range(start=forecast_start_date, periods=forecast_periods, freq='15T')
    forecasted_data = pd.DataFrame(index=forecast_index, columns=['wl_up'])

    # Prepare combined data
    combined_data = data.copy()

    # Forecast loop
    for idx in forecasted_data.index:
        lag_features = {}
        for lag in lags:
            lag_time = idx - pd.Timedelta(minutes=15 * lag)
            if lag_time in combined_data.index and not pd.isnull(combined_data.at[lag_time, 'wl_up']):
                lag_features[f'lag_{lag}'] = combined_data.at[lag_time, 'wl_up']
            else:
                lag_features[f'lag_{lag}'] = y_train.mean()

        # Create DataFrame for prediction
        X_pred = pd.DataFrame([lag_features], columns=feature_cols)

        try:
            forecast_value = model.predict(X_pred)[0]
            # Clip forecast_value
            forecast_value = np.clip(forecast_value, data['wl_up'].min(), data['wl_up'].max())
            forecasted_data.at[idx, 'wl_up'] = forecast_value

            # Update combined_data for next iteration
            combined_data = combined_data.append(pd.DataFrame({'wl_up': forecast_value}, index=[idx]))
        except Exception as e:
            st.warning(f"ไม่สามารถพยากรณ์ค่าในเวลา {idx} ได้: {e}")

    return forecasted_data

def create_comparison_table_streamlit(forecasted_data, actual_data):
    """
    Create a comparison table between forecasted and actual data.
    """
    comparison_df = pd.DataFrame({
        'Datetime': actual_data['datetime'],
        'ค่าจริง': actual_data['wl_up_target'],
        'ค่าที่พยากรณ์': forecasted_data['wl_up'].values
    })
    return comparison_df

def calculate_accuracy_metrics_linear(original, filled):
    """
    Calculate accuracy metrics between original and filled data.
    """
    # Merge on datetime
    merged_data = pd.merge(original[['datetime', 'wl_up_target']], filled[['datetime', 'wl_up']], on='datetime')
    merged_data = merged_data.dropna(subset=['wl_up_target', 'wl_up'])

    if merged_data.empty:
        st.info("ไม่มีข้อมูลจริงสำหรับช่วงเวลาที่พยากรณ์ ไม่สามารถคำนวณค่า MAE และ RMSE ได้")
        return None, None, None, merged_data

    mse = mean_squared_error(merged_data['wl_up_target'], merged_data['wl_up'])
    mae = mean_absolute_error(merged_data['wl_up_target'], merged_data['wl_up'])
    r2 = r2_score(merged_data['wl_up_target'], merged_data['wl_up'])

    return mse, mae, r2, merged_data

def plot_results(data_before, data_filled, data_deleted, data_deleted_option=False):
    """
    Plot the results after processing.
    """
    data_before_filled = pd.DataFrame({
        'datetime': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up_target']
    })

    data_after_filled = pd.DataFrame({
        'datetime': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up']
    })

    if data_deleted_option:
        data_after_deleted = pd.DataFrame({
            'datetime': data_deleted['datetime'],
            'ข้อมูลหลังลบ': data_deleted['wl_up']
        })
    else:
        data_after_deleted = None

    # Merge data
    data_filled_with_original = pd.merge(
        data_before_filled,
        data_after_filled,
        on='datetime',
        how='outer'
    )

    combined_data = pd.merge(data_before_filled, data_after_filled, on='datetime', how='outer')
    if data_after_deleted is not None:
        combined_data = pd.merge(combined_data, data_after_deleted, on='datetime', how='outer')

    # Define y_columns
    y_columns = ['ข้อมูลหลังเติมค่า', 'ข้อมูลเดิม']
    if data_after_deleted is not None:
        y_columns.append('ข้อมูลหลังลบ')

    # Plot with Plotly
    fig = px.line(combined_data, x='datetime', y=y_columns,
                  labels={'value': 'ระดับน้ำ (wl_up)', 'variable': 'ประเภทข้อมูล'},
                  color_discrete_sequence=px.colors.qualitative.Plotly)

    fig.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)")

    # Show plot
    st.header("ข้อมูลหลังจากการเติมค่าที่หายไป", divider='gray')
    st.plotly_chart(fig, use_container_width=True)

    # Show filled data table
    st.header("ตารางแสดงข้อมูลหลังเติมค่า", divider='gray')
    if data_after_deleted is not None:
        data_filled_selected = data_filled_with_original[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังลบ']]
    else:
        data_filled_selected = data_filled_with_original[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า']]
    st.dataframe(data_filled_selected, use_container_width=True)

    # Calculate accuracy if data_deleted_option
    if data_deleted_option:
        calculate_accuracy_metrics(data_before, data_filled, data_deleted)
    else:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีการลบข้อมูล")

def calculate_accuracy_metrics(original, filled, data_deleted):
    """
    Calculate and display accuracy metrics for the deleted data.
    """
    # Merge original and filled data
    merged_data = pd.merge(original[['datetime', 'wl_up_target']], filled[['datetime', 'wl_up']], on='datetime')

    # Select only the deleted datetimes
    deleted_datetimes = data_deleted[data_deleted['wl_up'].isna()]['datetime']
    merged_data = merged_data[merged_data['datetime'].isin(deleted_datetimes)]

    # Drop NaNs
    merged_data = merged_data.dropna(subset=['wl_up_target', 'wl_up'])

    if merged_data.empty:
        st.info("ไม่มีข้อมูลในช่วงที่ลบสำหรับการคำนวณค่าความแม่นยำ")
        return

    # Calculate metrics
    mse = mean_squared_error(merged_data['wl_up_target'], merged_data['wl_up'])
    mae = mean_absolute_error(merged_data['wl_up_target'], merged_data['wl_up'])
    r2 = r2_score(merged_data['wl_up_target'], merged_data['wl_up'])

    st.header("ผลค่าความแม่นยำในช่วงที่ลบข้อมูล", divider='gray')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
    with col3:
        st.metric(label="R-squared (R²)", value=f"{r2:.4f}")

def plot_data_preview(df_pre, df_up_pre, df_down_pre, total_time_lag_upstream, total_time_lag_downstream):
    """
    Plot data preview from main, upstream, and downstream stations.
    """
    data_pre1 = pd.DataFrame({
        'datetime': df_pre['datetime'],
        'สถานีที่ต้องการทำนาย': df_pre['wl_up']
    })

    combined_data_pre = data_pre1.copy()

    if df_up_pre is not None:
        data_pre2 = pd.DataFrame({
            'datetime': df_up_pre['datetime'] + total_time_lag_upstream,  # Adjust datetime for upstream
            'สถานีน้ำ Upstream': df_up_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre2, on='datetime', how='outer')

    if df_down_pre is not None:
        data_pre3 = pd.DataFrame({
            'datetime': df_down_pre['datetime'] - total_time_lag_downstream,  # Adjust datetime for downstream
            'สถานีน้ำ Downstream': df_down_pre['wl_up']
        })
        combined_data_pre = pd.merge(combined_data_pre, data_pre3, on='datetime', how='outer')

    # Define y_columns
    y_columns = ['สถานีที่ต้องการทำนาย']
    if df_up_pre is not None:
        y_columns.append('สถานีน้ำ Upstream')
    if df_down_pre is not None:
        y_columns.append('สถานีน้ำ Downstream')

    # Plot with Plotly
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

    # Show plot
    st.plotly_chart(fig, use_container_width=True)

def plot_data_combined_LR_stations(data, forecasted=None, upstream_data=None, downstream_data=None, label='สถานีที่ต้องการทำนาย'):
    """
    Plot combined data and forecast for Linear Regression.
    """
    # Plot actual data
    fig_actual = px.line(data, x=data.index, y='wl_up_target', title=f'ระดับน้ำจริงที่สถานี {label}', labels={'x': 'วันที่', 'wl_up_target': 'ระดับน้ำ (wl_up)'})
    fig_actual.update_traces(connectgaps=False, name='สถานีที่ต้องการทำนาย')
    
    # Plot upstream and downstream if available
    if upstream_data is not None:
        fig_actual.add_scatter(x=upstream_data.index, y=upstream_data['wl_up'], mode='lines', name='สถานี Upstream', line=dict(color='green'))
    
    if downstream_data is not None:
        fig_actual.add_scatter(x=downstream_data.index, y=downstream_data['wl_up'], mode='lines', name='สถานี Downstream', line=dict(color='purple'))

    fig_actual.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)", legend_title="สถานี")

    # Plot forecasted data
    if forecasted is not None and not forecasted.empty:
        forecast_start = forecasted.index.min()
        forecast_end = forecasted.index.max()
        actual_forecast_period = data[(data.index >= forecast_start) & (data.index <= forecast_end)]
        
        fig_forecast = px.line(forecasted, x=forecasted.index, y='wl_up', title='ระดับน้ำที่พยากรณ์', labels={'x': 'วันที่', 'wl_up': 'ระดับน้ำ (wl_up)'})
        fig_forecast.update_traces(connectgaps=False, name='ค่าที่พยากรณ์', line=dict(color='red'))
        
        # Compare with actual data in forecast period
        if not actual_forecast_period.empty:
            fig_forecast.add_scatter(x=actual_forecast_period.index, y=actual_forecast_period['wl_up_target'], mode='lines', name='ค่าจริง', line=dict(color='blue'))
        
        fig_forecast.update_layout(xaxis_title="วันที่", yaxis_title="ระดับน้ำ (wl_up)", legend_title="ประเภทข้อมูล")
    else:
        fig_forecast = None

    # Show plots
    st.plotly_chart(fig_actual, use_container_width=True)
    if fig_forecast is not None:
        st.plotly_chart(fig_forecast, use_container_width=True)

    # Show comparison table
    st.header("ตารางข้อมูลเปรียบเทียบ", divider='gray')
    if forecasted is not None:
        comparison_table_lr = create_comparison_table_streamlit(forecasted.reset_index(), data.reset_index())
        st.dataframe(comparison_table_lr, use_container_width=True)

    # Show accuracy metrics
    if forecasted is not None and not forecasted.empty:
        st.header("ผลค่าความแม่นยำ (Linear Regression)", divider='gray')
        mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics_linear(
            original=data.reset_index(),
            filled=forecasted.reset_index().rename(columns={'index': 'datetime'})
        )

        if mse_lr is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Mean Squared Error (MSE)", value=f"{mse_lr:.4f}")
            with col2:
                st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_lr:.4f}")
            with col3:
                st.metric(label="R-squared (R²)", value=f"{r2_lr:.4f}")
    else:
        st.header("ผลค่าความแม่นยำ", divider='gray')
        st.info("ไม่สามารถคำนวณความแม่นยำได้เนื่องจากไม่มีการพยากรณ์ในอนาคต")

def train_and_predict(target_data, upstream_data=None, downstream_data=None, use_upstream=False, use_downstream=False):
    """
    Train Linear Regression model and evaluate performance.
    """
    # Convert 'datetime' to datetime and localize
    target_data['datetime'] = pd.to_datetime(target_data['datetime'], errors='coerce').dt.tz_localize(None)
    
    # Rename 'wl_up' to 'wl_up_target'
    if 'wl_up' not in target_data.columns:
        st.error("ข้อมูลเป้าหมายไม่มีคอลัมน์ 'wl_up'")
        return None
    target_data = target_data.rename(columns={'wl_up': 'wl_up_target'})
    
    # Merge with upstream and downstream
    if use_upstream and upstream_data is not None:
        upstream_data['datetime'] = pd.to_datetime(upstream_data['datetime'], errors='coerce').dt.tz_localize(None)
        delay_hours_up = st.session_state.get('delay_hours_up_lr', 0)
        upstream_data['datetime'] = upstream_data['datetime'] + pd.Timedelta(hours=delay_hours_up)
        target_data = pd.merge(target_data, upstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_upstream'))
    
    if use_downstream and downstream_data is not None:
        downstream_data['datetime'] = pd.to_datetime(downstream_data['datetime'], errors='coerce').dt.tz_localize(None)
        delay_hours_down = st.session_state.get('delay_hours_down_lr', 0)
        downstream_data['datetime'] = downstream_data['datetime'] - pd.Timedelta(hours=delay_hours_down)
        target_data = pd.merge(target_data, downstream_data[['datetime', 'wl_up']], on='datetime', how='left', suffixes=('', '_downstream'))
    
    # Fill missing values
    if use_upstream and 'wl_up_upstream' in target_data.columns:
        target_data['wl_up_upstream'].fillna(method='ffill', inplace=True)
    
    if use_downstream and 'wl_up_downstream' in target_data.columns:
        target_data['wl_up_downstream'].fillna(method='ffill', inplace=True)
    
    # Create lag features
    lags = [1, 2, 4, 8]  # 15 min, 30 min, 1 hr, 2 hr
    for lag in lags:
        target_data[f'lag_{lag}'] = target_data['wl_up_target'].shift(lag)
        if use_upstream:
            target_data[f'lag_{lag}_upstream'] = target_data['wl_up_upstream'].shift(lag)
        if use_downstream:
            target_data[f'lag_{lag}_downstream'] = target_data['wl_up_downstream'].shift(lag)
    
    # Drop rows with NaN due to lagging
    target_data = target_data.dropna().copy()
    
    # Prepare features
    feature_cols = [f'lag_{lag}' for lag in lags]
    if use_upstream:
        feature_cols += [f'lag_{lag}_upstream' for lag in lags]
    if use_downstream:
        feature_cols += [f'lag_{lag}_downstream' for lag in lags]
    
    X = target_data[feature_cols]
    y = target_data['wl_up_target']
    
    # Train Linear Regression model
    pipeline = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Return metrics and predictions
    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'model': pipeline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }

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
    model_choice = st.sidebar.radio("", ("Random Forest", "Linear Regression"),
        label_visibility="collapsed"  # Hide label
    )

    st.sidebar.title("ตั้งค่าข้อมูล")
    if model_choice == "Random Forest":
        with st.sidebar.expander("ตั้งค่า Random Forest", expanded=False):
            use_upstream = st.checkbox("ต้องการใช้สถานี Upstream", value=False)
            use_downstream = st.checkbox("ต้องการใช้สถานี Downstream", value=False)
            
            # Upload upstream file
            if use_upstream:
                uploaded_up_file = st.file_uploader("ข้อมูลระดับน้ำ Upstream", type="csv", key="uploader_up")
                time_lag_upstream = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0)
                st.session_state['delay_hours_up'] = time_lag_upstream
                total_time_lag_upstream = pd.Timedelta(hours=time_lag_upstream)
            else:
                uploaded_up_file = None
                total_time_lag_upstream = pd.Timedelta(hours=0)
            
            # Upload downstream file
            if use_downstream:
                uploaded_down_file = st.file_uploader("ข้อมูลระดับน้ำ Downstream", type="csv", key="uploader_down")
                time_lag_downstream = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0)
                st.session_state['delay_hours_down'] = time_lag_downstream
                total_time_lag_downstream = pd.Timedelta(hours=time_lag_downstream)
            else:
                uploaded_down_file = None
                total_time_lag_downstream = pd.Timedelta(hours=0)

            # Upload main file
            uploaded_file = st.file_uploader("ข้อมูลระดับน้ำที่ต้องการทำนาย", type="csv", key="uploader1")

        # Select date range
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2024-05-01"))
            end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2024-05-31"))
            
            # Option to delete data
            delete_data_option = st.checkbox("ต้องการเลือกลบข้อมูล", value=False)

            if delete_data_option:
                st.header("เลือกช่วงที่ต้องการลบข้อมูล")
                delete_start_date = st.date_input("กำหนดเริ่มต้นลบข้อมูล", value=start_date, key='delete_start')
                delete_start_time = st.time_input("เวลาเริ่มต้น", value=pd.Timestamp("00:00:00").time(), key='delete_start_time')
                delete_end_date = st.date_input("กำหนดสิ้นสุดลบข้อมูล", value=end_date, key='delete_end')
                delete_end_time = st.time_input("เวลาสิ้นสุด", value=pd.Timestamp("23:45:00").time(), key='delete_end_time')

        process_button = st.button("ประมวลผล", type="primary")

    elif model_choice == "Linear Regression":
        with st.sidebar.expander("ตั้งค่า Linear Regression", expanded=False):
            use_nearby_lr = st.checkbox("ต้องการใช้สถานีใกล้เคียง", value=False)
            
            # Checkboxes for upstream and downstream
            if use_nearby_lr:
                use_upstream_lr = st.checkbox("ใช้สถานี Upstream", value=True)
                use_downstream_lr = st.checkbox("ใช้สถานี Downstream", value=False)
            else:
                use_upstream_lr = False
                use_downstream_lr = False
            
            # Upload upstream file
            if use_nearby_lr and use_upstream_lr:
                uploaded_up_lr = st.file_uploader("ข้อมูลสถานี Upstream", type="csv", key="uploader_up_lr")
                time_lag_up_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Upstream (ชั่วโมง)", value=0, min_value=0)
                st.session_state['delay_hours_up_lr'] = time_lag_up_lr
            else:
                uploaded_up_lr = None
                st.session_state['delay_hours_up_lr'] = 0

            # Upload downstream file
            if use_nearby_lr and use_downstream_lr:
                uploaded_down_lr = st.file_uploader("ข้อมูลสถานี Downstream", type="csv", key="uploader_down_lr")
                time_lag_down_lr = st.number_input("ระบุเวลาห่างระหว่างสถานี Downstream (ชั่วโมง)", value=0, min_value=0)
                st.session_state['delay_hours_down_lr'] = time_lag_down_lr
            else:
                uploaded_down_lr = None
                st.session_state['delay_hours_down_lr'] = 0

            # Upload main file
            uploaded_fill_lr = st.file_uploader("ข้อมูลสถานีที่ต้องการพยากรณ์", type="csv", key="uploader_fill_lr")
            
        # Select training date range
        with st.sidebar.expander("เลือกช่วงข้อมูลสำหรับฝึกโมเดล", expanded=False):
            training_start_date_lr = st.date_input("วันที่เริ่มต้นฝึกโมเดล", value=pd.to_datetime("2024-05-01"), key='training_start_lr')
            training_start_time_lr = st.time_input("เวลาเริ่มต้นฝึกโมเดล", value=pd.Timestamp("00:00:00").time(), key='training_start_time_lr')
            training_end_date_lr = st.date_input("วันที่สิ้นสุดฝึกโมเดล", value=pd.to_datetime("2024-05-31"), key='training_end_lr')
            training_end_time_lr = st.time_input("เวลาสิ้นสุดฝึกโมเดล", value=pd.Timestamp("23:45:00").time(), key='training_end_time_lr')

        # Select forecast settings
        with st.sidebar.expander("ตั้งค่าการพยากรณ์", expanded=False):
            forecast_days_lr = st.number_input("จำนวนวันที่ต้องการพยากรณ์", value=3, min_value=1, step=1)

        process_button_lr = st.button("ประมวลผล Linear Regression", type="primary")

# Main content: Random Forest processing
if model_choice == "Random Forest":
    if uploaded_file or uploaded_up_file or uploaded_down_file:
        # Check if main file is uploaded
        if uploaded_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำที่ต้องการทำนาย")
        # Check if upstream file is uploaded if used
        if use_upstream and uploaded_up_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Upstream")
        # Check if downstream file is uploaded if used
        if use_downstream and uploaded_down_file is None:
            st.warning("กรุณาอัปโหลดไฟล์ข้อมูลระดับน้ำ Downstream")

        # Proceed if main file and required upstream/downstream files are uploaded
        if uploaded_file and (not use_upstream or uploaded_up_file) and (not use_downstream or uploaded_down_file):
            df = load_data(uploaded_file)

            if df is not None:
                df_clean = clean_data(df)
                df_clean = generate_missing_dates(df_clean)

                # Process upstream data if used
                if use_upstream and uploaded_up_file is not None:
                    df_up = load_data(uploaded_up_file)
                    if df_up is not None:
                        df_up_clean = clean_data(df_up)
                        df_up_clean = generate_missing_dates(df_up_clean)
                    else:
                        df_up_clean = None
                else:
                    df_up_clean = None

                # Process downstream data if used
                if use_downstream and uploaded_down_file is not None:
                    df_down = load_data(uploaded_down_file)
                    if df_down is not None:
                        df_down_clean = clean_data(df_down)
                        df_down_clean = generate_missing_dates(df_down_clean)
                    else:
                        df_down_clean = None
                else:
                    df_down_clean = None

                # Show data preview
                plot_data_preview(df_clean, df_up_clean, df_down_clean, total_time_lag_upstream, total_time_lag_downstream)

                if process_button:
                    processing_placeholder = st.empty()
                    processing_placeholder.text("กำลังประมวลผลข้อมูล...")

                    # Adjust end_date
                    end_date_dt = pd.to_datetime(end_date) + pd.DateOffset(days=1)

                    # Filter data by date range
                    df_filtered = df_clean[(df_clean['datetime'] >= pd.to_datetime(start_date)) & (df_clean['datetime'] <= pd.to_datetime(end_date_dt))]

                    # Process upstream data
                    if use_upstream and df_up_clean is not None:
                        df_up_filtered = df_up_clean[(df_up_clean['datetime'] >= pd.to_datetime(start_date)) & (df_up_clean['datetime'] <= pd.to_datetime(end_date_dt))]
                        df_up_filtered['datetime'] = df_up_filtered['datetime'] + total_time_lag_upstream
                        df_up_filtered = clean_data(df_up_filtered)
                    else:
                        df_up_filtered = None

                    # Process downstream data
                    if use_downstream and df_down_clean is not None:
                        df_down_filtered = df_down_clean[(df_down_clean['datetime'] >= pd.to_datetime(start_date)) & (df_down_clean['datetime'] <= pd.to_datetime(end_date_dt))]
                        df_down_filtered['datetime'] = df_down_filtered['datetime'] - total_time_lag_downstream
                        df_down_filtered = clean_data(df_down_filtered)
                    else:
                        df_down_filtered = None

                    # Clean main data
                    df_main_clean = clean_data(df_filtered)

                    # Copy before deletion
                    df_before_deletion = df_main_clean.copy()

                    # Merge data
                    df_merged = merge_data(df_main_clean, df_up_filtered, df_down_filtered)

                    # Delete data if option selected
                    if delete_data_option:
                        delete_start_datetime = pd.to_datetime(f"{delete_start_date} {delete_start_time}")
                        delete_end_datetime = pd.to_datetime(f"{delete_end_date} {delete_end_time}") + pd.DateOffset(hours=23, minutes=45)
                        df_deleted = delete_data_by_date_range(df_merged, delete_start_datetime, delete_end_datetime)
                    else:
                        df_deleted = df_merged.copy()

                    # Generate missing dates
                    df_clean_final = generate_missing_dates(df_deleted)

                    # Fill 'code' column if exists
                    df_clean_final = fill_code_column(df_clean_final)

                    # Create time features
                    df_clean_final = create_time_features(df_clean_final)

                    # Fill missing 'wl_up_prev'
                    if 'wl_up_prev' not in df_clean_final.columns:
                        df_clean_final['wl_up_prev'] = df_clean_final['wl_up_target'].shift(1)
                    df_clean_final['wl_up_prev'] = df_clean_final['wl_up_prev'].interpolate(method='linear')

                    # Handle missing values
                    df_handled = handle_missing_values_by_week(df_clean_final, model_type='random_forest')

                    # Prepare features and target
                    lags_rf = [1,2,4,8]  # Assuming similar lags for Random Forest
                    X_rf, y_rf = prepare_features(df_handled, lags=lags_rf, window=672)  # window can be adjusted as needed

                    # Train Random Forest model
                    with st.spinner("กำลังฝึกโมเดล Random Forest..."):
                        model_rf = train_random_forest(X_rf, y_rf)
                    
                    # Make predictions
                    y_pred_rf = model_rf.predict(X_rf)
                    df_handled['predicted_rf'] = y_pred_rf

                    # Calculate accuracy metrics
                    mse_rf = mean_squared_error(y_rf, y_pred_rf)
                    mae_rf = mean_absolute_error(y_rf, y_pred_rf)
                    r2_rf = r2_score(y_rf, y_pred_rf)

                    # Show results
                    st.header("ผลการฝึกโมเดล Random Forest", divider='gray')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Mean Squared Error (MSE)", value=f"{mse_rf:.4f}")
                    with col2:
                        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_rf:.4f}")
                    with col3:
                        st.metric(label="R-squared (R²)", value=f"{r2_rf:.4f}")

                    # Plot Random Forest predictions
                    st.subheader("ค่าจริง vs ค่าพยากรณ์ (Random Forest)")
                    rf_df = pd.DataFrame({
                        'Actual': y_rf,
                        'Predicted': y_pred_rf
                    }, index=X_rf.index)
                    fig_rf = px.line(rf_df, title='Random Forest: Actual vs Predicted')
                    st.plotly_chart(fig_rf, use_container_width=True)

                    # Plot filled data
                    plot_results(df_before_deletion, df_handled, df_deleted, data_deleted_option=delete_data_option)

    else:
        st.info("กรุณาอัปโหลดไฟล์ CSV เพื่อเริ่มต้นการประมวลผลด้วย Random Forest")

# Main content: Linear Regression processing
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

                    # Process upstream data if used
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

                    # Process downstream data if used
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

                    # Merge data for training
                    if use_nearby_lr and (use_upstream_lr or use_downstream_lr):
                        merged_training_data_lr = merge_data_linear(target_df_lr, upstream_df_lr, downstream_df_lr)
                    else:
                        merged_training_data_lr = target_df_lr.copy()

                    # Filter training data by date range
                    training_start_datetime_lr = pd.Timestamp.combine(training_start_date_lr, training_start_time_lr)
                    training_end_datetime_lr = pd.Timestamp.combine(training_end_date_lr, training_end_time_lr)
                    training_data_lr = merged_training_data_lr[
                        (merged_training_data_lr['datetime'] >= training_start_datetime_lr) & 
                        (merged_training_data_lr['datetime'] <= training_end_datetime_lr)
                    ].copy()

                    if training_data_lr.empty:
                        st.error("ไม่มีข้อมูลในช่วงเวลาที่เลือกสำหรับการฝึกโมเดล")
                    else:
                        # Train and predict
                        with st.spinner("กำลังพยากรณ์..."):
                            results = train_and_predict(
                                target_data=training_data_lr,
                                upstream_data=upstream_df_lr if use_nearby_lr and use_upstream_lr else None,
                                downstream_data=downstream_df_lr if use_nearby_lr and use_downstream_lr else None,
                                use_upstream=use_upstream_lr,
                                use_downstream=use_downstream_lr
                            )

                            if results is not None:
                                # Display accuracy metrics
                                st.header("ผลค่าความแม่นยำ (Linear Regression)", divider='gray')
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(label="Train R²", value=f"{results['train_r2']:.4f}")
                                with col2:
                                    st.metric(label="Test R²", value=f"{results['test_r2']:.4f}")
                                with col3:
                                    st.metric(label="Test MSE", value=f"{results['test_mse']:.4f}")
                                
                                col4, col5, col6 = st.columns(3)
                                with col4:
                                    st.metric(label="Train MAE", value=f"{results['train_mae']:.4f}")
                                with col5:
                                    st.metric(label="Test MAE", value=f"{results['test_mae']:.4f}")
                                with col6:
                                    st.metric(label="Train MSE", value=f"{results['train_mse']:.4f}")

                                # Plot Actual vs Predicted for Train set
                                st.subheader("ค่าจริง vs ค่าพยากรณ์ (Train Set)")
                                train_df = pd.DataFrame({
                                    'Actual': results['y_train'],
                                    'Predicted': results['y_pred_train']
                                }, index=results['X_train'].index)
                                fig_train = px.line(train_df, title='Train Set: Actual vs Predicted')
                                st.plotly_chart(fig_train, use_container_width=True)

                                # Plot Actual vs Predicted for Test set
                                st.subheader("ค่าจริง vs ค่าพยากรณ์ (Test Set)")
                                test_df = pd.DataFrame({
                                    'Actual': results['y_test'],
                                    'Predicted': results['y_pred_test']
                                }, index=results['X_test'].index)
                                fig_test = px.line(test_df, title='Test Set: Actual vs Predicted')
                                st.plotly_chart(fig_test, use_container_width=True)

                                # Forecast future data
                                st.header("การพยากรณ์ในอนาคต", divider='gray')
                                forecast_start_date_actual_lr = training_end_datetime_lr + pd.Timedelta(minutes=15)
                                forecasted_data_lr = forecast_with_linear_regression_multi(
                                    data=target_df_lr.set_index('datetime'),
                                    forecast_start_date=forecast_start_date_actual_lr,
                                    forecast_days=forecast_days_lr,
                                    upstream_data=upstream_df_lr.set_index('datetime') if use_nearby_lr and use_upstream_lr else None,
                                    downstream_data=downstream_df_lr.set_index('datetime') if use_nearby_lr and use_downstream_lr else None,
                                    delay_hours_up=st.session_state.get('delay_hours_up_lr', 0),
                                    delay_hours_down=st.session_state.get('delay_hours_down_lr', 0)
                                )

                                if not forecasted_data_lr.empty:
                                    st.header("กราฟข้อมูลพร้อมการพยากรณ์ (Linear Regression)")
                                    plot_data_combined_LR_stations(
                                        target_df_lr.set_index('datetime'), 
                                        forecasted_data_lr, 
                                        upstream_df_lr.set_index('datetime') if use_nearby_lr and use_upstream_lr else None, 
                                        downstream_df_lr.set_index('datetime') if use_nearby_lr and use_downstream_lr else None, 
                                        label='สถานีที่ต้องการทำนาย'
                                    )
                                    st.markdown("---")  # Divider

                                    # Prepare data for accuracy metrics
                                    filled_lr = forecasted_data_lr.reset_index().rename(columns={'index': 'datetime'})
                                    filled_lr['wl_up2'] = filled_lr['wl_up']
                                    filled_lr.drop(columns=['wl_up'], inplace=True)

                                    mse_lr, mae_lr, r2_lr, merged_data_lr = calculate_accuracy_metrics_linear(
                                        original=target_df_lr.reset_index(),
                                        filled=filled_lr
                                    )

                                    if mse_lr is not None:
                                        st.header("ตารางข้อมูลเปรียบเทียบ")
                                        comparison_table_lr = create_comparison_table_streamlit(forecasted_data_lr.reset_index(), merged_data_lr)
                                        st.dataframe(comparison_table_lr, use_container_width=True)
                                        
                                        st.header("ผลค่าความแม่นยำ")
                                        st.markdown("---")  # Divider
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(label="Mean Squared Error (MSE)", value=f"{mse_lr:.4f}")
                                        with col2:
                                            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_lr:.4f}")
                                        with col3:
                                            st.metric(label="R-squared (R²)", value=f"{r2_lr:.4f}")
                                else:
                                    st.error("ไม่สามารถพยากรณ์ได้เนื่องจากข้อมูลไม่เพียงพอ")
                            st.markdown("---")  # Divider
        else:
            st.error("กรุณาอัปโหลดไฟล์สำหรับ Linear Regression")

























