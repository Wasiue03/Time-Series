# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_squared_error
# from kfp import dsl
# from kfp.dsl import component, Output

# @component(base_image='python:3.12.0', packages_to_install=['statsmodels', 'pandas', 'matplotlib', 'scikit-learn'])
# def sarima_forecast(data: str) -> Output[float]:
#     # Load data
#     df = pd.read_csv('data/metrics.csv', parse_dates=['time_stamp'])
#     df.set_index('time_stamp', inplace=True)
    
#     # Perform SARIMA forecasting
#     model = SARIMAX(df['value'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 24))
#     sarima_fit = model.fit(disp=False)
#     forecast = sarima_fit.forecast(steps=24)  # Adjust steps as needed
    
#     # Calculate RMSE
#     actual = df['value'][-24:].values  # Assuming forecasting last 24 hours
#     rmse = np.sqrt(mean_squared_error(actual, forecast))
    
#     # Plotting (optional)
#     plt.figure(figsize=(12, 6))
#     plt.plot(df.index[-24:], actual, label='Actual')
#     plt.plot(df.index[-24:], forecast, label='Forecast')
#     plt.title('SARIMA Forecast')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.savefig('/mnt/sarima_forecast_plot.png')  # Save plot to file
    
#     # Output RMSE as a float
#     return rmse

# from kfp import dsl

# # Define your pipeline
# @dsl.pipeline(
#     name='SARIMA Forecasting Pipeline',
#     description='Pipeline to perform SARIMA forecasting'
# )
# def sarima_pipeline(data_path: str):
#     sarima_task = sarima_forecast(data='data/metrics.csv')

# from kfp import compiler

# # Compile the pipeline into a .yaml file
# pipeline_file = 'sarima_pipeline.yaml'

# compiler.Compiler().compile(
#     pipeline_func=sarima_pipeline,
#     package_path=pipeline_file
# )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

def load_data(file_path):
    """
    Function to load and preprocess data.
    """
    df = pd.read_csv(file_path)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.groupby('time_stamp')['value'].sum()
    return df

def seasonal_decomposition(df):
    """
    Function to perform seasonal decomposition and save plot to file.
    """
    result_stl = seasonal_decompose(df, model='additive')
    fig = result_stl.plot()
    plt.title('Seasonal Decomposition')
    plt.savefig('/app/data/seasonal_decomposition.png')  # Save plot to file
    plt.close()  # Close plot to prevent showing in container output
    return result_stl

def test_stationarity(df):
    """
    Function to test stationarity using Augmented Dickey-Fuller test.
    """
    result = adfuller(df, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')


def difference_series(df, order=1):
    """
    Function to perform differencing on the series and save plot to file.
    """
    for i in range(order):
        df = df.diff().dropna()
    plt.figure(figsize=(12, 6))
    plt.plot(df)
    plt.title('Differenced Series')
    plt.savefig('/app/data/difference_series.png')  # Save plot to file
    plt.close()  # Close plot to prevent showing in container output
    return df

def plot_acf_pacf(series):
    """
    Function to plot ACF and PACF and save plots to file.
    """
    df = load_data('data/metrics.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(series, lags=48, ax=axes[0, 0])
    plot_pacf(series, lags=30, ax=axes[0, 1])
    plot_acf(df.diff().dropna(), lags=48, ax=axes[1, 0])
    plot_pacf(df.diff().dropna(), lags=30, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig('/app/data/acf_pacf.png')  # Save plot to file
    plt.close() 

def sarima_model(train, test):
    """
    Function to fit SARIMA model and forecast.
    """
    model = SARIMAX(train, order=(3, 1, 0), seasonal_order=(3, 1, 0, 7))
    sr_fit = model.fit()
    sr_fc = sr_fit.forecast(steps=len(test))
    sr_rmse = np.sqrt(mean_squared_error(test, sr_fc))
    
    plt.figure(figsize=(16, 4))
    plt.plot(test, label="Actual")
    plt.plot(sr_fc, label="Predicted")
    plt.title('SARIMA Forecast')
    plt.ylabel('% CPU Usage')
    plt.legend()
    plt.show()
    
    print(f"SARIMAX RMSE: {sr_rmse}")

def main():
    """
    Main function to orchestrate the workflow.
    """
    # Load data
    df = load_data('data/metrics.csv')
    
    # Plot original data
    plt.figure(figsize=(14, 4))
    plt.plot(df)
    plt.title('Hourly CPU Usage')
    plt.ylabel('% CPU Usage')
    plt.show()

    # Perform seasonal decomposition
    result_stl = seasonal_decomposition(df)
    
    # Detrend data
    dt_df = df - result_stl.trend
    dt_df.dropna(inplace=True)
    
    # Test stationarity
    test_stationarity(df)
    
    # Perform differencing
    df_diff = difference_series(df, order=1)
    
    # Plot ACF and PACF
    plot_acf_pacf(df)
    
    # Split data into train and test sets
    train = df.iloc[:2196]
    test = df.iloc[2196:]
    
    # Perform SARIMA modeling and forecasting
    sarima_model(train, test)

if __name__ == "__main__":
    main()


