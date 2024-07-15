# # import kfp
# # from kfp import dsl

# # @dsl.pipeline(
# #     name='SARIMA Pipeline',
# #     description='A pipeline to run SARIMA model functions.'
# # )
# # def sarima_pipeline():
    
# #     main_op = dsl.ContainerOp(
# #         name='Main Execution',
# #         image='wasiue/main:latest',
# #         command=['python', 'sarima.py', 'main'],
# #         file_outputs={'data': '/app/data/metrics.csv'}
# #     )

    
# #     load_data_op = dsl.ContainerOp(
# #         name='Load Data',
# #         image='wasiue/load_data:latest',
# #         command=['python', 'sarima.py', 'load_data']
# #     )

# #     load_data_op.after(main_op)
    
# #     plot_data_op = dsl.ContainerOp(
# #         name='Seasonality',
# #         image='wasiue/seasonal_decomposition:latest',
# #         command=['python', 'sarima.py', 'seasonal_decomposition'],
# #         arguments=['--data', main_op.outputs['data']]
# #     )
# #     plot_data_op.after(load_data_op)

    
# #     adf_test_op = dsl.ContainerOp(
# #         name='ADF Test',
# #         image='wasiue/test_stationarity:latest',
# #         command=['python', 'sarima.py', 'test_stationarity'],
# #         arguments=['--data', main_op.outputs['data']]
# #     )
# #     adf_test_op.after(plot_acf_pacf_op)

# #     difference_series_op = dsl.ContainerOp(
# #         name='Difference Series',
# #         image='wasiue/difference_series:latest',
# #         command=['python', 'sarima.py', 'difference_series'],
# #         arguments=['--data', main_op.outputs['data']]
# #     )
# #     difference_series_op.after(adf_test_op)

# #     plot_acf_pacf_op = dsl.ContainerOp(
# #         name='Plot ACF and PACF',
# #         image='wasiue/plot_acf_pacf:latest',
# #         command=['python', 'sarima.py', 'plot_acf_pacf'],
# #         arguments=['--data', main_op.outputs['data']]
# #     )
# #     plot_acf_pacf_op.after(difference_series_op)

    
   
# #     sarima_model_forecast_op = dsl.ContainerOp(
# #         name='SARIMA Model Forecast',
# #         image='wasiue/sarima_model:latest',
# #         command=['python', 'sarima.py', 'sarima_model'],
# #         arguments=['--data', main_op.outputs['data']]
# #     )
# #     sarima_model_forecast_op.after(plot_acf_pacf_op)

    
    

# # if __name__ == '__main__':
# #     kfp.compiler.Compiler().compile(sarima_pipeline, 'pipelineV4.yaml')


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
from kfp import dsl, components
import kfp

# Define your data preprocessing and SARIMA modeling functions
def load_data(file_path):
    """
    Function to load and preprocess data.
    """
    df = pd.read_csv('data/metrics.csv')
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.groupby('time_stamp')['value'].sum()
    return df

def seasonal_decomposition(df, model='additive'):
    """
    Function to perform seasonal decomposition and save plot to file.
    """
    result_stl = seasonal_decompose(df,model='additive')
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
    plt.savefig('/app/data/sarima_forecast.png')  # Save plot to file
    plt.close()
    
    print(f"SARIMAX RMSE: {sr_rmse}")

# Define Kubeflow components
sarima_forecast_op = components.create_component_from_func(
    sarima_model,
    base_image='python:3.12.0',
    packages_to_install=['pandas', 'numpy', 'matplotlib', 'statsmodels', 'scikit-learn']
)

# Define the Kubeflow pipeline
@dsl.pipeline(
    name='SARIMA Pipeline',
    description='A pipeline to run SARIMA model functions.'
)
def sarima_pipeline(data_path: str):
    main_execution = dsl.ContainerOp(
        name='Main Execution',
        image='wasiue/main:latest',
        command=['python', 'sarima.py', 'main'],
        file_outputs={'data': '/app/data/metrics.csv'}
    )
    
    load_data_op = dsl.ContainerOp(
        name='Load Data',
        image='wasiue/load_data:latest',
        command=['python', 'sarima.py', 'load_data'],
        file_outputs={'data': '/app/data/metrics.csv'}
    ).after(main_execution)
    
    # seasonal_op = dsl.ContainerOp(
    #     name='Seasonal Decomposition',
    #     image='wasiue/seasonal_decomposition:latest',
    #     command=['python', 'sarima.py', 'seasonal_decomposition'],
    #     arguments=['--data', load_data_op.outputs['data']]
    # ).after(load_data_op)
    
    adf_test_op = dsl.ContainerOp(
        name='ADF Test',
        image='wasiue/test_stationarity:latest',
        command=['python', 'sarima.py', 'test_stationarity'],
        arguments=['--data', load_data_op.outputs['data']]
    ).after(load_data_op)

    difference_series_op = dsl.ContainerOp(
        name='Difference Series',
        image='wasiue/difference_series:latest',
        command=['python', 'sarima.py', 'difference_series'],
        arguments=['--data', load_data_op.outputs['data']]
    ).after(adf_test_op)

    plot_acf_pacf_op = dsl.ContainerOp(
        name='Plot ACF and PACF',
        image='wasiue/plot_acf_pacf:latest',
        command=['python', 'sarima.py', 'plot_acf_pacf'],
        arguments=['--data', load_data_op.outputs['data']]
    ).after(difference_series_op)

    sarima_forecast = sarima_forecast_op(
        train=load_data_op.outputs['data'],
        test=load_data_op.outputs['data']
    ).after(plot_acf_pacf_op)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sarima_pipeline, 'pipelineV6.yaml')



# import kfp
# from kfp import dsl
# from kfp.components import create_component_from_func

# # Replace these with your actual Kubeflow details
# KUBEFLOW_HOST = "98.66.205.26"
# USERNAME = "mlops@kubar.io"
# PASSWORD = "Kubar.kubeFlow.IO.55"
# NAMESPACE = "kubeflow-user-example-com"

# client = kfp.Client(
#     host=KUBEFLOW_HOST,
#     username=USERNAME,
#     password=PASSWORD,
#     namespace=NAMESPACE
# )

# # Define components
# @create_component_from_func
# def load_data() -> str:
#     import pandas as pd
#     df = pd.read_csv('data/metrics.csv')
#     df = df[['time_stamp', 'value']]
#     df['time_stamp'] = pd.to_datetime(df['time_stamp'])
#     df = df.groupby('time_stamp')['value'].sum()
#     df.to_csv('/mnt/data/df.csv')
#     return '/mnt/data/df.csv'

# @create_component_from_func
# def preprocess_data(data_path: str) -> str:
#     import pandas as pd
#     df = pd.read_csv(data_path, index_col='time_stamp', parse_dates=True)
#     df_diff = df.diff().dropna()
#     df_diff.to_csv('/mnt/data/df_diff.csv')
#     return '/mnt/data/df_diff.csv'

# @create_component_from_func
# def stl_decompose(data_path: str) -> str:
#     import pandas as pd
#     from statsmodels.tsa.seasonal import seasonal_decompose
#     df = pd.read_csv(data_path, index_col='time_stamp', parse_dates=True)
#     result_stl = seasonal_decompose(df, model='additive')
#     trend = result_stl.trend.dropna()
#     trend.to_csv('/mnt/data/trend.csv')
#     return '/mnt/data/trend.csv'

# @create_component_from_func
# def train_sarima_model(data_path: str) -> str:
#     import pandas as pd
#     from statsmodels.tsa.statespace.sarimax import SARIMAX
#     df = pd.read_csv(data_path, index_col='time_stamp', parse_dates=True)
#     model = SARIMAX(df, order=(3, 1, 0), seasonal_order=(3, 1, 0, 7))
#     result = model.fit()
#     result.save('/mnt/data/sarima_model.pkl')
#     return '/mnt/data/sarima_model.pkl'

# @dsl.pipeline(
#     name='SARIMA Pipeline',
#     description='A pipeline to train a SARIMA model on CPU usage data'
# )
# def sarima_pipeline():
#     load_data_task = load_data()
#     preprocess_data_task = preprocess_data(load_data_task.output)
#     stl_decompose_task = stl_decompose(preprocess_data_task.output)
#     train_sarima_model_task = train_sarima_model(stl_decompose_task.output)

# # Run the pipeline
# if __name__ == '__main__':
#     client.create_run_from_pipeline_func(sarima_pipeline, arguments={}, namespace=NAMESPACE)
