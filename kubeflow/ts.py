import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def load_data(data_path: str):
    
    sales_data = pd.read_csv('shampoo_sales.csv')
    sales_data.to_csv(data_path + '/shampoo_sales.csv', index=False)
    print(f"Shampoo sales data saved to {data_path}/shampoo_sales.csv")

def train_sarima(data_path: str):
    
    sales_data = pd.read_csv(data_path + '/shampoo_sales.csv', parse_dates=['Month'], index_col='Month')
   
    model = SARIMAX(sales_data['Sales'], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    
    model_fit.save(data_path + '/sarima_model.pkl')
    print("SARIMA model trained and saved.")

def train_arima(data_path: str):
    
    sales_data = pd.read_csv(data_path + '/shampoo_sales.csv', parse_dates=['Month'], index_col='Month')
    
    model = ARIMA(sales_data['Sales'], order=(1, 0, 1))
    model_fit = model.fit()
    
    model_fit.save(data_path + '/arima_model.pkl')
    print("ARIMA model trained and saved.")

def evaluate_models(data_path: str):
    
    sales_data = pd.read_csv(data_path + '/shampoo_sales.csv', parse_dates=['Month'], index_col='Month')
    
    sarima_model = SARIMAXResults.load(data_path + '/sarima_model.pkl')
    sarima_predictions = sarima_model.predict(start=len(sales_data), end=len(sales_data) + 11, dynamic=False)
    
    arima_model = ARIMAResults.load(data_path + '/arima_model.pkl')
    arima_predictions = arima_model.predict(start=len(sales_data), end=len(sales_data) + 11, dynamic=False)
    
    sarima_rmse = np.sqrt(mean_squared_error(sales_data['Sales'][-12:], sarima_predictions))
    arima_rmse = np.sqrt(mean_squared_error(sales_data['Sales'][-12:], arima_predictions))
    print(f"SARIMA Model RMSE: {sarima_rmse:.2f}")
    print(f"ARIMA Model RMSE: {arima_rmse:.2f}")
import kfp
import kfp.components as comp
from kfp.dsl import PipelineVolume


load_data_op = comp.create_component_from_func(
    load_data,
    base_image='python:3.7',
    packages_to_install=['pandas', 'numpy', 'statsmodels']
)

train_sarima_op = comp.create_component_from_func(
    train_sarima,
    base_image='python:3.7',
    packages_to_install=['pandas', 'numpy', 'statsmodels']
)

train_arima_op = comp.create_component_from_func(
    train_arima,
    base_image='python:3.7',
    packages_to_install=['pandas', 'numpy', 'statsmodels']
)

evaluate_models_op = comp.create_component_from_func(
    evaluate_models,
    base_image='python:3.7',
    packages_to_install=['pandas', 'numpy', 'statsmodels', 'scikit-learn']
)

# Define your pipeline
@kfp.dsl.pipeline(
    name='Shampoo Sales Forecasting Pipeline',
    description='Pipeline to perform sales forecasting using SARIMA and ARIMA models.'
)
def shampoo_sales_forecasting_pipeline(data_path: str):
    # Define the volume to persist data across pipeline steps
    data_volume = kfp.dsl.PipelineVolume(pvc=data_path)

    # Define pipeline steps using the created component ops
    load_data_task = load_data_op(data_path).add_pvolumes({data_path: data_volume})
    train_sarima_task = train_sarima_op(data_path).add_pvolumes({data_path: data_volume}).after(load_data_task)
    train_arima_task = train_arima_op(data_path).add_pvolumes({data_path: data_volume}).after(train_sarima_task)
    evaluate_models_task = evaluate_models_op(data_path).add_pvolumes({data_path: data_volume}).after(train_arima_task)

    # Set caching options for each step
    load_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_sarima_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_arima_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    evaluate_models_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

# Compile the pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=shampoo_sales_forecasting_pipeline,
    package_path='shampoo_sales_forecasting_pipeline.yaml'
)

# Submit the pipeline run
client = kfp.Client()
client.create_run_from_pipeline_func(
    shampoo_sales_forecasting_pipeline,
    arguments={'data_path': '/mnt/data'}
)
