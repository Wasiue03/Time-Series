
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


# Define the Kubeflow pipeline
@dsl.pipeline(
    name='SARIMA Pipeline',
    description='A pipeline to run SARIMA model functions.'
)
def sarima_pipeline():
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
    
    seasonal_op = dsl.ContainerOp(
        name='Seasonal Decomposition',
        image='wasiue/seasonal_decomposition:latest',
        command=['python', 'sarima.py', 'seasonal_decomposition'],
        
    ).after(load_data_op)
    
    adf_test_op = dsl.ContainerOp(
        name='ADF Test',
        image='wasiue/test_stationarity:latest',
        command=['python', 'sarima.py', 'test_stationarity'],
        
    ).after(seasonal_op)

    difference_series_op = dsl.ContainerOp(
        name='Difference Series',
        image='wasiue/difference_series:latest',
        command=['python', 'sarima.py', 'difference_series'],
        
    ).after(adf_test_op)

    plot_acf_pacf_op = dsl.ContainerOp(
        name='Plot ACF and PACF',
        image='wasiue/plot_acf_pacf:latest',
        command=['python', 'sarima.py', 'plot_acf_pacf'],
        
    ).after(difference_series_op)

    sarima_forecast = dsl.ContainerOp(
        name = 'Sarima Forecasting',
        image = 'wasiue/sarima_model:latest',
        command = ["python", "sarima.py", "sarima_model"]
    ).after(plot_acf_pacf_op)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sarima_pipeline, 'pipelineV7.yaml')

