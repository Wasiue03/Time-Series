import kfp
from kfp import dsl

@dsl.pipeline(
    name='SARIMA Pipeline',
    description='A pipeline to run SARIMA model functions.'
)
def sarima_pipeline():
    # Load Data Step
    load_data_op = dsl.ContainerOp(
        name='Load Data',
        image='wasiue/load_data:latest',
        command=['python', 'sarima.py', 'load_data'],
        file_outputs={'data': '/app/data/metrics.csv'}
    )

    # Plot Data Step
    plot_data_op = dsl.ContainerOp(
        name='Seasonality',
        image='wasiue/seasonal_decomposition:latest',
        command=['python', 'sarima.py', 'seasonal_decomposition'],
        arguments=['--data', load_data_op.outputs['data']]
    )

    # ADF Test Step
    adf_test_op = dsl.ContainerOp(
        name='ADF Test',
        image='wasiue/test_stationarity:latest',
        command=['python', 'sarima.py', 'test_stationarity'],
        arguments=['--data', load_data_op.outputs['data']]
    )

    # Plot ACF and PACF Step
    plot_acf_pacf_op = dsl.ContainerOp(
        name='Plot ACF and PACF',
        image='wasiue/plot_acf_pacf:latest',
        command=['python', 'sarima.py', 'plot_acf_pacf'],
        arguments=['--data', load_data_op.outputs['data']]
    )

    # SARIMA Model and Forecast Step
    sarima_model_forecast_op = dsl.ContainerOp(
        name='SARIMA Model Forecast',
        image='wasiue/sarima_model_forecast:latest',
        command=['python', 'sarima.py', 'sarima_model_forecast'],
        arguments=['--data', load_data_op.outputs['data']]
    )

    # Difference Series Step
    difference_series_op = dsl.ContainerOp(
        name='Difference Series',
        image='wasiue/difference_series:latest',
        command=['python', 'sarima.py', 'difference_series'],
        arguments=['--data', load_data_op.outputs['data']]
    )

    # Main Execution Step
    main_op = dsl.ContainerOp(
        name='Main Execution',
        image='wasiue/main:latest',
        command=['python', 'sarima.py', 'main'],
        arguments=['--data', load_data_op.outputs['data']]
    )

# Compile the pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sarima_pipeline, 'pipelineV2.yaml')
