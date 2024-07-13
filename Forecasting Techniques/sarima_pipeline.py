import kfp
from kfp import dsl

@dsl.pipeline(
    name='SARIMA Pipeline',
    description='A pipeline to run SARIMA model functions.'
)
def sarima_pipeline():
    
    main_op = dsl.ContainerOp(
        name='Main Execution',
        image='wasiue/main:latest',
        command=['python', 'sarima.py', 'main'],
        file_outputs={'data': '/app/data/metrics.csv'}
    )

    
    load_data_op = dsl.ContainerOp(
        name='Load Data',
        image='wasiue/load_data:latest',
        command=['python', 'sarima.py', 'load_data']
    )

    load_data_op.after(main_op)
    
    plot_data_op = dsl.ContainerOp(
        name='Seasonality',
        image='wasiue/seasonal_decomposition:latest',
        command=['python', 'sarima.py', 'seasonal_decomposition'],
        arguments=['--data', main_op.outputs['data']]
    )
    plot_data_op.after(load_data_op)

    
    adf_test_op = dsl.ContainerOp(
        name='ADF Test',
        image='wasiue/test_stationarity:latest',
        command=['python', 'sarima.py', 'test_stationarity'],
        arguments=['--data', main_op.outputs['data']]
    )
    adf_test_op.after(load_data_op)

    difference_series_op = dsl.ContainerOp(
        name='Difference Series',
        image='wasiue/difference_series:latest',
        command=['python', 'sarima.py', 'difference_series'],
        arguments=['--data', main_op.outputs['data']]
    )
    difference_series_op.after(adf_test_op)

    plot_acf_pacf_op = dsl.ContainerOp(
        name='Plot ACF and PACF',
        image='wasiue/plot_acf_pacf:latest',
        command=['python', 'sarima.py', 'plot_acf_pacf'],
        arguments=['--data', main_op.outputs['data']]
    )
    plot_acf_pacf_op.after(difference_series_op)

    
   
    sarima_model_forecast_op = dsl.ContainerOp(
        name='SARIMA Model Forecast',
        image='wasiue/sarima_model:latest',
        command=['python', 'sarima.py', 'sarima_model'],
        arguments=['--data', main_op.outputs['data']]
    )
    sarima_model_forecast_op.after(plot_acf_pacf_op)

    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sarima_pipeline, 'pipelineV4.yaml')
