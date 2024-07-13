import kfp
from kfp import Client

# Define the client for the Kubeflow Pipelines service
client = Client()

# Define pipeline file path
pipeline_file = 'pipelineV4.yaml'

# Upload the pipeline to Kubeflow
pipeline_name = 'sarima-pipeline'
description = 'A pipeline to run SARIMA model functions.'

# If the pipeline already exists, it will be replaced
try:
    client.upload_pipeline(pipeline_package_path=pipeline_file, pipeline_name=pipeline_name)
    print(f'Pipeline {pipeline_name} uploaded successfully.')
except Exception as e:
    print(f'Error uploading pipeline: {e}')

# Define the experiment name and run name
experiment_name = 'sarima-experiment'
run_name = 'sarima-run'

# Create an experiment
experiment = client.create_experiment(name=experiment_name)

# Run the pipeline
run = client.run_pipeline(
    experiment_id=experiment.id,
    job_name=run_name,
    pipeline_package_path=pipeline_file
)

print(f'Pipeline run started: {run}')
