import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import kfp
from kfp import dsl
from kfp.dsl import component, Output, Model
import os

@component(base_image='python:3.12.0')
def train_and_evaluate_iris(model_path: Output[Model]):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Save model
    joblib.dump(model, model_path.path + '/iris_model.pkl')

# Define your pipeline
@dsl.pipeline(
    name='Simple ML Pipeline',
    description='A simple ML pipeline example'
)
def simple_ml_pipeline():
    # Create a step in the pipeline that uses the component
    train_and_evaluate_task = train_and_evaluate_iris()

# Get the current working directory
cwd = os.getcwd()
yaml_file_path = os.path.join(cwd, 'simple_ml_pipeline.yaml')

if __name__ == '__main__':
    # Compile the pipeline into a .yaml file
    kfp.compiler.Compiler().compile(
        pipeline_func=simple_ml_pipeline,
        package_path=yaml_file_path
    )
