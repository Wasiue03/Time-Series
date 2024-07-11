# Use an official Python runtime as a parent image
FROM python:3.12.0

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and dependencies
COPY sarima.py /app/sarima.py

RUN mkdir /app/data

# Copy the dataset into the container
COPY data/metrics.csv /app/data/metrics.csv

# Install pandas, numpy, matplotlib, and any other dependencies required
RUN pip install pandas numpy statsmodels matplotlib scikit-learn

# Specify the command to run your function
CMD ["python", "sarima.py", "plot_acf_pacf"]
