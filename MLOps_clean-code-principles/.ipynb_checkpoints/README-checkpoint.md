# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project re-implements a jupyter notebook as a pyhton package implementing best practices production code: modular, documented, tested)

## Running Files
The code is expected to run unit tests and to generate logs, images and model files.
The code can be run locally as follows:
```console
pip install -r requirements.txt
cd src
python churn_script_logging_and_tests.py
```

The code can be run as a container as follows:
```console
docker build . -t app
docker run -it app
```
