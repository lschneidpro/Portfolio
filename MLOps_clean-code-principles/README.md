# Predict Customer Churn

 **Predict Customer Churn** Project from the ML DevOps Engineer Nanodegree of Udacity

## Project Description
This is a project to identify credit card customers that are most likely to churn. The project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, logs and tested).

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

## Ideas for improvement
- Re-organize each script to work as a class.
- Update functions to move constants to their own constants.py file, which can then be passed to the necessary functions, rather than being created as variables within functions.
- Work towards pylint score of 10/10.
- Add dependencies and libraries (or dockerfile) to README.md
- Add requirements.txt with needed dependencies and libraries for simple install.
