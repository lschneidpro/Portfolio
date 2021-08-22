"""
A module to perform unit tests

@author: luca
Created on Sun Aug 22 19:08:30 2021
"""

import os
import logging

import churn_library as cl
import constants as cst

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
    '''
    try:
        df_raw = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_raw.shape[0] > 0
        assert df_raw.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df_raw = cl.import_data("./data/bank_data.csv")
        perform_eda(df_raw)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_eda: Something went wrong with the EDA plots")
        raise err

    try:
        assert os.path.isfile('./images/eda/churn_histogram.png')
        assert os.path.isfile('./images/eda/customer-age_histogram.png')
        assert os.path.isfile('./images/eda/marital-status_bar.png')
        assert os.path.isfile('./images/eda/churn_histogram.png')
        assert os.path.isfile('./images/eda/distplot.png')
        assert os.path.isfile('./images/eda/heatmap.png')
    except AssertionError as err:
        logging.error("Testing perform_eda: The plot has not been saved")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df_raw = cl.import_data("./data/bank_data.csv")
        df_encoded = encoder_helper(df_raw, cst.cat_columns, cst.response)
        logging.info("Testing encoder_helper: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err
    except Exception as err:
        logging.error(
            "Testing encoder_helper: Something went wrong with the encoder")
        raise err

    try:
        assert df_encoded.shape[0] == df_raw.shape[0]
        assert df_encoded.shape[1] == df_raw.shape[1]
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have correct rows and columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df_raw = cl.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_raw, cst.response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering: The file wasn't found")
        raise err
    except Exception as err:
        logging.error(
            """Testing perform_feature_engineering:
                Something went wrong with the feature engineering""")
        raise err

    try:
        assert x_test.shape[0] + x_train.shape[0] == df_raw.shape[0]
        assert y_test.shape[0] + y_train.shape[0] == df_raw.shape[0]
        assert x_test.shape[1] == len(cst.keep_cols)
        assert x_train.shape[1] == len(cst.keep_cols)
    except AssertionError as err:
        logging.error(
            """Testing perform_feature_engineering:
                The dataframes don't appear to have correct rows and columns""")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        df_raw = cl.import_data("./data/bank_data.csv")
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            df_raw, cst.response)
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The file wasn't found")
        raise err
    except Exception as err:
        logging.error(
            "Testing train_models: Something went wrong with training the models")
        raise err

    try:
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./models/rfc_model.pkl')
    except AssertionError as err:
        logging.error("Testing train_models: The model has not been saved")
        raise err
    try:
        assert os.path.isfile('./images/results/roc.png')
        assert os.path.isfile("./images/results/feature_importances.png")
        assert os.path.isfile('./images/results/lr.png')
        assert os.path.isfile('./images/results/rf.png')
    except AssertionError as err:
        logging.error("Testing train_models: The plot has not been saved")
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
