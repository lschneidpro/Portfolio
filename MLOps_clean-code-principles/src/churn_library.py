"""
Created on Sun Aug 22 19:08:30 2021

@author: luca
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants as cst

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_raw = pd.read_csv(pth)
    return df_raw


def perform_eda(df_raw):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df_raw['Churn'] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    fig, axs = plt.subplots(figsize=cst.figsize)
    df_raw['Churn'].hist(ax=axs)
    fig.savefig('./images/eda/churn_histogram.png')

    fig, axs = plt.subplots(figsize=cst.figsize)
    df_raw['Customer_Age'].hist()
    fig.savefig('./images/eda/customer-age_histogram.png')

    fig, axs = plt.subplots(figsize=cst.figsize)
    df_raw.Marital_Status.value_counts('normalize').plot(kind='bar')
    fig.savefig('./images/eda/marital-status_bar.png')

    fig, axs = plt.subplots(figsize=cst.figsize)
    sns.distplot(df_raw['Total_Trans_Ct'], ax=axs)
    fig.savefig('./images/eda/churn_histogram.png')

    fig, axs = plt.subplots(figsize=cst.figsize)
    sns.distplot(df_raw['Total_Trans_Ct'], ax=axs)
    fig.savefig('./images/eda/distplot.png')

    fig, axs = plt.subplots(figsize=cst.figsize)
    sns.heatmap(
        df_raw.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2,
        ax=axs)
    fig.savefig('./images/eda/heatmap.png')


def encoder_helper(df_raw, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    df_raw[response] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    for cat in category_lst:
        groups = df_raw.groupby(cat).mean()[response]
        column_tag = cat + "_" + response
        df_raw[column_tag] = df_raw[cat].map(groups)
    return df_raw


def perform_feature_engineering(df_raw, response="Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    df_raw[response] = df_raw['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_raw = encoder_helper(df_raw, cst.cat_columns, response)
    y = df_raw[response]
    x = df_raw[cst.keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # logistic regression
    fig, axs = plt.subplots(figsize=(5, 5))
    axs.text(0.01, 1.1, str('Logistic Regression Train'),
             {'fontsize': 9}, fontproperties='monospace')
    axs.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 9}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axs.text(0.01, 0.5, str('Logistic Regression Test'),
             {'fontsize': 9}, fontproperties='monospace')
    axs.text(0.01, 0.07, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 9}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axs.axis('off')
    fig.savefig('./images/results/lr.png')

    # random forest
    fig, axs = plt.subplots(figsize=(5, 5))
    axs.text(0.01, 1.1, str('Random Forest Train'), {
             'fontsize': 9}, fontproperties='monospace')
    axs.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 9}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axs.text(0.01, 0.5, str('Random Forest Test'), {
             'fontsize': 9}, fontproperties='monospace')
    axs.text(0.01, 0.07, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 9}, fontproperties='monospace')  # approach improved by OP -> monospace!
    axs.axis('off')
    fig.savefig('./images/results/rf.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    fig, axs = plt.subplots(figsize=(20, 5))
    # Create plot title
    axs.set_title("Feature Importance")
    axs.set_ylabel('Importance')
    # Add bars
    axs.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    axs.set_xticks(range(x_data.shape[1]))
    axs.set_xticklabels(names, rotation=90)
    fig.tight_layout()
    fig.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # logistic regression
    lrc = LogisticRegression()
    lrc.fit(x_train, y_train)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)
    joblib.dump(lrc, './models/logistic_model.pkl')

    # random forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')

    fig, axs = plt.subplots(figsize=(15, 8))
    plot_roc_curve(lrc, x_test, y_test, ax=axs, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=axs, alpha=0.8)
    fig.savefig('./images/results/roc.png')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc.best_estimator_, x_train,
                            "./images/results/feature_importances.png")
