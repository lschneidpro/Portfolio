# Explainable Drog Breed Classifier Webapp
[Link to Webapp](https://xaidog-app.herokuapp.com)

## Description

This is a personal machine learning engineering project as part of the [Udacity Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t). The project consists of deploying an interpretable dog breed classifier web app. After uploading a dog picture, top 5 probable breeds (across 133 breeds) will be returned as long as the original picture with a heatmap of the picture area that contributed to the classification decision. The classification is performed with DenseNet-121 while explanations are performed with Grad-CAM.

More information about the computer vision development can be found in the *notebook* folder while the web app can be found in the *webapp* folder.

## Getting Started

### Prerequisites

This app has been tested for python 3.6 on linux.

```
pip install requirements.txt
```

### Runninf

```
python app.py
```

## Deployment

The app can be deployed locally. This app has been deployed on Heroku (see link at the top)

## Know Issues

Increase memory consumption after few images preprocessing causing crash on the heroku instance.

## Built With

* [Dash](https://plotly.com/dash/) - The web framework used
* [Pytorch](https://pytorch.org) - The deep learning framework used

## Screenshot
The web app looks like as follows:

![Alt text](screenshot.jpg?raw=true "Optional Title")
