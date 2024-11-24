# Midterm Project ML-Zoomcamp

This project is based on a binarized version of the California Housing Dataset in [OpenML](https://www.openml.org/search?type=data&status=active&id=45578&sort=runs).

The original target variable, median house value for California districts, expressed in hundreds of thousands of dollars ($100,000) was binarized in this dataset. See the following link for descriptio of the [original dataset](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html).

Unfortunately, the methodology for converting it into a binary variable was not provided in the dataset description.

This dataset describes the median value of a house in terms of the following characteristics: median income, housing median age, total rooms, total bedrooms, population, households, latitude, and longitude.

Based on these features, the goal is to classify the binary medianHouseValue.


## Dataset

This dataset can be downloaded via `sklearn` using the fetch_openml function. Please refer to training script or notebook.

## Project Structure

├── Dockerfile
|
├── Pipfile
├── Pipfile.lock
├── README.md
├── artifacts
│   ├── df_test.csv
│   └── model.bin
├── notebooks
│   └── notebook.ipynb
├── predict.py
├── score_results.py
└── train.py

## Project Setup

### Conda environment for running the notebook

conda create -n ml-zoomcamp python=3.11

conda install numpy pandas scikit-learn seaborn jupyter pyarrow pipenv


### Pipenv

pipenv install pandas==2.2.2 scikit-learn==1.5.1 flask gunicorn

### Docker
