# Phishing Website Detection

This project focuses on building a machine learning model to detect phishing websites. The primary goal is to create a robust classification model that can distinguish between legitimate and phishing URLs based on a set of features.

This repository contains a Jupyter Notebook (`script.ipynb`) that walks through the process of data loading, model exploration, training, and evaluation.

## Project Overview

The project follows these key steps:

1.  **Data Loading and Preparation**: The dataset is loaded from an ARFF file (`Training Dataset.arff`) and converted into a pandas DataFrame. The data is then split into training and testing sets.

2.  **Model Exploration**: Several classification models are evaluated to determine the best-performing one. The models tested include:
    *   Logistic Regression
    *   Random Forest
    *   Gradient Boosting
    *   Support Vector Machine (SVM)
    *   K-Nearest Neighbors (KNN)

3.  **Hyperparameter Tuning**: The best-performing model (Random Forest) is selected for hyperparameter tuning using `GridSearchCV` to find the optimal combination of parameters.

4.  **Pipeline Creation**: A full machine learning pipeline is built using `scikit-learn`'s `Pipeline` class. This encapsulates preprocessing (scaling) and classification, making the model easy to reuse and deploy.

5.  **Model Evaluation**: The final model is evaluated on the test set, and its performance is measured using metrics like precision, recall, and F1-score. The feature importances are also examined to understand which features are most influential in the model's predictions.

## Getting Started

To run the notebook and reproduce the results, you will need to have Python and the following libraries installed:

*   pandas
*   scikit-learn
*   scipy
*   joblib

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn scipy joblib
```

After installing the dependencies, you can run the `script.ipynb` notebook using Jupyter Notebook or JupyterLab.

## Future Work

While the current model performs well, there are several avenues for future improvement:

*   **Advanced Feature Engineering**: Explore creating new features from the existing ones to potentially improve model accuracy.
*   **Alternative Models**: Experiment with other advanced models like XGBoost, LightGBM, or neural networks.
*   **Deployment**: Deploy the trained model as a REST API using a web framework like Flask or FastAPI, allowing it to be integrated into other applications.
*   **Real-time Prediction**: Develop a browser extension or a web service that uses the model to classify URLs in real-time.
*   **Automated Retraining**: Set up a system to automatically retrain the model on new data to keep it up-to-date with the latest phishing techniques.