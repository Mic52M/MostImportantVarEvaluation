# MostImportantVarEvaluation

This probe analyzes datasets to identify the most important and potentially sensitive variables using mutual information and feature importance techniques. It helps ensure fairness in machine learning models by detecting attributes that could introduce bias or disproportionately affect model outcomes.

## Overview

The `MostImportantVarProbe` is designed for MLOps workflows to identify the most important variables in a dataset and assess their potential sensitivity. This probe helps ensure that the dataset is balanced and free from biases that could result in discrimination, aiming to detect privileged or underprivileged groups within the data. By analyzing the dataset's feature importance and mutual information, the probe evaluates whether certain attributes may disproportionately impact the model’s outcomes, guiding towards fairer and more balanced machine learning models.

## Core Functionality

### Feature Evaluation Process

The core of this probe revolves around evaluating the importance of variables within a dataset. The process is divided into several key steps, each contributing to the identification of potentially sensitive and important variables:

### 1. **Data Loading and Preparation**
The probe integrates with GitLab or GitHub to automatically retrieve datasets stored in CI/CD pipelines. After downloading the dataset artifact, the data is loaded and prepared for analysis. The preparation includes filtering out low-variance features and encoding categorical variables to ensure all features are in a suitable format for machine learning analysis.

### 2. **Low-Variance Filtering**
Features with extremely low variance (below a configurable threshold, typically 0.1) are removed from the dataset. This step ensures that only features with meaningful variability are included in the subsequent analysis. Features with low variance do not provide useful information to a machine learning model and are typically excluded from importance evaluation.

### 3. **Mutual Information Calculation**
The probe computes the **mutual information** (MI) between all pairs of variables. Mutual information measures the dependency between two variables:
- For **continuous variables**, the probe uses `mutual_info_regression` from scikit-learn.
- For **categorical variables**, it uses `mutual_info_classif`.

Mutual information is computed in parallel across the dataset to speed up the process. The result is an MI matrix that quantifies the amount of information each feature shares with the others. Higher MI values suggest stronger relationships between features, which can indicate potential sensitivity when such features are highly correlated with other variables or outcomes.

### 4. **Feature Importance Evaluation**
The probe applies a **Random Forest Classifier** to evaluate the relative importance of each feature based on two different criteria:
- **Impurity-based importance**: This measures how much each feature reduces the impurity (uncertainty) in the decision trees of the Random Forest model.
- **Permutation importance**: This is calculated by permuting each feature's values and measuring the impact on the model’s accuracy. A higher permutation importance score indicates that the feature is crucial for the model's performance.

The probe uses these two methods to create a ranked list of features based on their importance. Features with high scores in both metrics are considered most important for predicting the target outcome.

### 5. **Correlation and Sensitivity Analysis**
After calculating the mutual information and feature importance, the probe checks for features that might be sensitive based on:
- **High mutual information (MI)**: Features with MI values above the 99.5th percentile are considered sensitive because they share a large amount of information with other variables.
- **High correlation**: Features with a correlation coefficient above 0.95 (or another threshold) are flagged as potentially redundant and sensitive.

### 6. **Identifying Sensitive and Important Features**
The probe identifies **potentially sensitive attributes** by intersecting the high mutual information and high correlation features. These attributes are then cross-referenced with the top 20 most important features from the Random Forest model.

### 7. **Final Output: Sensitive and Important Features**
The most important and potentially sensitive features are output by the probe. These features may contribute disproportionately to the model's decisions and thus need further scrutiny to avoid introducing bias into the model. The probe produces a detailed result that includes:
- A list of **potentially sensitive attributes**.
- A list of **important sensitive attributes** that both have high importance and are flagged as sensitive.

This analysis helps identify whether certain groups or attributes are privileged in the dataset, ensuring that the model treats all variables equitably and avoids unintentional discrimination.

## Use Case in MLOps

This probe is integrated into MLOps pipelines to automate the analysis of datasets before training machine learning models. By identifying important and sensitive variables, the probe helps ensure that the dataset is fair and that the model's outcomes are balanced across different feature groups. This process helps mitigate the risk of privileged or underprivileged groups affecting the model’s predictions, leading to fairer and more responsible machine learning systems.

